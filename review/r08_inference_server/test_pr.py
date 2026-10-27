import asyncio
import time

import pytest
import torch
from torch import Tensor

from review.r08_inference_server.pr import (
    GenerationRequest,
    InferenceService,
    KVCache,
    SamplingParams,
    ServiceConfig,
    TinyCausalLM,
    apply_top_k,
    apply_top_p,
    generate,
    pad_prompts,
    sample_next_token,
)

# Everything runs on CPU: small, deterministic, and the async tests measure wall-clock gaps.
GREEDY = SamplingParams(top_k=1)  # top-1 sampling always picks the most likely token
PROMPTS = [[5, 9, 3, 7, 2], [11, 4], [8, 1, 6, 13, 2, 10, 3]]  # ragged, not sorted by length


def make_model(seed: int = 0, dropout: float = 0.0) -> TinyCausalLM:
    torch.manual_seed(seed)
    return TinyCausalLM(vocab_size=32, d_model=32, n_heads=4, n_layers=2, dropout=dropout).eval()


def reference_greedy(model: TinyCausalLM, prompt: list[int], n: int) -> list[int]:
    """Plain loop: re-run the full sequence every step, no padding, no cache."""
    seq = list(prompt)
    with torch.no_grad():
        for _ in range(n):
            logits, _ = model(torch.tensor([seq]))
            seq.append(int(logits[0, -1].argmax()))
    return seq[len(prompt) :]


class SlowModel(TinyCausalLM):
    """Each forward pass blocks the calling thread, like a real GPU-bound model would."""

    delay_s = 0.05

    def forward(self, *args, **kwargs) -> tuple[Tensor, KVCache]:  # type: ignore[no-untyped-def]
        time.sleep(self.delay_s)
        return super().forward(*args, **kwargs)


class FailingModel(TinyCausalLM):
    def forward(self, *args, **kwargs) -> tuple[Tensor, KVCache]:  # type: ignore[no-untyped-def]
        raise RuntimeError("CUDA out of memory (simulated)")


class GradSpyModel(TinyCausalLM):
    def __init__(self) -> None:
        super().__init__(vocab_size=32)
        self.grad_enabled: list[bool] = []

    def forward(self, *args, **kwargs) -> tuple[Tensor, KVCache]:  # type: ignore[no-untyped-def]
        self.grad_enabled.append(torch.is_grad_enabled())
        return super().forward(*args, **kwargs)


# ----------------------------------------------------------------------------- model / padding


def test_kv_cache_forward_matches_full_forward() -> None:
    model = make_model()
    ids = torch.tensor([[3, 1, 4, 1, 5, 9]])
    full, _ = model(ids)
    _, cache = model(ids[:, :4])
    step, _ = model(ids[:, 4:], past_key_values=cache)
    torch.testing.assert_close(step, full[:, 4:], rtol=1e-5, atol=1e-5)


def test_padded_batch_puts_every_prompts_last_token_in_final_column() -> None:
    ids, mask = pad_prompts([[1, 2, 3], [7]], pad_token_id=0)
    assert ids.shape == mask.shape == (2, 3)
    assert ids[:, -1].tolist() == [3, 7]
    assert mask.sum(dim=1).tolist() == [3, 1]
    assert mask[:, -1].tolist() == [1, 1]


# ----------------------------------------------------------------------------- sampling


def test_top_k_keeps_only_k_largest() -> None:
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    out = apply_top_k(logits, 2)
    assert torch.isfinite(out).tolist() == [[False, True, True, False]]


def test_top_p_keeps_smallest_set_reaching_p() -> None:
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    out = apply_top_p(probs.log(), top_p=0.6)
    # 0.5 alone is < 0.6, so the nucleus is {0.5, 0.3}.
    assert torch.isfinite(out).tolist() == [[True, True, False, False]]


def test_top_p_always_keeps_most_likely_token() -> None:
    probs = torch.tensor([[0.05, 0.9, 0.05]])
    out = apply_top_p(probs.log(), top_p=0.5)
    assert torch.isfinite(out).tolist() == [[False, True, False]]
    token = sample_next_token(probs.log(), SamplingParams(top_p=0.5))
    assert token.tolist() == [1]


def test_top_p_preserves_original_token_order() -> None:
    logits = torch.tensor([[0.0, 2.0, 1.0, 3.0]])
    out = apply_top_p(logits, top_p=1.0 - 1e-6)
    torch.testing.assert_close(out, logits)


def test_temperature_zero_is_greedy() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0], [0.5, -1.0, 0.25]])
    token = sample_next_token(logits, SamplingParams(temperature=0.0))
    assert token.tolist() == [1, 0]


def test_low_temperature_concentrates_on_argmax() -> None:
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([[1.0, 3.0, 2.0]]).expand(200, 3)
    tokens = sample_next_token(logits, SamplingParams(temperature=0.05), generator=g)
    assert (tokens == 1).all()


# ----------------------------------------------------------------------------- generate


def test_generate_returns_requested_number_of_tokens() -> None:
    out = generate(make_model(), PROMPTS, max_new_tokens=4, params=GREEDY)
    assert [len(r) for r in out] == [4, 4, 4]
    assert all(0 <= t < 32 for r in out for t in r)


def test_single_prompt_generation_matches_reference_loop() -> None:
    model = make_model()
    out = generate(model, [PROMPTS[0]], max_new_tokens=6, params=GREEDY)
    assert out == [reference_greedy(model, PROMPTS[0], 6)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batched_generation_matches_each_prompt_alone(seed: int) -> None:
    model = make_model(seed)
    batched = generate(model, PROMPTS, max_new_tokens=6, params=GREEDY)
    expected = [reference_greedy(model, p, 6) for p in PROMPTS]
    assert batched == expected


def test_generation_stops_after_eos() -> None:
    model = make_model()
    first = reference_greedy(model, PROMPTS[1], 1)[0]
    out = generate(model, PROMPTS, max_new_tokens=5, params=GREEDY, eos_token_id=first)
    assert out[1] == [first]


def test_generate_is_deterministic_even_if_model_left_in_train_mode() -> None:
    model = make_model(dropout=0.5).train()
    a = generate(model, PROMPTS, max_new_tokens=6, params=GREEDY)
    b = generate(model, PROMPTS, max_new_tokens=6, params=GREEDY)
    assert a == b == [reference_greedy(model.eval(), p, 6) for p in PROMPTS]


def test_generate_restores_training_flag() -> None:
    model = make_model().train()
    generate(model, PROMPTS, max_new_tokens=2, params=GREEDY)
    assert model.training


def test_generate_does_not_track_gradients() -> None:
    model = GradSpyModel()
    generate(model, PROMPTS, max_new_tokens=3, params=GREEDY)
    assert model.grad_enabled and not any(model.grad_enabled)


# ----------------------------------------------------------------------------- service


def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def test_service_rejects_invalid_requests() -> None:
    async def main() -> None:
        async with InferenceService(make_model(), ServiceConfig(sampling=GREEDY)) as svc:
            for bad in [
                GenerationRequest(prompt_ids=[]),
                GenerationRequest(prompt_ids=[1, 99]),
                GenerationRequest(prompt_ids=[1, 2], max_new_tokens=0),
                GenerationRequest(prompt_ids=[1] * 60, max_new_tokens=10),
            ]:
                with pytest.raises(ValueError):
                    await svc.generate(bad)

    run(main())


def test_service_requires_start() -> None:
    async def main() -> None:
        svc = InferenceService(make_model())
        with pytest.raises(RuntimeError):
            await svc.generate(GenerationRequest(prompt_ids=[1]))

    run(main())


def test_concurrent_requests_are_batched_together() -> None:
    model = make_model()
    calls: list[int] = []
    original = model.forward

    def spy(input_ids, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(input_ids.shape[0])
        return original(input_ids, *args, **kwargs)

    model.forward = spy  # type: ignore[method-assign]

    async def main() -> None:
        cfg = ServiceConfig(max_batch_size=8, max_wait_ms=50, sampling=GREEDY)
        async with InferenceService(model, cfg) as svc:
            reqs = [GenerationRequest(prompt_ids=p, max_new_tokens=3) for p in PROMPTS]
            await asyncio.gather(*(svc.generate(r) for r in reqs))

    run(main())
    assert calls[0] == 3  # the prefill already holds all three requests
    assert len(calls) <= 3  # one prefill + at most two decode steps: no second batch


def test_each_caller_gets_its_own_completion() -> None:
    model = make_model()
    expected = [reference_greedy(model, p, n) for p, n in zip(PROMPTS, [3, 5, 4], strict=True)]

    async def main() -> list[list[int]]:
        cfg = ServiceConfig(max_batch_size=8, max_wait_ms=50, sampling=GREEDY)
        async with InferenceService(model, cfg) as svc:
            reqs = [
                GenerationRequest(prompt_ids=p, max_new_tokens=n)
                for p, n in zip(PROMPTS, [3, 5, 4], strict=True)
            ]
            return list(await asyncio.gather(*(svc.generate(r) for r in reqs)))

    assert run(main()) == expected


def test_event_loop_stays_responsive_during_generation() -> None:
    torch.manual_seed(0)
    model = SlowModel(vocab_size=32).eval()

    async def main() -> float:
        gaps: list[float] = []
        done = asyncio.Event()

        async def heartbeat() -> None:
            last = time.perf_counter()
            while not done.is_set():
                await asyncio.sleep(0.005)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        cfg = ServiceConfig(max_wait_ms=1, sampling=GREEDY)
        async with InferenceService(model, cfg) as svc:
            hb = asyncio.create_task(heartbeat())
            await asyncio.sleep(0.02)
            await svc.generate(GenerationRequest(prompt_ids=[1, 2, 3], max_new_tokens=5))
            done.set()
            await hb
        return max(gaps)

    # 6 forward passes x 50 ms: a blocked loop would show a gap of ~300 ms.
    assert run(main()) < 0.15


def test_slow_request_times_out_and_service_keeps_working() -> None:
    torch.manual_seed(0)
    model = SlowModel(vocab_size=32).eval()

    async def main() -> list[int]:
        cfg = ServiceConfig(max_wait_ms=1, sampling=GREEDY)
        async with InferenceService(model, cfg) as svc:
            with pytest.raises(TimeoutError):
                await svc.generate(
                    GenerationRequest(prompt_ids=[1, 2], max_new_tokens=10, timeout_s=0.1)
                )
            return await svc.generate(GenerationRequest(prompt_ids=[3], max_new_tokens=1))

    assert len(run(main())) == 1


def test_concurrent_requests_that_each_fit_are_all_served() -> None:
    model = make_model()  # max_seq_len=64
    specs = [([7] * 60, 2), ([3, 5], 40)]  # long prompt + short budget, short prompt + long budget
    expected = [reference_greedy(model, p, n) for p, n in specs]

    async def main() -> list[list[int]]:
        cfg = ServiceConfig(max_batch_size=8, max_wait_ms=50, sampling=GREEDY)
        async with InferenceService(model, cfg) as svc:
            reqs = [GenerationRequest(prompt_ids=p, max_new_tokens=n) for p, n in specs]
            return list(
                await asyncio.wait_for(asyncio.gather(*(svc.generate(r) for r in reqs)), 5.0)
            )

    assert run(main()) == expected


def test_stop_fails_in_flight_requests_instead_of_hanging() -> None:
    torch.manual_seed(0)
    model = SlowModel(vocab_size=32).eval()

    async def main() -> None:
        cfg = ServiceConfig(max_wait_ms=1, sampling=GREEDY, default_timeout_s=None)
        svc = InferenceService(model, cfg)
        await svc.start()
        caller = asyncio.create_task(
            svc.generate(GenerationRequest(prompt_ids=[1, 2], max_new_tokens=10))
        )
        await asyncio.sleep(0.1)  # the batch is now running
        await svc.stop()
        with pytest.raises(RuntimeError, match="stopped"):
            await asyncio.wait_for(caller, 2.0)

    run(main())


def test_model_failure_is_raised_to_callers() -> None:
    torch.manual_seed(0)
    model = FailingModel(vocab_size=32).eval()

    async def main() -> None:
        async with InferenceService(model, ServiceConfig(sampling=GREEDY)) as svc:
            reqs = [GenerationRequest(prompt_ids=p, max_new_tokens=2) for p in PROMPTS[:2]]
            results = await asyncio.wait_for(
                asyncio.gather(*(svc.generate(r) for r in reqs), return_exceptions=True),
                timeout=2.0,
            )
            assert all(isinstance(r, RuntimeError) for r in results), results

    run(main())


def test_service_recovers_after_a_failed_batch() -> None:
    model = make_model()
    fail_next = [True]
    original = model.forward

    def flaky(*args, **kwargs):  # type: ignore[no-untyped-def]
        if fail_next[0]:
            fail_next[0] = False
            raise RuntimeError("transient failure")
        return original(*args, **kwargs)

    model.forward = flaky  # type: ignore[method-assign]

    async def main() -> list[int]:
        async with InferenceService(model, ServiceConfig(sampling=GREEDY)) as svc:
            with pytest.raises(RuntimeError, match="transient"):
                await asyncio.wait_for(
                    svc.generate(GenerationRequest(prompt_ids=[1, 2], max_new_tokens=2)), 2.0
                )
            return await asyncio.wait_for(
                svc.generate(GenerationRequest(prompt_ids=[1, 2], max_new_tokens=2)), 2.0
            )

    assert run(main()) == reference_greedy(model, [1, 2], 2)
