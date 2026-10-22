import inspect
import logging
import re
import time
from collections.abc import Callable

import pytest
import torch
from torch import nn

from practice.p03_decorators.decorators import (
    Registry,
    Timer,
    eval_mode,
    inference,
    retry,
    timed,
    validate_shapes,
)

SLEEP = 0.01  # seconds; tiny, but well above perf_counter resolution


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.bn(self.lin(x)))


# ---------------------------------------------------------------------------
# timed
# ---------------------------------------------------------------------------


def test_timed_bare_preserves_metadata_and_records_duration() -> None:
    def add(a: int, b: int = 1) -> int:
        """Add two ints."""
        return a + b

    wrapped = timed(add)
    assert wrapped.__name__ == "add"
    assert wrapped.__doc__ == "Add two ints."
    assert wrapped.__wrapped__ is add
    assert inspect.signature(wrapped) == inspect.signature(add)
    assert wrapped.last_duration is None  # nothing timed yet

    assert wrapped(2, b=3) == 5
    assert isinstance(wrapped.last_duration, float)
    assert wrapped.last_duration >= 0.0


def test_timed_logs_to_given_or_module_logger(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("p03.custom")

    @timed(logger=logger)
    def nap() -> str:
        time.sleep(SLEEP)
        return "done"

    with caplog.at_level(logging.INFO, logger="p03.custom"):
        assert nap() == "done"

    records = [r for r in caplog.records if r.name == "p03.custom"]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert "nap" in records[0].getMessage()
    assert "ms" in records[0].getMessage()
    assert nap.last_duration is not None and nap.last_duration >= SLEEP * 0.9

    # Bare form: defaults to the decorated function's module logger.
    def work() -> None:
        pass

    caplog.clear()
    with caplog.at_level(logging.INFO):
        timed(work)()
    assert [r.name for r in caplog.records] == [work.__module__]
    assert work.__qualname__ in caplog.records[0].getMessage()


def test_timed_sync_synchronizes_device(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("mps"))

    timed(lambda: None)()
    assert calls == []  # sync defaults to False

    timed(sync=True)(lambda: None)()
    assert calls == ["mps", "mps"]  # once before starting, once before stopping the clock


def test_timed_records_duration_when_function_raises() -> None:
    @timed
    def boom() -> None:
        time.sleep(SLEEP)
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        boom()
    assert boom.last_duration is not None and boom.last_duration >= SLEEP * 0.9


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------


def test_retry_succeeds_after_transient_failures() -> None:
    calls = 0

    @retry(times=3, exceptions=(ConnectionError,))
    def flaky(x: int) -> int:
        """Fails twice, then works."""
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError(f"attempt {calls}")
        return x * 2

    assert flaky(21) == 42
    assert calls == 3
    assert flaky.__name__ == "flaky"
    assert flaky.__doc__ == "Fails twice, then works."


def test_retry_reraises_last_exception_after_final_attempt() -> None:
    calls = 0

    @retry(times=4)
    def always_fails() -> None:
        nonlocal calls
        calls += 1
        raise TimeoutError(f"attempt {calls}")

    with pytest.raises(TimeoutError, match="attempt 4"):  # the LAST one, not the first
        always_fails()
    assert calls == 4


def test_retry_does_not_retry_unlisted_exception() -> None:
    calls = 0

    @retry(times=5, exceptions=(ConnectionError,))
    def buggy() -> None:
        nonlocal calls
        calls += 1
        raise KeyError("a real bug")

    with pytest.raises(KeyError):
        buggy()
    assert calls == 1


def test_retry_validates_times_and_sleeps_between_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError):
        retry(times=0)
    with pytest.raises(ValueError):
        retry(times=2, delay=-1.0)

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    @retry(times=3, delay=0.5)
    def fails() -> None:
        raise OSError("disk")

    with pytest.raises(OSError):
        fails()
    assert sleeps == [0.5, 0.5]  # between attempts only, not after the last


# ---------------------------------------------------------------------------
# eval_mode / inference
# ---------------------------------------------------------------------------


def test_inference_no_grad_eval_and_restores_even_on_exception() -> None:
    torch.manual_seed(0)
    model = TinyNet().train()
    seen: dict[str, bool] = {}

    @inference
    def predict(m: TinyNet, x: torch.Tensor) -> torch.Tensor:
        seen["grad"] = torch.is_grad_enabled()
        seen["training"] = any(sub.training for sub in m.modules())
        return m(x)

    out = predict(model, torch.randn(3, 4))
    assert seen == {"grad": False, "training": False}
    assert not out.requires_grad
    assert torch.is_grad_enabled()
    assert all(sub.training for sub in model.modules())
    assert predict.__name__ == "predict"

    @inference
    def broken(m: TinyNet) -> None:
        raise RuntimeError("bad batch")

    with pytest.raises(RuntimeError, match="bad batch"):
        broken(model)
    assert torch.is_grad_enabled()
    assert all(sub.training for sub in model.modules())


def test_eval_mode_restores_mixed_submodule_modes() -> None:
    model = TinyNet().train()
    model.bn.eval()  # e.g. a frozen BatchNorm, deliberately kept in eval

    with pytest.raises(ValueError), eval_mode(model) as m:
        assert m is model
        assert not any(sub.training for sub in model.modules())
        raise ValueError("inside")

    assert model.training and model.lin.training and model.drop.training
    assert not model.bn.training  # must NOT be flipped back to train


# ---------------------------------------------------------------------------
# validate_shapes
# ---------------------------------------------------------------------------


def make_masked_mean() -> Callable[..., torch.Tensor]:
    @validate_shapes(x="B,T,D", mask="B,T", w="D,2")
    def fn(
        x: torch.Tensor, mask: torch.Tensor | None = None, w: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Masked mean over T."""
        return x.mean(dim=1)

    return fn


def test_validate_shapes_accepts_consistent_args() -> None:
    masked_mean = make_masked_mean()
    x, mask, w = torch.zeros(2, 5, 3), torch.ones(2, 5), torch.zeros(3, 2)
    assert masked_mean(x, mask).shape == (2, 3)  # positional
    assert masked_mean(x=x, mask=mask, w=w).shape == (2, 3)  # keyword
    assert masked_mean(x).shape == (2, 3)  # optional args left as None
    assert masked_mean.__name__ == "fn" and masked_mean.__doc__ == "Masked mean over T."


def test_validate_shapes_rank_mismatch_message() -> None:
    masked_mean = make_masked_mean()
    with pytest.raises(ValueError, match=r"mask") as info:
        masked_mean(torch.zeros(2, 5, 3), torch.ones(2, 5, 1))
    assert "(2, 5, 1)" in str(info.value)  # actual shape is reported

    with pytest.raises(ValueError, match=r"\bw\b") as info:
        masked_mean(torch.zeros(2, 5, 3), w=torch.zeros(3, 4))  # literal dim 2 violated
    assert "(3, 4)" in str(info.value)

    with pytest.raises(TypeError):
        masked_mean([[1.0]])


def test_validate_shapes_named_dim_mismatch_message() -> None:
    masked_mean = make_masked_mean()
    with pytest.raises(ValueError) as info:
        masked_mean(torch.zeros(2, 5, 3), mask=torch.ones(2, 4))
    msg = str(info.value)
    assert "'T'" in msg and re.search(r"\bx\b", msg) and re.search(r"\bmask\b", msg)
    assert "5" in msg and "4" in msg

    with pytest.raises(ValueError, match="'D'"):
        masked_mean(torch.zeros(2, 5, 3), w=torch.zeros(7, 2))


def test_validate_shapes_rejects_unknown_parameter_at_decoration() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        return x

    with pytest.raises(ValueError, match=r"\by\b"):
        validate_shapes(x="B", y="B")(f)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_register_get_build() -> None:
    models = Registry("model")

    @models.register("mlp")
    class MLP(nn.Module):
        def __init__(self, width: int = 8) -> None:
            super().__init__()
            self.width = width

    @models.register("identity")
    def make_identity() -> nn.Module:
        return nn.Identity()

    assert MLP.__name__ == "MLP"  # register returns the class unchanged
    assert models.get("mlp") is MLP
    built = models.build("mlp", width=16)
    assert isinstance(built, MLP) and built.width == 16
    assert isinstance(models.build("identity"), nn.Identity)
    assert "mlp" in models and "cnn" not in models
    assert models.names() == ["identity", "mlp"]


def test_registry_duplicate_and_unknown_names() -> None:
    models = Registry("model")
    models.register("mlp")(nn.Linear)

    with pytest.raises(ValueError, match="mlp"):
        models.register("mlp")(nn.Conv1d)
    assert models.get("mlp") is nn.Linear  # the original entry survives

    with pytest.raises(KeyError) as info:
        models.get("mpl")  # typo
    assert "mpl" in str(info.value) and "mlp" in str(info.value)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------


def test_timer_sets_elapsed_and_propagates_exceptions() -> None:
    t = Timer()
    assert t.elapsed is None
    with t as bound:
        assert bound is t
        time.sleep(SLEEP)
    assert t.elapsed is not None
    assert SLEEP * 0.9 <= t.elapsed < 1.0

    # Exceptions propagate (not swallowed) and elapsed is still recorded.
    with pytest.raises(ZeroDivisionError), Timer() as t2:
        time.sleep(SLEEP)
        _ = 1 / 0
    assert t2.elapsed is not None and t2.elapsed >= SLEEP * 0.9
