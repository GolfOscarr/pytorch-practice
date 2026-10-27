"""Reference solution for r08: batched generation + async inference service.

Same public API as pr.py. Lines that fix a planted bug carry `# FIX(Bn)` comments; see REVIEW.md.
Production extras applied here: SamplingParams/ServiceConfig validation in __post_init__,
inference_mode + eval/train restore, graceful stop() that fails queued and in-flight requests,
cancelled futures skipped, worker survives a failed batch.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

KVCache = list[tuple[Tensor, Tensor]]


# ----------------------------------------------------------------------------- model


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        past: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (z.view(b, t, self.n_heads, self.head_dim).transpose(1, 2) for z in (q, k, v))
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        s = k.size(2)
        causal = torch.ones(t, s, dtype=torch.bool, device=x.device).tril(diagonal=s - t)
        allowed = causal[None, None] & attention_mask[:, None, None, :].bool()
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attn = self.dropout(scores.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return self.proj(out), (k, v)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: Tensor, attention_mask: Tensor, past: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        h, kv = self.attn(self.ln1(x), attention_mask, past)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, kv


class TinyCausalLM(nn.Module):
    """A small GPT-style decoder with learned positions and a KV cache."""

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, dropout) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: KVCache | None = None,
    ) -> tuple[Tensor, KVCache]:
        """Return `(logits [B, T, V], new_cache)`.

        `attention_mask` is `[B, past_len + T]` with 1 for real tokens; `position_ids` is `[B, T]`.
        """
        b, t = input_ids.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(2)
        if attention_mask is None:
            attention_mask = torch.ones(b, past_len + t, dtype=torch.long, device=input_ids.device)
        if position_ids is None:
            positions = torch.arange(past_len, past_len + t, device=input_ids.device)
            position_ids = positions.expand(b, t)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(position_ids))
        new_cache: KVCache = []
        for i, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[i]
            x, kv = block(x, attention_mask, past)
            new_cache.append(kv)
        return self.lm_head(self.ln_f(x)), new_cache


# ----------------------------------------------------------------------------- sampling


@dataclass(frozen=True)
class SamplingParams:
    """Decoding knobs. `temperature=0` means greedy decoding."""

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0

    def __post_init__(self) -> None:
        # Production: reject bad configs at construction, not deep inside a batch.
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


def apply_top_k(logits: Tensor, k: int) -> Tensor:
    """Keep the `k` largest logits per row, set the rest to -inf."""
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def apply_top_p(logits: Tensor, top_p: float) -> Tensor:
    """Nucleus filtering: keep the smallest set of top tokens whose total probability >= top_p."""
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    # FIX(B4): drop a token only if the mass *before* it already reaches top_p (exclusive
    # cumsum). This keeps the token that crosses the threshold and always keeps the top token.
    mass_before = probs.cumsum(dim=-1) - probs
    remove = mass_before >= top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter(-1, sorted_idx, sorted_logits)


def sample_next_token(
    logits: Tensor, params: SamplingParams, generator: torch.Generator | None = None
) -> Tensor:
    """Pick one token id per row from `[B, V]` logits."""
    logits = logits.float()
    if params.top_k is not None:
        logits = apply_top_k(logits, params.top_k)
    if params.temperature == 0.0:
        # FIX(B3): greedy before any division; logits / 0 turns every positive logit into +inf
        # and argmax then returns the first +inf, not the best token.
        return logits.argmax(dim=-1)
    logits = logits / params.temperature
    if params.top_p < 1.0:
        logits = apply_top_p(logits, params.top_p)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


# ----------------------------------------------------------------------------- generation


def pad_prompts(
    prompts: list[list[int]], pad_token_id: int, device: torch.device | str = "cpu"
) -> tuple[Tensor, Tensor]:
    """Pad a ragged batch into `(input_ids, attention_mask)`, both `[B, max_len]`.

    Prompts are LEFT-padded so every row's last real token sits in the final column, which is
    where the next token is read from and where new tokens are appended.
    """
    max_len = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros(len(prompts), max_len, dtype=torch.long)
    for i, p in enumerate(prompts):
        # FIX(B1): left-pad. Right padding puts pads between the prompt and the generated tokens,
        # and logits[:, -1] would be read at a pad position for every shorter prompt.
        ids[i, max_len - len(p) :] = torch.tensor(p, dtype=torch.long)
        mask[i, max_len - len(p) :] = 1
    return ids.to(device), mask.to(device)


@torch.inference_mode()  # FIX(B5): no autograd graph / activation memory during decoding
def generate(
    model: TinyCausalLM,
    prompts: list[list[int]],
    max_new_tokens: int | list[int],
    params: SamplingParams | None = None,
    pad_token_id: int = 0,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """Generate up to `max_new_tokens` tokens for each prompt in one batched, KV-cached loop.

    `max_new_tokens` is one budget for every row or a list with one budget per prompt.
    Returns only the new tokens per prompt, truncated after `eos_token_id` (inclusive).
    """
    if not prompts or any(len(p) == 0 for p in prompts):
        raise ValueError("prompts must be a non-empty list of non-empty token lists")
    if isinstance(max_new_tokens, int):
        budgets = [max_new_tokens] * len(prompts)
    else:
        budgets = list(max_new_tokens)
    if len(budgets) != len(prompts):
        raise ValueError("max_new_tokens needs one budget per prompt")
    if any(n < 1 for n in budgets):
        raise ValueError("max_new_tokens must be >= 1")
    if any(len(p) + n > model.max_seq_len for p, n in zip(prompts, budgets, strict=True)):
        raise ValueError(f"prompt length + max_new_tokens exceeds max_seq_len={model.max_seq_len}")
    params = params or SamplingParams()
    device = next(model.parameters()).device

    was_training = model.training
    model.eval()  # FIX(B5): dropout off while decoding; restored in `finally`
    try:
        input_ids, attention_mask = pad_prompts(prompts, pad_token_id, device)
        # FIX(B2): positions count real tokens only, so a left-padded row starts at 0 exactly
        # like it would unbatched. Pads get a dummy position 0 (they are masked anyway).
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        logits, cache = model(input_ids, attention_mask, position_ids)
        next_logits = logits[:, -1, :]

        budget = torch.tensor(budgets, device=device)
        finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)
        steps: list[Tensor] = []
        for step in range(max(budgets)):
            next_tokens = sample_next_token(next_logits, params, generator)
            next_tokens = next_tokens.masked_fill(finished, pad_token_id)
            if eos_token_id is not None:
                finished |= next_tokens == eos_token_id
            # FIX(B9): a row is finished once it has its own budget. Finished rows append a
            # masked token, so their position stops advancing and never runs past max_seq_len.
            finished |= budget <= step + 1
            steps.append(next_tokens)
            if bool(finished.all()):
                break
            attention_mask = torch.cat([attention_mask, (~finished)[:, None].long()], dim=1)
            # FIX(B2): next position = number of real tokens before the new one, per row.
            position_ids = attention_mask.sum(dim=-1, keepdim=True) - 1
            logits, cache = model(next_tokens[:, None], attention_mask, position_ids, cache)
            next_logits = logits[:, -1, :]
    finally:
        model.train(was_training)

    rows: list[list[int]] = torch.stack(steps, dim=1).tolist()
    rows = [r[:n] for r, n in zip(rows, budgets, strict=True)]
    if eos_token_id is not None:
        rows = [r[: r.index(eos_token_id) + 1] if eos_token_id in r else r for r in rows]
    return rows


# ----------------------------------------------------------------------------- service


@dataclass(frozen=True)
class GenerationRequest:
    prompt_ids: list[int]
    max_new_tokens: int = 16
    timeout_s: float | None = None


@dataclass(frozen=True)
class ServiceConfig:
    max_batch_size: int = 8
    max_wait_ms: float = 5.0
    max_new_tokens_limit: int = 256
    sampling: SamplingParams = field(default_factory=SamplingParams)
    pad_token_id: int = 0
    eos_token_id: int | None = None
    default_timeout_s: float | None = 30.0

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")


@dataclass
class _Pending:
    request: GenerationRequest
    future: asyncio.Future[list[int]]
    enqueued_at: float = field(default_factory=time.perf_counter)


class InferenceService:
    """Micro-batching front end: concurrent `generate()` calls share one forward pass.

    Usage::

        async with InferenceService(model, ServiceConfig(max_batch_size=8)) as svc:
            tokens = await svc.generate(GenerationRequest(prompt_ids=[1, 2, 3]))
    """

    def __init__(
        self, model: TinyCausalLM, config: ServiceConfig | None = None, seed: int = 0
    ) -> None:
        self.model = model.eval()
        self.config = config or ServiceConfig()
        self._generator = torch.Generator(device=next(model.parameters()).device)
        self._generator.manual_seed(seed)
        self._queue: asyncio.Queue[_Pending] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._in_flight: list[_Pending] = []  # taken off the queue by the worker, not answered

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="inference-worker")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        # Production: fail every request the cancelled worker was holding (collecting or running)
        # and everything still queued, instead of leaving callers waiting forever.
        stranded, self._in_flight = self._in_flight, []
        while not self._queue.empty():
            stranded.append(self._queue.get_nowait())
        for pending in stranded:
            if not pending.future.done():
                pending.future.set_exception(RuntimeError("inference service stopped"))

    async def __aenter__(self) -> InferenceService:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def _validate(self, request: GenerationRequest) -> None:
        prompt = request.prompt_ids
        if not prompt:
            raise ValueError("prompt_ids must be non-empty")
        if any(not 0 <= t < self.model.vocab_size for t in prompt):
            raise ValueError("prompt_ids contains out-of-vocabulary token ids")
        if not 1 <= request.max_new_tokens <= self.config.max_new_tokens_limit:
            raise ValueError(f"max_new_tokens must be in [1, {self.config.max_new_tokens_limit}]")
        if len(prompt) + request.max_new_tokens > self.model.max_seq_len:
            raise ValueError(
                f"prompt length + max_new_tokens exceeds max_seq_len={self.model.max_seq_len}"
            )

    async def generate(self, request: GenerationRequest) -> list[int]:
        """Queue one request and wait for its tokens (raises TimeoutError past the deadline)."""
        if self._worker is None:
            raise RuntimeError("InferenceService is not running; use `async with` or start()")
        self._validate(request)
        future: asyncio.Future[list[int]] = asyncio.get_running_loop().create_future()
        await self._queue.put(_Pending(request, future))
        timeout = request.timeout_s
        if timeout is None:
            timeout = self.config.default_timeout_s
        return await asyncio.wait_for(future, timeout)

    async def _collect_batch(self) -> list[_Pending]:
        # The same list object is `_in_flight`, so stop() sees requests collected so far.
        batch = self._in_flight = [await self._queue.get()]
        deadline = time.perf_counter() + self.config.max_wait_ms / 1000
        while len(batch) < self.config.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), remaining))
            except TimeoutError:
                break
        return batch

    def _run_batch(self, batch: list[_Pending]) -> list[list[int]]:
        cfg = self.config
        # FIX(B7): keep prompts in the same order as `batch` so outputs zip back to the right
        # futures. (Sorting by length buys nothing here: every row is padded to the max anyway.)
        prompts = [p.request.prompt_ids for p in batch]
        # FIX(B9): give each row its own budget. Decoding every row for max(max_new_tokens)
        # steps pushes a long-prompt row past max_seq_len although it passed `_validate`.
        budgets = [p.request.max_new_tokens for p in batch]
        return generate(
            self.model,
            prompts,
            budgets,
            cfg.sampling,
            pad_token_id=cfg.pad_token_id,
            eos_token_id=cfg.eos_token_id,
            generator=self._generator,
        )

    async def _run(self) -> None:
        while True:
            batch = await self._collect_batch()
            batch = [p for p in batch if not p.future.done()]  # drop timed-out / cancelled
            if not batch:
                continue
            try:
                # FIX(B6): run the blocking forward passes in a worker thread so the event loop
                # keeps accepting requests, firing timeouts and serving other coroutines.
                outputs = await asyncio.to_thread(self._run_batch, batch)
            except Exception as exc:
                logger.exception("inference batch of %d failed", len(batch))
                # FIX(B8): propagate the failure to every caller in the batch; otherwise they
                # wait until their timeout (or forever when timeout is None).
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(exc)
                continue
            for p, tokens in zip(batch, outputs, strict=True):
                if not p.future.done():
                    p.future.set_result(tokens)
