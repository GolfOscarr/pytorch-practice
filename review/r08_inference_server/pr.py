"""Add batched generation + an async inference service.

## Summary
First cut of the serving path for our small decoder models. Instead of one forward pass per
request, concurrent requests are micro-batched into a single KV-cached `generate()` call. No new
dependencies: plain torch + asyncio, so the HTTP layer can be whatever we pick later.

## Changes
- `TinyCausalLM`: small GPT-style decoder (learned positions, pre-LN) with a KV cache and an
  attention mask, so it doubles as a fast fixture for serving tests.
- Sampling utilities: `SamplingParams` (temperature / top-k / top-p), `apply_top_k`,
  `apply_top_p`, `sample_next_token`. `temperature=0` gives greedy decoding.
- `generate(model, prompts, max_new_tokens, ...)`: pads a ragged batch, runs one prefill, then
  decodes token by token reusing the cache. Supports per-row EOS early stop.
- `InferenceService`: `async def generate(request)` enqueues onto an `asyncio.Queue`; a single
  worker drains up to `max_batch_size` requests (waiting at most `max_wait_ms` for stragglers)
  and runs them as one batch. Requests are validated up front and honor a per-request timeout.
  Prompts are grouped by length inside a batch to keep padding contiguous.

## Testing
- KV-cache forward matches a full forward.
- Generation shapes / EOS handling, sampling filters on hand-made distributions.
- Service: validation errors, batching of concurrent requests, per-request timeouts.
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


def apply_top_k(logits: Tensor, k: int) -> Tensor:
    """Keep the `k` largest logits per row, set the rest to -inf."""
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def apply_top_p(logits: Tensor, top_p: float) -> Tensor:
    """Nucleus filtering: keep the smallest set of top tokens whose total probability >= top_p."""
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter(-1, sorted_idx, sorted_logits)


def sample_next_token(
    logits: Tensor, params: SamplingParams, generator: torch.Generator | None = None
) -> Tensor:
    """Pick one token id per row from `[B, V]` logits."""
    logits = logits.float() / params.temperature
    if params.top_k is not None:
        logits = apply_top_k(logits, params.top_k)
    if params.temperature == 0.0:
        return logits.argmax(dim=-1)
    if params.top_p < 1.0:
        logits = apply_top_p(logits, params.top_p)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


# ----------------------------------------------------------------------------- generation


def pad_prompts(
    prompts: list[list[int]], pad_token_id: int, device: torch.device | str = "cpu"
) -> tuple[Tensor, Tensor]:
    """Pad a ragged batch into `(input_ids, attention_mask)`, both `[B, max_len]`.

    The mask is 1 on real tokens and 0 on padding.
    """
    max_len = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros(len(prompts), max_len, dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, : len(p)] = torch.tensor(p, dtype=torch.long)
        mask[i, : len(p)] = 1
    return ids.to(device), mask.to(device)


def generate(
    model: TinyCausalLM,
    prompts: list[list[int]],
    max_new_tokens: int,
    params: SamplingParams | None = None,
    pad_token_id: int = 0,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """Generate up to `max_new_tokens` tokens for each prompt in one batched, KV-cached loop.

    Returns only the new tokens per prompt, truncated after `eos_token_id` (inclusive).
    """
    if not prompts or any(len(p) == 0 for p in prompts):
        raise ValueError("prompts must be a non-empty list of non-empty token lists")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    params = params or SamplingParams()
    device = next(model.parameters()).device

    input_ids, attention_mask = pad_prompts(prompts, pad_token_id, device)
    batch, prompt_len = input_ids.shape
    position_ids = torch.arange(prompt_len, device=device).expand(batch, prompt_len)
    logits, cache = model(input_ids, attention_mask, position_ids)
    next_logits = logits[:, -1, :]

    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    steps: list[Tensor] = []
    for step in range(max_new_tokens):
        next_tokens = sample_next_token(next_logits, params, generator)
        if eos_token_id is not None:
            next_tokens = next_tokens.masked_fill(finished, pad_token_id)
            finished |= next_tokens == eos_token_id
        steps.append(next_tokens)
        if step == max_new_tokens - 1 or bool(finished.all()):
            break
        attention_mask = torch.cat(
            [attention_mask, torch.ones(batch, 1, dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        position_ids = torch.full((batch, 1), prompt_len + step, device=device)
        logits, cache = model(next_tokens[:, None], attention_mask, position_ids, cache)
        next_logits = logits[:, -1, :]

    rows: list[list[int]] = torch.stack(steps, dim=1).tolist()
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

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="inference-worker")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

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
        batch = [await self._queue.get()]
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
        prompts = sorted((p.request.prompt_ids for p in batch), key=len)
        max_new = max(p.request.max_new_tokens for p in batch)
        outputs = generate(
            self.model,
            prompts,
            max_new,
            cfg.sampling,
            pad_token_id=cfg.pad_token_id,
            eos_token_id=cfg.eos_token_id,
            generator=self._generator,
        )
        return [out[: p.request.max_new_tokens] for p, out in zip(batch, outputs, strict=True)]

    async def _run(self) -> None:
        while True:
            batch = await self._collect_batch()
            try:
                outputs = self._run_batch(batch)
            except Exception:
                logger.exception("inference batch of %d failed", len(batch))
                continue
            for p, tokens in zip(batch, outputs, strict=True):
                if not p.future.done():
                    p.future.set_result(tokens)
