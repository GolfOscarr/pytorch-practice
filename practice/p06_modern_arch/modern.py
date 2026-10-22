"""Modern LLM building blocks (Llama-style): RMSNorm, RoPE, SwiGLU, GQA with a KV cache.

Build the pieces a modern decoder block is made of, then prove the fast inference path
(KV cache, one token at a time) gives the same answers as the simple path (one full forward).

What to build
1. `RMSNorm(dim, eps)`: y = x / sqrt(mean(x^2 over the last dim) + eps) * weight.
   No mean subtraction, no bias. Must match `torch.nn.RMSNorm(dim, eps=eps)`.
2. RoPE (rotary position embedding), ROTATE-HALF convention (as in HF Llama / GPT-NeoX):
   - `precompute_rope(head_dim, max_seq_len, theta)` returns `(cos, sin)`, each (max_seq_len, D).
     inv_freq[i] = theta ** (-2i / D) for i in 0 .. D/2-1; angle[p, i] = p * inv_freq[i];
     cos = cos(cat([angle, angle], -1)), sin likewise.
   - `apply_rope(x, cos, sin, offset)`: x is (B, H, T, D); token t is at absolute position
     offset + t. With x = [x1, x2] split into halves, rotate_half(x) = [-x2, x1] and
     out = x * cos + rotate_half(x) * sin. So dim i is paired with dim i + D/2 (NOT with
     dim i+1 as in the interleaved / complex-number convention of the original Llama code).
3. `SwiGLU(dim, hidden_dim)`: w2(silu(w1(x)) * w3(x)), three bias-free Linears.
4. `KVCache`: preallocated k and v buffers of shape (B, n_kv_heads, max_seq_len, D).
   `update(k, v, start_pos)` writes the new (B, n_kv_heads, T, D) keys/values at positions
   start_pos .. start_pos+T-1 and returns the valid prefix (positions 0 .. start_pos+T-1).
5. `GroupedQueryAttention(dim, n_heads, n_kv_heads)`: n_heads query heads share n_kv_heads
   key/value heads (n_kv_heads == n_heads is MHA, n_kv_heads == 1 is MQA).
   forward(x, cache=None, start_pos=0) with x of shape (B, T, dim), returns (B, T, dim).

Rules
- Public attribute names are part of the API (tests use them): `RMSNorm.weight`;
  `SwiGLU.w1/w2/w3`; `GroupedQueryAttention.wq/wk/wv/wo` (bias=False), `.n_heads`,
  `.n_kv_heads`, `.head_dim`; `KVCache.k/.v`.
- Projection layout is head-major: `wq(x).view(B, T, n_heads, head_dim)` splits heads (same
  for wk/wv with n_kv_heads). Query head h uses kv head h // (n_heads // n_kv_heads).
- Raise ValueError for: odd head_dim in RoPE, dim % n_heads != 0, n_heads % n_kv_heads != 0,
  RoPE positions past the table, and writes past the end of the cache.
- Use F.scaled_dot_product_attention or write softmax(q k^T / sqrt(D)) v yourself.

Hints (the tricky parts)
- Apply RoPE to q and k at their ABSOLUTE positions (offset=start_pos) BEFORE writing k into
  the cache. Then cached keys are already rotated and never need touching again.
- GQA: cache only n_kv_heads heads (that is the memory win). Expand to n_heads right before
  the attention matmul with `repeat_interleave(n_rep, dim=1)`, so query head h uses kv head
  h // n_rep.
- Causal mask with a cache: T queries attend to S = start_pos + T keys. Query row i is at
  position start_pos + i and may see keys j <= start_pos + i. A plain lower-triangular (T, T)
  mask is only right when start_pos == 0. Careful: SDPA's `is_causal=True` aligns its mask to
  the top-left, which is wrong whenever S > T (single-token decode, chunked prefill).
- Without a cache, keys are just the current T tokens, so the plain causal mask applies.
- After attention, (B, H, T, D) -> transpose -> (B, T, H, D) is non-contiguous: use reshape
  (or .contiguous().view) to get (B, T, H*D).

Run:
    uv run pytest practice/p06_modern_arch
    uv run pytest practice/p06_modern_arch -rA   # list each test
"""

import torch
import torch.nn.functional as F  # noqa: F401  (you will likely need it)
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm with a learnable per-feature scale (`weight`, init ones)."""

    weight: nn.Parameter

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """
        Args:
            dim: size of the last (normalized) dimension.
            eps: added to the mean square before the square root, for stability.
        """
        super().__init__()
        raise NotImplementedError("Your code here")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim).

        Returns:
            Same shape as x: x / sqrt(mean(x^2, -1) + eps) * weight.
        """
        raise NotImplementedError("Your code here")


def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotate-half RoPE tables.

    Args:
        head_dim: per-head dimension D (must be even).
        max_seq_len: number of positions to precompute.
        theta: RoPE base.
        device: where to create the tables.

    Returns:
        (cos, sin), each float32 of shape (max_seq_len, head_dim), where row p holds
        cos/sin of cat([p * inv_freq, p * inv_freq]) and inv_freq[i] = theta ** (-2i / D).

    Raises:
        ValueError: if head_dim is odd.
    """
    raise NotImplementedError("Your code here")


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    """Rotate x by its absolute positions offset .. offset+T-1 (rotate-half convention).

    Args:
        x: (B, H, T, D).
        cos, sin: tables from `precompute_rope`, (max_seq_len, D).
        offset: absolute position of x[..., 0, :].

    Returns:
        Tensor like x: x * cos[pos] + rotate_half(x) * sin[pos], rotate_half([x1, x2]) = [-x2, x1].

    Raises:
        ValueError: if offset < 0 or offset + T > max_seq_len.
    """
    raise NotImplementedError("Your code here")


class SwiGLU(nn.Module):
    """Gated FFN: w2(silu(w1(x)) * w3(x)); w1, w3: dim->hidden_dim, w2: hidden_dim->dim, no bias."""

    w1: nn.Linear
    w2: nn.Linear
    w3: nn.Linear

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        raise NotImplementedError("Your code here")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim).

        Returns:
            (..., dim).
        """
        raise NotImplementedError("Your code here")


class KVCache:
    """Preallocated key/value cache for incremental decoding.

    Attributes:
        k, v: (batch_size, n_kv_heads, max_seq_len, head_dim) buffers, zero-initialized.
    """

    k: torch.Tensor
    v: torch.Tensor

    def __init__(
        self,
        batch_size: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        raise NotImplementedError("Your code here")

    def update(
        self, k: torch.Tensor, v: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new keys/values in place and return everything cached so far.

        Args:
            k, v: (B, n_kv_heads, T, head_dim) with B <= batch_size.
            start_pos: absolute position of the first new token.

        Returns:
            (k_all, v_all), each (B, n_kv_heads, start_pos + T, head_dim), i.e. positions
            0 .. start_pos+T-1.

        Raises:
            ValueError: if start_pos < 0 or start_pos + T > max_seq_len.
        """
        raise NotImplementedError("Your code here")


class GroupedQueryAttention(nn.Module):
    """Causal grouped-query self-attention with RoPE and an optional KV cache."""

    n_heads: int
    n_kv_heads: int
    head_dim: int
    wq: nn.Linear
    wk: nn.Linear
    wv: nn.Linear
    wo: nn.Linear

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 256,
        rope_theta: float = 10000.0,
    ) -> None:
        """
        Args:
            dim: model width; head_dim = dim // n_heads.
            n_heads: number of query heads.
            n_kv_heads: number of key/value heads (divides n_heads).
            max_seq_len: RoPE table length (register cos/sin as buffers so .to(device) moves them).
            rope_theta: RoPE base.

        Raises:
            ValueError: if dim % n_heads != 0 or n_heads % n_kv_heads != 0.
        """
        super().__init__()
        raise NotImplementedError("Your code here")

    def forward(
        self, x: torch.Tensor, cache: KVCache | None = None, start_pos: int = 0
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim), tokens at absolute positions start_pos .. start_pos+T-1.
            cache: if given, new k/v are written into it and attention covers all cached
                positions 0 .. start_pos+T-1. If None, attention covers only these T tokens.
            start_pos: absolute position of x[:, 0].

        Returns:
            (B, T, dim).
        """
        raise NotImplementedError("Your code here")
