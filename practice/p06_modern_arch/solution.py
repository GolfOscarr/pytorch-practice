"""Reference solution for p06_modern_arch. Try modern.py yourself before reading this."""

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    weight: nn.Parameter

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No mean subtraction and no bias (unlike LayerNorm): just rescale by the root mean square.
        # rsqrt(...) is one fused op instead of 1 / sqrt(...).
        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms_inv * self.weight


def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    # One frequency per pair of dims: theta^(-2i/D), i = 0 .. D/2-1. High freq first.
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, inv_freq)  # (max_seq_len, D/2): angle = position * freq
    # Rotate-half pairs dim i with dim i + D/2, so both halves share the same angle.
    emb = torch.cat([angles, angles], dim=-1)  # (max_seq_len, D)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # [x1, x2] -> [-x2, x1]: a 90-degree rotation of every (x1[i], x2[i]) pair.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    seq_len = x.shape[-2]
    if offset < 0 or offset + seq_len > cos.shape[0]:
        raise ValueError(
            f"positions [{offset}, {offset + seq_len}) out of range for table of {cos.shape[0]}"
        )
    # Rows for absolute positions offset .. offset+T-1; (T, D) broadcasts over (B, H).
    c = cos[offset : offset + seq_len].to(x.dtype)
    s = sin[offset : offset + seq_len].to(x.dtype)
    # 2-D rotation of each pair: (a, b) -> (a cos - b sin, b cos + a sin).
    return x * c + _rotate_half(x) * s


class SwiGLU(nn.Module):
    w1: nn.Linear
    w2: nn.Linear
    w3: nn.Linear

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The silu(gate) multiplies the linear "up" branch elementwise: a learned, smooth gate.
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class KVCache:
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
        shape = (batch_size, n_kv_heads, max_seq_len, head_dim)
        # Allocate once up front. Growing with torch.cat every step would re-copy the whole
        # history each token (O(T^2) copying) and fragment the allocator.
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    def update(
        self, k: torch.Tensor, v: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, seq_len, _ = k.shape
        end = start_pos + seq_len
        if start_pos < 0 or end > self.k.shape[2]:
            raise ValueError(f"cache overflow: positions up to {end} > {self.k.shape[2]}")
        # In-place slice writes: no new allocation. Batch may be smaller than the cache's max.
        self.k[:bsz, :, start_pos:end] = k
        self.v[:bsz, :, start_pos:end] = v
        # Return views of the valid prefix (positions 0 .. end-1), not the whole buffer:
        # the unwritten zeros after `end` must never be attended to.
        return self.k[:bsz, :, :end], self.v[:bsz, :, :end]


class GroupedQueryAttention(nn.Module):
    n_heads: int
    n_kv_heads: int
    head_dim: int
    wq: nn.Linear
    wk: nn.Linear
    wv: nn.Linear
    wo: nn.Linear
    cos: torch.Tensor  # registered buffers (annotated for type checkers)
    sin: torch.Tensor

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 256,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        if n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads  # query heads per kv head

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        # K/V projections are n_rep times smaller than Q: this is where GQA saves cache memory.
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        cos, sin = precompute_rope(self.head_dim, max_seq_len, rope_theta)
        # Buffers follow .to(device) but are not parameters; persistent=False keeps them
        # out of the state_dict since they are recomputable.
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self, x: torch.Tensor, cache: KVCache | None = None, start_pos: int = 0
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        # (B, T, H*D) -> (B, T, H, D) -> (B, H, T, D): heads become a batch-like dim.
        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Rotate by ABSOLUTE position. Keys are rotated before caching, so cached keys never
        # need to be touched again.
        q = apply_rope(q, self.cos, self.sin, offset=start_pos)
        k = apply_rope(k, self.cos, self.sin, offset=start_pos)

        if cache is not None:
            k, v = cache.update(k, v, start_pos)  # now (B, Hkv, start_pos + T, D)
        kv_len = k.shape[2]

        # The cache stores Hkv heads; expand only for the matmul. repeat_interleave gives
        # [kv0, kv0, kv1, kv1, ...], so query head h uses kv head h // n_rep.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Query row i sits at position (kv_len - T) + i relative to key 0, so it may see keys
        # j <= i + (kv_len - T). The diagonal offset handles prompt (offset 0), single-token
        # decode (one full row) and chunked prefill alike. Note: is_causal=True in SDPA aligns
        # the mask top-left, which is WRONG whenever kv_len > T.
        mask = torch.ones(seq_len, kv_len, dtype=torch.bool, device=x.device).tril(
            diagonal=kv_len - seq_len
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (B, H, T, D)

        # transpose makes it non-contiguous; reshape copies into (B, T, H*D) layout.
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.n_heads * self.head_dim)
        return self.wo(out)
