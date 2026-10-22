"""Reference solution for p04_attention. Try attention.py yourself before reading this."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def attention_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    if attn_mask is not None and is_causal:
        raise ValueError("pass either attn_mask or is_causal=True, not both")

    # (..., L, E) @ (..., E, S) -> (..., L, S). transpose(-2, -1) keeps any leading dims intact.
    # Dividing by sqrt(E) keeps the logits' variance ~1 so softmax doesn't saturate.
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))

    if is_causal:
        L, S = q.size(-2), k.size(-2)
        # Query i may see keys 0..i (top-left aligned, same as F.scaled_dot_product_attention).
        attn_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).tril()

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # True = keep. -inf becomes exactly 0 after softmax. masked_fill broadcasts the mask.
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            # A float mask is *added* to the logits (0 = keep, -inf/large negative = drop).
            scores = scores + attn_mask

    # torch.softmax subtracts the row max internally, so huge logits don't overflow exp().
    # Writing it by hand would be: (scores - scores.amax(-1, keepdim=True)).exp() / sum.
    return torch.softmax(scores, dim=-1)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    weights = attention_weights(q, k, attn_mask=attn_mask, is_causal=is_causal)
    if dropout_p > 0.0:
        # Dropout on the attention probabilities (not the output), like the PyTorch reference.
        weights = F.dropout(weights, p=dropout_p, training=True)
    # (..., L, S) @ (..., S, Ev) -> (..., L, Ev)
    return weights @ v


class MultiHeadAttention(nn.Module):
    # Class-level annotations tell type checkers what these submodules are
    # (nn.Module.__getattr__ would otherwise type them as "Tensor | Module").
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    out_proj: nn.Linear

    def __init__(
        self, embed_dim: int, num_heads: int, bias: bool = True, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        # Separate projections are easiest to read; a fused (E -> 3E) Linear is the same math.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        # (B, T, E) -> (B, T, H, D) -> (B, H, T, D): heads become a batch dim for SDPA.
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        key = query if key is None else key
        value = key if value is None else value
        B, L, _ = query.shape
        S = key.size(1)

        q = self._split_heads(self.q_proj(query))  # (B, H, L, D)
        k = self._split_heads(self.k_proj(key))  # (B, H, S, D)
        v = self._split_heads(self.v_proj(value))  # (B, H, S, D)

        # Build ONE boolean "keep" mask that broadcasts to (B, H, L, S).
        mask: torch.Tensor | None = None
        if key_padding_mask is not None:
            # Convention flip: key_padding_mask uses True = PAD, SDPA's bool mask uses True = KEEP.
            # (B, S) -> (B, 1, 1, S): same for every head and every query.
            mask = ~key_padding_mask.bool()[:, None, None, :]
        if is_causal:
            causal = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
            mask = causal if mask is None else mask & causal

        # nn.Module.training is flipped by .train()/.eval(); no dropout at inference.
        dropout_p = self.dropout if self.training else 0.0
        out = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

        # (B, H, L, D) -> (B, L, H, D) -> (B, L, E). transpose makes it non-contiguous,
        # so use reshape (or .contiguous().view) to merge the heads back.
        out = out.transpose(1, 2).reshape(B, L, self.embed_dim)
        return self.out_proj(out)
