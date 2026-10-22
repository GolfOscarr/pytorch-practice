"""Attention from scratch: scaled dot-product attention and multi-head attention.

Build the core of every Transformer, then prove it is right by matching PyTorch's own
`F.scaled_dot_product_attention` and `nn.MultiheadAttention`. Run:
    uv run pytest practice/p04_attention          (add -rA to list each test)

What to build
-------------
1. `attention_weights(q, k, attn_mask, is_causal)` -> softmax(q @ k^T / sqrt(E) + mask),
   shape (..., L, S).
2. `scaled_dot_product_attention(q, k, v, attn_mask, is_causal, dropout_p)` -> weights @ v,
   shape (..., L, Ev). Same core arguments and semantics as `F.scaled_dot_product_attention`
   (minus its optional `scale` and `enable_gqa`).
   (API choice: the output and the weights are two separate functions instead of a
   `return_weights=True` flag, so each function has one return type.)
3. `MultiHeadAttention(embed_dim, num_heads, bias=True, dropout=0.0)` – an nn.Module.

Rules
-----
- Do NOT call `F.scaled_dot_product_attention`, `F.multi_head_attention_forward` or
  `nn.MultiheadAttention`. Those are only the reference the tests compare against.
- Shapes: q (..., L, E), k (..., S, E), v (..., S, Ev). "..." is ANY number of leading dims
  (batch, heads, ...). L (queries) and S (keys) may differ.
- Masks (same convention as F.scaled_dot_product_attention), broadcastable to (..., L, S):
    * bool mask:  True = KEEP (may attend), False = masked out.
    * float mask: ADDED to the logits (0 = keep, -inf = masked out, anything else = bias).
    * is_causal=True: query i may only attend to keys j <= i (top-left aligned lower triangle
      of an (L, S) matrix, i.e. `torch.ones(L, S).tril()`). Passing both attn_mask and
      is_causal=True raises ValueError.
- `MultiHeadAttention.forward` takes `key_padding_mask` of shape (B, S) where True = PAD
  (ignore that key). Note this is the OPPOSITE of the bool attn_mask convention above.
- MultiHeadAttention must use exactly these submodules (the tests copy weights into them):
  `q_proj`, `k_proj`, `v_proj`, `out_proj`, each `nn.Linear(embed_dim, embed_dim, bias=bias)`.

Hints
-----
- `k.transpose(-2, -1)` (not `.T`) keeps the leading dims intact.
- Masked logits should become -inf so softmax gives them exactly 0 weight. `masked_fill`
  broadcasts the mask for you.
- Numerical stability: softmax(x) == softmax(x - max(x)). `torch.softmax` already does this;
  a hand-written `exp(x) / exp(x).sum()` overflows to inf/nan for large logits.
- Splitting heads: (B, T, E) -view-> (B, T, H, D) -transpose(1, 2)-> (B, H, T, D), D = E // H.
  Merging back: transpose(1, 2) makes the tensor non-contiguous, so use `.reshape` (or
  `.contiguous().view`).
- key_padding_mask (B, S) must broadcast against scores (B, H, L, S): index it as
  [:, None, None, :].
  To combine it with causal masking, build one bool keep-mask (`&` the two) and pass that.
- Dropout on attention weights only while training: check `self.training`.
- How the tests map nn.MultiheadAttention's weights onto yours: its `in_proj_weight` (3E, E)
  is [W_q; W_k; W_v] stacked along dim 0 (same for `in_proj_bias`), so
  `W_q, W_k, W_v = ref.in_proj_weight.chunk(3)`, and `ref.out_proj` maps to `out_proj`.
"""

import torch
from torch import nn


def attention_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Attention probabilities softmax(q @ k^T / sqrt(E) + mask) (before any dropout).

    Args:
        q: queries, shape (..., L, E).
        k: keys, shape (..., S, E).
        attn_mask: optional bool mask (True = keep) or float mask (added to logits),
            broadcastable to (..., L, S).
        is_causal: if True, apply a top-left aligned lower-triangular (L, S) mask.

    Returns:
        Weights of shape (..., L, S); each row over S sums to 1 (unless fully masked).

    Raises:
        ValueError: if both attn_mask and is_causal=True are given.
    """
    raise NotImplementedError("Your code here")


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Scaled dot-product attention, matching F.scaled_dot_product_attention's core arguments.

    (No `scale` / `enable_gqa` arguments.)

    Args:
        q: queries, shape (..., L, E).
        k: keys, shape (..., S, E).
        v: values, shape (..., S, Ev).
        attn_mask: optional bool mask (True = keep) or float mask (added to logits),
            broadcastable to (..., L, S).
        is_causal: if True, apply a top-left aligned lower-triangular (L, S) mask.
        dropout_p: dropout probability applied to the attention weights (0.0 = off).

    Returns:
        Output of shape (..., L, Ev).

    Raises:
        ValueError: if both attn_mask and is_causal=True are given.
    """
    raise NotImplementedError("Your code here")


class MultiHeadAttention(nn.Module):
    """Multi-head attention with batch-first inputs, like nn.MultiheadAttention(batch_first=True).

    Attributes (required names, the tests rely on them):
        q_proj, k_proj, v_proj, out_proj: nn.Linear(embed_dim, embed_dim, bias=bias).
    """

    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    out_proj: nn.Linear

    def __init__(
        self, embed_dim: int, num_heads: int, bias: bool = True, dropout: float = 0.0
    ) -> None:
        """
        Args:
            embed_dim: model dimension E.
            num_heads: number of heads H; each head has dimension E // H.
            bias: whether the four projections have a bias.
            dropout: dropout probability on the attention weights (only in training mode).

        Raises:
            ValueError: if num_heads <= 0 or embed_dim is not divisible by num_heads.
        """
        super().__init__()
        raise NotImplementedError("Your code here")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, L, E).
            key: (B, S, E); defaults to `query` (self-attention).
            value: (B, S, E); defaults to `key`.
            key_padding_mask: optional bool (B, S), True = padding (that key is ignored).
            is_causal: if True, query i only attends to keys j <= i.

        Returns:
            Output of shape (B, L, E).
        """
        raise NotImplementedError("Your code here")
