"""Transformer block and a tiny GPT (decoder-only language model).

Build a GPT from parts, bottom up, then run:
    uv run pytest practice/p05_transformer
    uv run pytest practice/p05_transformer -rA     # list every test with its result

What to build (in this order; each piece is used by the next):
1. `LayerNorm`: layer normalization over the last dim, from scratch.
2. `FeedForward`: Linear(C, 4C) -> GELU -> Linear(4C, C) -> Dropout.
3. `CausalSelfAttention`: multi-head causal self-attention with ONE fused Linear(C, 3C) for q/k/v,
   then an output Linear(C, C) and dropout.
4. `TransformerBlock`: pre-norm block, x = x + attn(ln1(x)); x = x + ffn(ln2(x)).
5. `GPTConfig`: a @dataclass of hyperparameters that validates itself in __post_init__.
6. `GPT`: embeddings + blocks + final LayerNorm + lm_head, with loss and sampling.

Rules:
- You MAY use `torch.nn.functional.scaled_dot_product_attention` for the attention core (you built
  attention by hand in p04). Everything else is yours.
- No `nn.LayerNorm` / `F.layer_norm` anywhere (the tests check). Use your own `LayerNorm`.
- No `nn.MultiheadAttention` / `nn.TransformerDecoderLayer` and friends.
- All Linear layers have a bias, except `lm_head` (bias=False, its weight is tied).
- The tests count parameters exactly, so stick to the architecture above (hidden size 4 * n_embd,
  learned positional embeddings of shape (block_size, n_embd), one LayerNorm per sublayer plus a
  final one).

Hints:
- LayerNorm uses the *biased* variance (divide by N): `x.var(dim=-1, keepdim=True, correction=0)`.
  weight starts at 1, bias at 0, both `nn.Parameter`s of shape (normalized_shape,).
- Heads: (B, T, C) -> view (B, T, n_head, head_dim) -> transpose to (B, n_head, T, head_dim).
  After attention, transpose back and call `.contiguous()` before `.view(B, T, C)`.
- SDPA has `is_causal=True` and `dropout_p`. Only apply dropout while `self.training`.
- Submodules must be registered to be trained/moved/saved: assign them as attributes, and use
  `nn.ModuleList` (never a plain Python list) for the stack of blocks.
- Position ids: `self.register_buffer("pos_ids", torch.arange(block_size), persistent=False)`.
  A buffer follows `.to(device)` but is not a parameter.
- Weight tying: `self.lm_head.weight = self.tok_emb.weight`. nn.Linear stores weight as
  (out_features, in_features) = (vocab_size, n_embd), exactly the embedding's shape.
- Init (given in `_init_weights`): normal(0, 0.02) for Linear/Embedding weights, zero biases.
  Small weights make logits ~0 at init, so the initial loss is close to ln(vocab_size).
- `F.cross_entropy` wants logits of shape (N, vocab) and targets of shape (N,): flatten B and T.
- generate: crop the context to the last `block_size` tokens, take the logits of the LAST
  position, divide by temperature, optionally keep only the top_k logits (set the rest to -inf),
  softmax, `torch.multinomial`, append, repeat.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: F401  (you will need it)
from torch import nn

from common import get_device


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension, equivalent to nn.LayerNorm(normalized_shape).

    y = (x - mean) / sqrt(var + eps) * weight + bias, with mean/var over the last dim (biased var).
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        """
        Args:
            normalized_shape: size of the last dimension of the input.
            eps: added to the variance for numerical stability.

        Attributes to create:
            weight: nn.Parameter of ones, shape (normalized_shape,).
            bias: nn.Parameter of zeros, shape (normalized_shape,).
        """
        super().__init__()
        raise NotImplementedError("Your code here")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., normalized_shape).

        Returns:
            Tensor of the same shape as x.
        """
        raise NotImplementedError("Your code here")


class FeedForward(nn.Module):
    """Position-wise MLP: Linear(C, 4C) -> GELU -> Linear(4C, C) -> Dropout."""

    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        """
        Args:
            n_embd: model width C.
            dropout: dropout probability applied to the output.
        """
        super().__init__()
        raise NotImplementedError("Your code here")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C).

        Returns:
            (B, T, C).
        """
        raise NotImplementedError("Your code here")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention: fused qkv Linear(C, 3C), SDPA, output Linear(C, C)."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        """
        Args:
            n_embd: model width C.
            n_head: number of heads; head_dim = C // n_head.
            dropout: dropout on the attention weights (training only) and on the output.

        Raises:
            ValueError: if n_embd is not divisible by n_head.
        """
        super().__init__()
        raise NotImplementedError("Your code here")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C).

        Returns:
            (B, T, C). Output at position t depends only on inputs at positions <= t.
        """
        raise NotImplementedError("Your code here")


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x + attn(ln1(x)), then x + ffn(ln2(x))."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        """
        Args:
            n_embd: model width C.
            n_head: number of attention heads.
            dropout: dropout probability used by the sublayers.
        """
        super().__init__()
        raise NotImplementedError("Your code here")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C).

        Returns:
            (B, T, C).
        """
        raise NotImplementedError("Your code here")


@dataclass
class GPTConfig:
    """Hyperparameters of the GPT. Validated on construction."""

    vocab_size: int = 64
    block_size: int = 32  # maximum context length
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 32
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate the fields.

        Raises:
            ValueError: if any of vocab_size, block_size, n_layer, n_head, n_embd is not positive,
                if n_embd is not divisible by n_head, or if dropout is not in [0, 1).
        """
        raise NotImplementedError("Your code here")


class GPT(nn.Module):
    """Decoder-only language model.

    Attributes to create (the tests use these names):
        config: the GPTConfig.
        tok_emb: nn.Embedding(vocab_size, n_embd).
        pos_emb: nn.Embedding(block_size, n_embd).
        pos_ids: buffer torch.arange(block_size) (non-persistent).
        blocks: nn.ModuleList of n_layer TransformerBlocks.
        ln_f: final LayerNorm.
        lm_head: nn.Linear(n_embd, vocab_size, bias=False), weight tied to tok_emb.weight.
    Plus an nn.Dropout(dropout) on the summed embeddings.
    """

    # Given: class-level annotations so type checkers know the submodule types
    # (nn.Module.__getattr__ would otherwise type them as `Tensor | Module`).
    config: GPTConfig
    tok_emb: nn.Embedding
    pos_emb: nn.Embedding
    pos_ids: torch.Tensor
    blocks: nn.ModuleList
    ln_f: LayerNorm
    lm_head: nn.Linear

    def __init__(self, config: GPTConfig) -> None:
        """
        Args:
            config: model hyperparameters.

        Call `self.apply(self._init_weights)` at the end (after building every submodule).
        """
        super().__init__()
        raise NotImplementedError("Your code here")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Given: GPT-2 style init. Small weights => logits near 0 => initial loss ~ ln(vocab_size).
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        """Return the number of trainable scalars, counting the tied weight once."""
        raise NotImplementedError("Your code here")

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx: (B, T) int64 token ids, T <= block_size.
            targets: optional (B, T) int64 next-token ids.

        Returns:
            logits: (B, T, vocab_size).
            loss: mean cross-entropy over all B*T positions if targets is given, else None.

        Raises:
            ValueError: if T > block_size.
        """
        raise NotImplementedError("Your code here")

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample max_new_tokens tokens.

        Args:
            idx: (B, T) int64 prompt; T may exceed block_size (only the last block_size tokens
                are fed to the model at each step).
            max_new_tokens: number of tokens to append.
            temperature: logits are divided by this before softmax (> 0).
            top_k: if given, sample only among the top_k most likely tokens (top_k=1 is greedy).

        Returns:
            (B, T + max_new_tokens) tensor: the prompt followed by the sampled tokens.

        Raises:
            ValueError: if temperature <= 0.
        """
        raise NotImplementedError("Your code here")


if __name__ == "__main__":
    device = get_device()
    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    idx = torch.randint(cfg.vocab_size, (2, 8), device=device)
    _, loss = model(idx, targets=torch.randint(cfg.vocab_size, (2, 8), device=device))
    assert loss is not None
    print(f"{model.num_parameters():,} params, initial loss {loss.item():.3f}")
    print(f"ln(vocab_size) = {math.log(cfg.vocab_size):.3f}")
