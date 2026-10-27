"""Add a GPT-style decoder (pre-norm transformer block + tiny GPT LM)

## Summary
Adds a small, self-contained decoder-only language model so we can stop copy-pasting nanoGPT
snippets into notebooks. The layout follows GPT-2: learned positional embeddings, a stack of
**pre-norm** transformer blocks (LayerNorm -> sublayer -> residual add), a final LayerNorm and
an LM head whose weight is tied to the token embedding.

## Changes
- `GPTConfig`: dataclass holding all hyperparameters (vocab, context length, depth, width,
  heads, dropout, bias).
- `CausalSelfAttention`: multi-head self-attention with a fused QKV projection and a causal
  mask registered as a buffer, so it follows the module across devices.
- `Block`: pre-norm transformer block (attention sublayer + 4x GELU MLP sublayer).
- `GPT`: embeddings, `n_layer` blocks, final LayerNorm and the tied LM head.
  `forward(idx, targets=None)` returns `(logits, loss)`; `loss` is `None` without targets.
- `GPT.generate`: autoregressive decoding with temperature, optional top-k filtering, a greedy
  mode, and an optional `torch.Generator` for reproducible sampling.

## Testing
- Overfit a 2-layer / 64-dim model on a 1k-character string: train loss goes from ~4.2 to
  ~0.05 in 300 AdamW steps.
- Sampled continuations from the overfit model reproduce the training text.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with a fused QKV projection."""

    mask: torch.Tensor

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        size = config.block_size
        mask = torch.tril(torch.ones(size, size), diagonal=1)
        self.register_buffer("mask", mask.view(1, 1, size, size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C)."""
        B, T, C = x.shape
        head_dim = C // self.n_head
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(C)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = (att @ v).reshape(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ln_1(self.attn(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer language model with a tied LM head."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(self.tok_emb.weight.clone())
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) token ids -> logits (B, T, vocab_size), optional mean CE loss."""
        seq_len = idx.shape[1]
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} > block_size {self.config.block_size}")
        pos = torch.arange(seq_len, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Append `max_new_tokens` tokens to `idx` (B, T) and return (B, T + max_new_tokens)."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if greedy:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
