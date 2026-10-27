"""Reference fix for review/r01_transformer/pr.py (same public API).

See REVIEW.md for the bug list (B1..B8) and the production improvements applied here.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        # Fail fast on configs that would otherwise break deep inside forward().
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} must be divisible by n_head={self.n_head}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with a fused QKV projection."""

    mask: torch.Tensor

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        size = config.block_size
        # FIX(B4): lower triangle INCLUDING the diagonal (diagonal=0): token t sees 0..t, not t+1.
        mask = torch.tril(torch.ones(size, size, dtype=torch.bool))
        # Derived from config, so keep it out of state_dict (persistent=False).
        self.register_buffer("mask", mask.view(1, 1, size, size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C)."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # FIX(B5): scale by sqrt(head_dim) -- the dot product is over head_dim terms, not C.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # FIX(B6): (B, H, T, D) -> (B, T, H, D) before merging heads; reshape copies if needed.
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
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
        # FIX(B3): pre-norm -- normalize the sublayer INPUT, keep the residual path un-normalized.
        x = x + self.attn(self.ln_1(x))
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
        self.apply(self._init_weights)
        # FIX(B7): share the SAME Parameter object (one tensor, one grad), not a cloned copy.
        self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        """Unique trainable parameters (a tied weight is counted once)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
            # FIX(B1): cross_entropy wants (N, V) vs (N,) -- class dim is dim 1, not the last one.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # FIX(B8): no autograd graph during decoding (each step would otherwise save activations
    # for a backward pass that never comes).
    @torch.no_grad()
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
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0 (use greedy=True), got {temperature}")
        # FIX(B8): decode in eval mode (dropout off) and restore the caller's mode afterwards.
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # FIX(B2): crop to the last block_size tokens so decoding can run past the window.
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)
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
        finally:
            self.train(was_training)
        return idx
