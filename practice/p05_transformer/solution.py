"""Reference solution for p05_transformer. Try transformer.py yourself before reading this."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from common import get_device


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # nn.Parameter registers the tensor: it shows up in .parameters(), state_dict, .to(device).
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        # correction=0 (divide by N, not N-1) is what nn.LayerNorm uses.
        var = x.var(dim=-1, keepdim=True, correction=0)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.weight + self.bias


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        # nn.Sequential composes modules; parameters of the children are registered automatically.
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.dropout = dropout
        # One matmul produces q, k and v together (faster than three separate Linears).
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # (B, T, C) -> (B, n_head, T, head_dim): SDPA treats every leading dim as a batch dim.
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            # Attention dropout only while training; module.training is flipped by .train()/.eval().
            dropout_p=self.dropout if self.training else 0.0,
        )
        # transpose makes the tensor non-contiguous, and view needs contiguous memory.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: normalize the *input* of each sublayer, keep the residual path untouched.
        # The clean identity path x -> x + ... is what lets deep stacks train stably.
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int = 64
    block_size: int = 32
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 32
    dropout: float = 0.0

    def __post_init__(self) -> None:
        # Fail fast at construction time, with a message that names the bad field.
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class GPT(nn.Module):
    # Class-level annotations tell type checkers what the submodules are
    # (nn.Module.__getattr__ would otherwise type them as `Tensor | Module`).
    config: GPTConfig
    tok_emb: nn.Embedding
    pos_emb: nn.Embedding
    pos_ids: torch.Tensor
    blocks: nn.ModuleList
    ln_f: LayerNorm
    lm_head: nn.Linear

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        # A buffer moves with .to(device) but is not a trainable parameter.
        # persistent=False keeps it out of state_dict (it is derivable from the config).
        self.register_buffer("pos_ids", torch.arange(config.block_size), persistent=False)
        self.drop = nn.Dropout(config.dropout)
        # nn.ModuleList (not a plain list!) so the blocks' parameters are registered.
        self.blocks = nn.ModuleList(
            TransformerBlock(config.n_embd, config.n_head, config.dropout)
            for _ in range(config.n_layer)
        )
        self.ln_f = LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the output projection reuses the input embedding matrix (same Parameter).
        # Both are (vocab_size, n_embd), since nn.Linear stores weight as (out, in).
        self.lm_head.weight = self.tok_emb.weight

        # Small init keeps the initial logits near zero, so the initial loss is ~ln(vocab_size).
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        # .parameters() yields each Parameter object once, so the tied weight is counted once.
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )
        pos = self.pos_ids[:seq_len]
        # (B, T, C) + (T, C) broadcasts the positional embedding over the batch.
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            # cross_entropy wants (N, num_classes) logits and (N,) class indices.
            loss = F.cross_entropy(logits.view(batch * seq_len, -1), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        for _ in range(max_new_tokens):
            # The model only knows block_size positions: feed it the most recent window.
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # only the last position predicts the next
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[:, [-1]]
                # Everything below the k-th largest logit gets probability exactly 0.
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


if __name__ == "__main__":
    device = get_device()
    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    idx = torch.randint(cfg.vocab_size, (2, 8), device=device)
    _, loss = model(idx, targets=torch.randint(cfg.vocab_size, (2, 8), device=device))
    assert loss is not None
    print(f"{model.num_parameters():,} params, initial loss {loss.item():.3f}")
    print(f"ln(vocab_size) = {math.log(cfg.vocab_size):.3f}")
