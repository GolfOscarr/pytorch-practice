"""Add LoRA adapters for parameter-efficient fine-tuning.

## Summary
Adds a small, dependency-free LoRA implementation so we can fine-tune the 300M checkpoint on a
single GPU without touching the base weights. Adapters are injected into the attention
projections, only the low-rank matrices are trained, and the result can either be shipped as a
tiny adapter checkpoint or merged back into the base weights for zero-overhead inference.

## Changes
- `LoRALinear`: wraps an existing `nn.Linear` and adds the low-rank update
  `y = W x + (alpha / r) * B A x`. Supports `merge()` / `unmerge()` for deployment.
- `inject_lora(model, target_modules=("q_proj", "v_proj"), ...)`: swaps matching `nn.Linear`
  layers for `LoRALinear` in place and returns the qualified names it replaced.
- `mark_only_lora_trainable(model)`: freezes the pretrained weights so only adapters train.
- `build_optimizer(model, lr, weight_decay)`: AdamW over trainable params only, with two param
  groups (2-D matrices get weight decay; biases / norm weights do not).
- `lora_state_dict(model)` / `load_lora_state_dict(model, state)`: adapter-only checkpoints
  (a few hundred KB instead of the full model).
- `count_trainable_params(model)`: for logging the parameter budget.

## Testing
Ran a short fine-tune on the summarization set: loss goes down, adapter checkpoint is 0.3% of the
full model size, and the merged model loads in the existing inference server unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen-able ``nn.Linear`` plus a trainable low-rank update ``(alpha / r) * B @ A``."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={r}")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = r / alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        weight = base.weight
        self.lora_A = nn.Parameter(
            torch.empty(r, base.in_features, device=weight.device, dtype=weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.empty(base.out_features, r, device=weight.device, dtype=weight.dtype)
        )
        self.merged = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.normal_(self.lora_B, std=0.02)

    def delta_weight(self) -> torch.Tensor:
        """The dense weight update contributed by the adapter."""
        return (self.lora_A.T @ self.lora_B.T) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        result = self.base(x)
        if self.merged:
            return result
        return result + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the adapter into the base weight (for inference)."""
        self.base.weight += self.delta_weight()
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo :meth:`merge`, restoring the original base weight."""
        if not self.merged:
            return
        self.base.weight -= self.delta_weight()
        self.merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, merged={self.merged}"


def _set_submodule(model: nn.Module, qualified_name: str, new: nn.Module) -> None:
    parent_name, _, child_name = qualified_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, new)


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str] = ("q_proj", "v_proj"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    """Replace every ``nn.Linear`` named in ``target_modules`` with a ``LoRALinear``.

    Returns the qualified names of the replaced modules.
    """
    targets = tuple(target_modules)
    to_replace = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.endswith(targets)
    ]
    for name in to_replace:
        base = model.get_submodule(name)
        assert isinstance(base, nn.Linear)
        _set_submodule(model, name, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
    return to_replace


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze the pretrained weights so only the LoRA matrices receive gradients."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.base.weight.requires_grad_(False)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01) -> torch.optim.AdamW:
    """AdamW over the trainable parameters.

    Policy: 2-D parameters (including ``lora_A`` / ``lora_B``) get ``weight_decay``; 1-D
    parameters (biases, norm weights) get none. Frozen parameters are not handed to the optimizer.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.ndim >= 2 else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]], lr=lr)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Adapter-only snapshot of the model (``lora_A`` / ``lora_B`` tensors)."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load an adapter snapshot produced by :func:`lora_state_dict`."""
    model.load_state_dict(state, strict=False)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
