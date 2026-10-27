"""Reference fix for r09_lora: LoRA adapters for parameter-efficient fine-tuning.

Same public API as ``pr.py``. Lines marked ``# FIX(Bn)`` map to solution/REVIEW.md; lines marked
``# PROD`` are production improvements that are not bug fixes.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import torch
from torch import nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """A frozen-able ``nn.Linear`` plus a trainable low-rank update ``(alpha / r) * B @ A``."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={r}")
        if not 0.0 <= dropout < 1.0:  # PROD: validate config early
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r  # FIX(B2): the paper's scale is alpha / r, not r / alpha
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
        nn.init.zeros_(self.lora_B)  # FIX(B3): B = 0 so B @ A = 0 and the wrapped layer == base

    def delta_weight(self) -> torch.Tensor:
        """The dense weight update contributed by the adapter, shaped like ``base.weight``."""
        return (self.lora_B @ self.lora_A) * self.scaling  # FIX(B4): (out, r) @ (r, in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)  # FIX(B6): the frozen base path never sees adapter dropout
        if self.merged:
            return result
        return result + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the adapter into the base weight (for inference). Idempotent."""
        if self.merged:  # FIX(B5): a second merge would add the delta twice
            return
        self.base.weight += self.delta_weight()
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo :meth:`merge`, restoring the original base weight. Idempotent."""
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
    """Replace every ``nn.Linear`` whose *own* name is in ``target_modules`` with ``LoRALinear``.

    Matching is on the last path component (``blocks.0.attn.q_proj`` -> ``q_proj``), so
    ``v_proj`` does not also match ``kv_proj``. Returns the qualified names replaced.
    """
    targets = frozenset(target_modules)
    to_replace = [
        name
        for name, module in model.named_modules()
        # FIX(B1): exact match on the leaf name; str.endswith("v_proj") also hits "kv_proj"
        if isinstance(module, nn.Linear) and name.rpartition(".")[2] in targets
    ]
    if not to_replace:  # PROD: a typo in target_modules should not silently train nothing
        raise ValueError(f"no nn.Linear modules matched target_modules={sorted(targets)}")
    for name in to_replace:
        base = model.get_submodule(name)
        assert isinstance(base, nn.Linear)
        _set_submodule(model, name, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
    logger.info("inject_lora: wrapped %d modules (r=%d, alpha=%s)", len(to_replace), r, alpha)
    return to_replace


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze every parameter except the LoRA matrices."""
    # FIX(B7): freeze *all* non-adapter params (biases, embeddings, norms, untargeted Linears),
    # not just the weight of the wrapped layers.
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)


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
    """Adapter-only snapshot of the model (``lora_A`` / ``lora_B`` tensors).

    The tensors are independent copies: later optimizer steps do not change the snapshot.
    """
    # FIX(B8): state_dict() tensors alias the live parameters; clone so in-place
    # optimizer updates cannot rewrite a snapshot taken earlier.
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load an adapter snapshot produced by :func:`lora_state_dict`.

    Raises ``ValueError`` if the snapshot's keys do not exactly match the model's adapters.
    """
    # PROD: strict=False alone silently ignores a mismatched adapter (wrong model / targets).
    expected = {k for k in model.state_dict() if "lora_" in k}
    missing, unexpected = expected - state.keys(), state.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"adapter keys mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    model.load_state_dict(state, strict=False)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
