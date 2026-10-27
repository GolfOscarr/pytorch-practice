"""Enable mixed precision training + numerically safer losses.

## Summary
Moves our training step onto automatic mixed precision (autocast + GradScaler) so we can
run fp16/bf16 forward passes, and collects the loss/metric helpers that were scattered
across three notebooks into one module so every experiment computes them the same way.

## Changes
- `train_step(...)`: one AMP optimisation step. Runs the forward pass under
  `torch.autocast`, scales the loss with the `GradScaler`, clips gradients to `grad_clip`
  and lets the scaler skip the update when it sees inf/NaN gradients. Returns a small
  `StepStats` record (loss, grad norm, loss scale, whether the step was skipped) so the
  trainer can log it without holding on to tensors.
- `smoothed_cross_entropy`: cross-entropy with label smoothing.
- `multilabel_bce`: binary cross-entropy for multi-label targets, optional `pos_weight`.
- `sequence_log_probs`: per-token log-probabilities of a token sequence (used by the
  reranker and the perplexity metric).
- `masked_mean`: mean of per-token values over a padding mask.
- `l2_normalize` / `cosine_similarity`: normalisation with an `eps` so zero vectors are safe.
- `softmax_with_temperature`: temperature-scaled softmax for distillation targets.

## Testing
Ran the language-model and tagging configs for 2k steps each with `dtype=torch.float16`;
loss curves match the fp32 baseline and throughput is up ~1.7x. Added unit tests for
the helpers.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class StepStats:
    """Scalar summary of one optimisation step (plain Python values, safe to log/store)."""

    loss: float
    grad_norm: float
    scale: float
    skipped: bool


def train_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    grad_clip: float = 1.0,
    dtype: torch.dtype = torch.float16,
    loss_fn: LossFn = F.cross_entropy,
) -> StepStats:
    """Run one mixed-precision optimisation step.

    Args:
        model: the model; its parameters' device decides where the step runs.
        batch: ``(inputs, targets)`` already on the model's device.
        optimizer: optimizer over ``model.parameters()``.
        scaler: loss scaler; steps with non-finite gradients are skipped by it.
        grad_clip: max global L2 norm of the gradients.
        dtype: autocast compute dtype (``torch.float16`` or ``torch.bfloat16``).
        loss_fn: ``loss_fn(outputs, targets) -> scalar loss``.
    """
    if grad_clip <= 0:
        raise ValueError(f"grad_clip must be positive, got {grad_clip}")
    inputs, targets = batch
    # FIX(B2): autocast on the device the model actually lives on (cpu / mps / cuda).
    device_type = next(model.parameters()).device.type
    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device_type, dtype=dtype):
        outputs = model(inputs)
        loss = loss_fn(outputs.float(), targets)

    scaler.scale(loss).backward()
    # FIX(B1): unscale first so the clip threshold applies to the true gradients.
    scaler.unscale_(optimizer)
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < scale_before

    return StepStats(
        loss=loss.item(),
        grad_norm=grad_norm.item(),
        scale=scaler.get_scale(),
        skipped=skipped,
    )


def smoothed_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1
) -> torch.Tensor:
    """Mean cross-entropy with label smoothing.

    Args:
        logits: ``(N, C)`` unnormalised scores.
        targets: ``(N,)`` class indices.
        smoothing: probability mass spread uniformly over all classes.
    """
    # FIX(B3): F.cross_entropy applies log_softmax itself -- pass raw logits, not probabilities.
    return F.cross_entropy(logits.float(), targets, label_smoothing=smoothing)


def multilabel_bce(
    logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean binary cross-entropy for multi-label classification.

    Args:
        logits: ``(N, C)`` unnormalised scores, one independent label per column.
        targets: ``(N, C)`` 0/1 labels.
        pos_weight: optional ``(C,)`` weight on the positive term of each label.
    """
    # FIX(B4): BCE-with-logits applies the sigmoid itself -- no sigmoid beforehand.
    return F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), pos_weight=pos_weight
    )


def sequence_log_probs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Log-probability the model assigns to each observed token.

    Args:
        logits: ``(B, T, V)`` next-token scores.
        tokens: ``(B, T)`` observed token ids.

    Returns:
        ``(B, T)`` float32 tensor with ``log p(tokens[b, t])``.
    """
    # FIX(B6): log_softmax uses the log-sum-exp trick; log(softmax) underflows to -inf.
    log_probs = F.log_softmax(logits.float(), dim=-1)
    # FIX(B5): gather needs an index with the same ndim as the input -> (B, T, 1).
    return log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over the positions where ``mask`` is true (e.g. non-padding tokens).

    Args:
        values: any shape, e.g. ``(B, T)`` per-token losses.
        mask: bool tensor with the same shape as ``values``.
    """
    # FIX(B8): reduce in float32 -- an fp16 sum overflows past 65504 to inf.
    weights = mask.to(torch.float32)
    total = (values.float() * weights).sum()
    return total / weights.sum().clamp_min(1.0)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Scale ``x`` to unit L2 norm along ``dim``; all-zero vectors stay zero."""
    # FIX(B7): eps=1e-8 rounds to 0 in fp16 (0/0 = NaN); compute the norm in float32 and
    # clamp it (as F.normalize does) instead of adding eps in the input dtype.
    norm = x.float().norm(dim=dim, keepdim=True).clamp_min(eps)
    return (x.float() / norm).to(x.dtype)


def cosine_similarity(
    a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8
) -> torch.Tensor:
    """Cosine similarity of ``a`` and ``b`` along ``dim`` (0 when either vector is zero)."""
    return (l2_normalize(a, dim, eps) * l2_normalize(b, dim, eps)).sum(dim=dim)


def softmax_with_temperature(
    logits: torch.Tensor, temperature: float = 1.0, dim: int = -1
) -> torch.Tensor:
    """Softmax of ``logits / temperature`` computed in float32."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return torch.softmax(logits.float() / temperature, dim=dim)
