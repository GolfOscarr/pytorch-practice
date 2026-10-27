"""Reference fix for r02_training_loop: train/validate loop with accumulation, clipping, LR.

Same public API as pr.py. Lines marked `# FIX(Bn)` map to solution/REVIEW.md; lines marked
`# PROD` are production improvements beyond the bug fixes.
"""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

logger = logging.getLogger(__name__)  # PROD: per-epoch metrics go to logging, not print


@dataclass
class TrainConfig:
    epochs: int = 1
    accumulation_steps: int = 1
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:  # PROD: fail fast on nonsense configs
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {self.accumulation_steps}")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0, got {self.max_grad_norm}")


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    lr: float


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    *,
    accumulation_steps: int = 1,
    max_grad_norm: float | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    """Train ``model`` for one epoch and return the mean training loss per sample.

    Gradients of ``accumulation_steps`` consecutive micro-batches are summed before each
    optimizer step. The loss of every micro-batch is scaled by ``1 / accumulation_steps`` so a
    full window produces the gradient of the mean loss over that window. A trailing partial
    window is still stepped. ``scheduler`` (if given) advances once per optimizer step.
    """
    if accumulation_steps < 1:  # PROD: validate at the API boundary too
        raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")
    device = _model_device(model)
    model.train()

    running_loss = torch.zeros((), device=device)  # on-device: no per-step host sync
    num_samples = 0

    def optimizer_step() -> None:
        if max_grad_norm is not None:
            # FIX(B3): clip the gradients that are about to be applied, i.e. BEFORE step().
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        # FIX(B1): clear gradients after every optimizer step, not once per epoch.
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            # FIX(B4): per-iteration schedule advances once per optimizer step.
            scheduler.step()

    optimizer.zero_grad(set_to_none=True)
    pending = 0
    for step, (inputs, targets) in enumerate(loader):
        # PROD: non_blocking overlaps host->device copies with compute for pinned loaders
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        loss = loss_fn(model(inputs), targets)
        # FIX(B2): scale so the summed gradient of a window equals the gradient of its mean.
        (loss / accumulation_steps).backward()
        pending += 1

        if (step + 1) % accumulation_steps == 0:
            optimizer_step()
            pending = 0

        batch_size = inputs.size(0)
        # FIX(B5): detach -- accumulating the live tensor chains every batch's autograd graph.
        running_loss += loss.detach() * batch_size
        num_samples += batch_size

    if pending:
        # FIX(B8): the final partial accumulation window must also be applied.
        optimizer_step()

    if num_samples == 0:  # PROD: clear error instead of ZeroDivisionError
        raise ValueError("train loader yielded no batches")
    return running_loss.item() / num_samples


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable[Batch], loss_fn: LossFn) -> float:
    """Return the mean loss per sample of ``model`` over ``loader``.

    Runs in eval mode (dropout off, BatchNorm uses running stats) and restores the previous
    train/eval mode afterwards.
    """
    device = _model_device(model)
    was_training = model.training
    model.eval()  # FIX(B6): disable dropout / freeze BatchNorm statistics during validation

    total_loss = torch.zeros((), device=device)
    num_samples = 0
    try:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            # FIX(B7): weight each batch's mean loss by its size (last batch may be smaller).
            total_loss += loss_fn(model(inputs), targets) * batch_size
            num_samples += batch_size
    finally:
        model.train(was_training)  # PROD: leave the model in the mode we found it in

    if num_samples == 0:
        raise ValueError("validation loader yielded no batches")
    return total_loss.item() / num_samples


def fit(
    model: nn.Module,
    train_loader: Iterable[Batch],
    val_loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    config: TrainConfig,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> list[EpochMetrics]:
    """Run ``config.epochs`` rounds of training + validation and return per-epoch metrics."""
    history: list[EpochMetrics] = []
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            accumulation_steps=config.accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            scheduler=scheduler,
        )
        val_loss = evaluate(model, val_loader, loss_fn)
        lr = optimizer.param_groups[0]["lr"]
        history.append(EpochMetrics(epoch, train_loss, val_loss, lr))
        logger.info(
            "epoch %d train_loss=%.4f val_loss=%.4f lr=%.3g", epoch, train_loss, val_loss, lr
        )
    return history
