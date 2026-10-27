"""Add train/validate loop with gradient accumulation, clipping and LR schedule.

## Summary
Every experiment script in the repo had its own copy-pasted training loop. This PR adds one
shared, tested loop so new experiments only have to supply a model, loaders and an optimizer.

## Changes
- `train_one_epoch`: one pass over the training loader with
  - gradient accumulation (`accumulation_steps` micro-batches per optimizer step, so we can
    reach large effective batch sizes on the M4's memory budget),
  - optional global-norm gradient clipping (`max_grad_norm`),
  - an optional per-iteration LR scheduler (warmup / cosine / OneCycle style). The scheduler
    is advanced once per optimizer step, which is the convention those schedulers expect.
  Returns the sample-weighted mean training loss of the epoch. The running loss stays on the
  device and is synced once at the end of the epoch, so there is no per-step `.item()` stall
  on MPS/CUDA.
- `evaluate`: validation pass, returns the sample-weighted mean loss over the dataset.
- `fit`: runs `epochs` x (train, validate) and returns one `EpochMetrics` record per epoch,
  including the learning rate at the end of the epoch so it is easy to plot.
- `TrainConfig`: small dataclass that holds the loop hyperparameters.

`loss_fn` is expected to use `reduction="mean"` (the PyTorch default); the loop handles the
re-weighting for accumulation and for averaging across batches.

## Testing
Ran the MNIST MLP experiment through `fit` for 5 epochs with `accumulation_steps=4` and
`max_grad_norm=1.0`: training loss goes down every epoch and validation tracks it. Also ran it
on MPS and CPU, both fine.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainConfig:
    epochs: int = 1
    accumulation_steps: int = 1
    max_grad_norm: float | None = None


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
    full window produces the gradient of the mean loss over that window.
    """
    device = _model_device(model)
    model.train()

    running_loss = torch.zeros((), device=device)
    num_samples = 0

    optimizer.zero_grad(set_to_none=True)
    for step, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        loss = loss_fn(model(inputs), targets)
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if scheduler is not None:
            scheduler.step()

        batch_size = inputs.size(0)
        running_loss += loss * batch_size
        num_samples += batch_size

    return running_loss.item() / num_samples


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable[Batch], loss_fn: LossFn) -> float:
    """Return the mean loss per sample of ``model`` over ``loader``."""
    device = _model_device(model)

    total_loss = 0.0
    num_batches = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        total_loss += loss_fn(model(inputs), targets).item()
        num_batches += 1

    return total_loss / num_batches


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
    return history
