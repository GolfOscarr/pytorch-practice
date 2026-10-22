"""Production training pipeline: config, data, callbacks, trainer, checkpoint/resume.

Build the small but production-shaped training stack that most real PyTorch projects grow into.
Run the tests with:
    uv run pytest practice/p07_trainer          (add -rA to list each test)

What to build (the API below is fixed; the tests use it):

1. `TrainConfig` - a frozen dataclass (given). Implement:
   - `__post_init__`: validate, raising ValueError for: lr <= 0 or NaN, epochs < 1,
     batch_size < 1, grad_clip not None and <= 0, weight_decay < 0, warmup_steps < 0,
     log_every < 1, or a `device` string that `torch.device()` cannot parse.
   - `to_dict()`: plain, JSON-serializable dict (checkpoint_dir as str, not Path).
   - `from_dict(d)`: inverse of to_dict. Unknown keys raise a ValueError whose message names
     the unknown key(s) (catches typos).
   Why frozen: a config describes ONE run; if it could change mid-run, the copy saved in the
   checkpoint would no longer describe how the weights were produced. Use
   `dataclasses.replace(cfg, lr=...)` to make variants.

2. `ArrayDataset(features, targets)` - map-style dataset over two tensors of equal length, and
   `make_dataloaders(train_ds, val_ds=None, *, batch_size, seed)` - train loader shuffles using a
   private `torch.Generator` seeded with `seed`; the val loader (if any) does not shuffle.

3. Callbacks - `Callback` is a base class with no-op hooks (given); subclasses override only
   what they need. Implement:
   - `EarlyStopping(patience, min_delta, monitor="val_loss")`: an epoch "improves" if
     metrics[monitor] < best - min_delta. After `patience` epochs in a row without improvement,
     set `trainer.should_stop = True`. Reset state in on_train_start.
   - `HistoryCallback`: append a copy of every epoch's metrics to `self.history`.

4. `warmup_cosine_factor(step, warmup_steps, total_steps)` - the LR multiplier:
   - step < warmup_steps:             step / warmup_steps   (so step 0 uses lr 0)
   - total_steps <= warmup_steps:     1.0                   (no decay horizon known)
   - otherwise: 0.5 * (1 + cos(pi * p)), p = (step - warmup) / (total - warmup), clipped to 1.

5. `Trainer(model, config, loss_fn, callbacks=())`:
   - __init__: device = `common.get_device(config.device)`; move the model there; build
     `torch.optim.AdamW(lr, weight_decay)` and a `LambdaLR` whose lambda calls
     warmup_cosine_factor with `self.total_steps` (set to 0 here; fit() sets it to
     `epochs * len(train_loader)`). Set epoch = global_step = 0, should_stop = False.
   - fit(train_loader, val_loader=None) -> list of per-epoch metrics dicts
     {"epoch", "train_loss", "lr", and "val_loss" if val_loader}. Epochs are 1-based in metrics.
     Loop epochs from `self.epoch` (so a restored trainer continues) up to config.epochs.
     Each training step: zero_grad(set_to_none=True) -> forward+loss (under autocast if use_amp)
     -> backward -> clip_grad_norm_ if grad_clip (else just measure the norm) ->
     optimizer.step() -> scheduler.step() -> global_step += 1 -> callbacks' on_step_end with
     {"step": global_step, "loss", "lr": the lr THIS step used, "grad_norm": pre-clip norm}.
     After each epoch: build metrics, log, on_epoch_end, save `checkpoint_dir / "last.pt"` if
     checkpoint_dir is set, then break if `should_stop`. Hook order: on_train_start,
     (on_step_end...) on_epoch_end per epoch, on_train_end; callbacks in list order.
   - evaluate(loader) -> {"loss": sample-weighted mean loss}, in eval mode under torch.no_grad,
     restoring whatever train/eval mode the model had before (even on exceptions).
   - save_checkpoint(path) / load_checkpoint(path): a dict with exactly these keys:
       "model", "optimizer", "scheduler" (state_dicts), "epoch", "global_step",
       "config" (config.to_dict()), "rng" (a dict: torch CPU state, python `random` state, and
       the accelerator's generator state when training on MPS/CUDA). Must load with
     `torch.load(..., weights_only=True)`. Train 2 epochs + save + load into a FRESH trainer +
     fit must equal 4 epochs straight.
   - Logging: module-level `logger = logging.getLogger(__name__)`, never print. INFO message
     starting with "step " whenever global_step % log_every == 0, and one starting with
     "epoch " after every epoch. These are the ONLY messages starting with "step "/"epoch ",
     and a normal fit logs only at INFO (other INFO messages, e.g. "saved checkpoint ...", are
     fine).

Hints (the tricky parts):
- Resume-exactness has three enemies: (a) the shuffle order - the loader's generator has
  advanced differently in a fresh process. Reseed it at the start of every epoch with
  `config.seed + epoch` (`train_loader.generator.manual_seed(...)`), like
  DistributedSampler.set_epoch; (b) dropout draws from the RNG of the device it runs on -
  save/restore the CPU generator and, on MPS/CUDA, `torch.mps.get_rng_state()` /
  `torch.cuda.get_rng_state()` too;
  (c) the LR schedule - restore the scheduler and keep `epochs` the same so the cosine horizon
  matches.
- `LambdaLR.state_dict()` does not store a plain lambda, so the lambda reading `self.total_steps`
  lazily is fine. LambdaLR calls the lambda once in its constructor (factor(0)).
- `torch.get_rng_state()` must be restored as a CPU tensor: load with map_location="cpu";
  `load_state_dict` moves weights / optimizer state to the right device.
- Save atomically: write `<path>.tmp` then `os.replace` - a crash never leaves a half-written
  checkpoint under the real name. Create parent directories.
- AMP: `torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)` works
  on CPU and MPS in this torch version. bf16 needs no GradScaler (fp32-sized exponent).
- `clip_grad_norm_` returns the norm BEFORE clipping. `nn.utils.get_total_norm` measures one
  without clipping.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters and run settings for one training run.

    Attributes:
        lr: peak learning rate (reached at the end of warmup).
        epochs: total number of epochs of the run (also sets the cosine horizon).
        batch_size: batch size for make_dataloaders.
        grad_clip: max total gradient L2 norm, or None to disable clipping.
        weight_decay: AdamW decoupled weight decay.
        warmup_steps: optimizer steps of linear warmup from 0 to lr.
        seed: seed for data shuffling.
        device: e.g. "cpu", "mps"; None picks the best available device.
        checkpoint_dir: if set, fit() saves checkpoint_dir / "last.pt" after every epoch.
        log_every: log a "step ..." line every this many optimizer steps.
        use_amp: run forward passes under bfloat16 autocast.
    """

    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 32
    grad_clip: float | None = None
    weight_decay: float = 0.0
    warmup_steps: int = 0
    seed: int = 0
    device: str | None = None
    checkpoint_dir: Path | None = None
    log_every: int = 10
    use_amp: bool = False

    def __post_init__(self) -> None:
        """Validate fields.

        Raises:
            ValueError: if any field is out of range (see module docstring).
        """
        raise NotImplementedError("Your code here")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of all fields (checkpoint_dir as str or None)."""
        raise NotImplementedError("Your code here")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        """Build a config from `to_dict()` output. Missing keys take their defaults.

        Raises:
            ValueError: on unknown keys (the message names them), or invalid values.
        """
        raise NotImplementedError("Your code here")


class ArrayDataset(Dataset[Batch]):
    """Map-style dataset: item i is (features[i], targets[i]).

    Attributes:
        features: the features tensor passed in.
        targets: the targets tensor passed in.
    """

    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Args:
            features: tensor of shape (N, ...).
            targets: tensor of shape (N, ...).

        Raises:
            ValueError: if the first dimensions differ.
        """
        raise NotImplementedError("Your code here")

    def __len__(self) -> int:
        raise NotImplementedError("Your code here")

    def __getitem__(self, index: int) -> Batch:
        raise NotImplementedError("Your code here")


def make_dataloaders(
    train_ds: Dataset[Batch],
    val_ds: Dataset[Batch] | None = None,
    *,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader[Batch], DataLoader[Batch] | None]:
    """Build train/val loaders with reproducible shuffling.

    Args:
        train_ds: training dataset (shuffled with a private generator seeded by `seed`).
        val_ds: optional validation dataset (not shuffled).
        batch_size: batch size for both loaders.
        seed: shuffle seed.

    Returns:
        (train_loader, val_loader or None).
    """
    raise NotImplementedError("Your code here")


class Callback:
    """Base class for trainer callbacks. Every hook is a no-op; override what you need."""

    def on_train_start(self, trainer: Trainer) -> None:
        pass

    def on_step_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        """Called after optimizer.step() and scheduler.step(); gradients are still populated."""

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        pass


class EarlyStopping(Callback):
    """Stop training when `monitor` stops improving by more than `min_delta`."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0, monitor: str = "val_loss"):
        """
        Args:
            patience: epochs without improvement before stopping (>= 1).
            min_delta: minimum decrease that counts as an improvement (>= 0).
            monitor: key of the epoch metrics dict to watch (lower is better).

        Raises:
            ValueError: if patience < 1 or min_delta < 0.
        """
        raise NotImplementedError("Your code here")

    def on_train_start(self, trainer: Trainer) -> None:
        raise NotImplementedError("Your code here")

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        raise NotImplementedError("Your code here")


class HistoryCallback(Callback):
    """Records a copy of every epoch's metrics in `self.history`."""

    def __init__(self) -> None:
        raise NotImplementedError("Your code here")

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        raise NotImplementedError("Your code here")


def warmup_cosine_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup from 0, then cosine decay to 0 at total_steps.

    Args:
        step: number of optimizer steps taken so far.
        warmup_steps: length of the linear warmup.
        total_steps: total steps of the run (0 or <= warmup_steps means "unknown": hold 1.0).

    Returns:
        A factor in [0, 1].
    """
    raise NotImplementedError("Your code here")


class Trainer:
    """Trains `model` with AdamW + warmup/cosine LR, callbacks, logging and checkpoints.

    Attributes (public, used by callbacks and tests):
        model, config, loss_fn, callbacks (list), device, optimizer, scheduler,
        total_steps (int), epoch (completed epochs), global_step (optimizer steps taken),
        should_stop (bool; a callback sets it to end fit() after the current epoch).
    """

    model: nn.Module
    config: TrainConfig
    loss_fn: LossFn
    callbacks: list[Callback]
    device: torch.device
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    total_steps: int
    epoch: int
    global_step: int
    should_stop: bool

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        loss_fn: LossFn,
        callbacks: Sequence[Callback] = (),
    ) -> None:
        raise NotImplementedError("Your code here")

    def fit(
        self, train_loader: DataLoader[Batch], val_loader: DataLoader[Batch] | None = None
    ) -> list[dict[str, float]]:
        """Train from `self.epoch` up to `config.epochs` (or until should_stop).

        Args:
            train_loader: training batches (x, y). Its generator is reseeded every epoch.
            val_loader: optional; adds "val_loss" to each epoch's metrics.

        Returns:
            The metrics dicts of the epochs run in THIS call.
        """
        raise NotImplementedError("Your code here")

    def evaluate(self, loader: DataLoader[Batch]) -> dict[str, float]:
        """Mean loss over `loader` (weighted by batch size), without building a graph.

        Returns:
            {"loss": float}. The model's previous train/eval mode is restored.
        """
        raise NotImplementedError("Your code here")

    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        """Atomically save full training state to `path` (parent dirs are created).

        Keys: "model", "optimizer", "scheduler", "epoch", "global_step", "config", "rng".
        """
        raise NotImplementedError("Your code here")

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        """Restore state saved by save_checkpoint so fit() continues exactly where it stopped."""
        raise NotImplementedError("Your code here")
