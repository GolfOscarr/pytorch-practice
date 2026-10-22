"""Reference solution for p07_trainer. Try trainer.py yourself before reading this."""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import get_device

# One logger per module, named after the module. The application decides handlers/levels;
# a library never calls logging.basicConfig() or print().
logger = logging.getLogger(__name__)

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------------------
# 1. Config
# --------------------------------------------------------------------------------------


# frozen=True: a config describes ONE run. If code could mutate it mid-run, the copy saved in
# a checkpoint would no longer describe how the weights were produced. Variants are made
# explicitly with dataclasses.replace(cfg, lr=...).
@dataclass(frozen=True)
class TrainConfig:
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
        # Fail fast at construction, not 3 hours into a run.
        if not self.lr > 0:  # `not >` also rejects NaN
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.grad_clip is not None and not self.grad_clip > 0:
            raise ValueError(f"grad_clip must be None or > 0, got {self.grad_clip}")
        if not self.weight_decay >= 0:  # `not >=` also rejects NaN
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")
        if self.device is not None:
            try:
                torch.device(self.device)
            except RuntimeError as e:
                # Re-raise as ValueError so callers have one exception type for bad configs.
                raise ValueError(f"invalid device {self.device!r}") from e

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        # Path is not JSON-serializable (and not allowed by torch.load(weights_only=True)).
        if self.checkpoint_dir is not None:
            d["checkpoint_dir"] = str(self.checkpoint_dir)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            # Silently dropping keys hides typos like "learning_rate".
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        d = dict(d)  # don't mutate the caller's dict
        if d.get("checkpoint_dir") is not None:
            d["checkpoint_dir"] = Path(d["checkpoint_dir"])
        return cls(**d)


# --------------------------------------------------------------------------------------
# 2. Data
# --------------------------------------------------------------------------------------


class ArrayDataset(Dataset[Batch]):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"features and targets differ in length: {features.shape[0]} vs {targets.shape[0]}"
            )
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Batch:
        return self.features[index], self.targets[index]


def make_dataloaders(
    train_ds: Dataset[Batch],
    val_ds: Dataset[Batch] | None = None,
    *,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader[Batch], DataLoader[Batch] | None]:
    # A private generator makes the shuffle order depend only on `seed`, not on how many
    # random numbers the rest of the program happened to draw from the global RNG.
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    # Validation order never matters, so don't shuffle it.
    val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


# --------------------------------------------------------------------------------------
# 3. Callbacks
# --------------------------------------------------------------------------------------


class Callback:
    # A base class with no-op hooks (rather than an ABC/Protocol with required methods):
    # concrete callbacks override only the hooks they care about.
    def on_train_start(self, trainer: Trainer) -> None:
        pass

    def on_step_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        pass


class EarlyStopping(Callback):
    def __init__(self, patience: int = 3, min_delta: float = 0.0, monitor: str = "val_loss"):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best = math.inf
        self.wait = 0

    def on_train_start(self, trainer: Trainer) -> None:
        self.best = math.inf
        self.wait = 0

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        value = metrics[self.monitor]  # KeyError on a typo'd monitor is better than never stopping
        if value < self.best - self.min_delta:
            self.best = value
            self.wait = 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            logger.info("early stopping: no %s improvement for %d epochs", self.monitor, self.wait)
            trainer.should_stop = True


class HistoryCallback(Callback):
    def __init__(self) -> None:
        self.history: list[dict[str, float]] = []

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        self.history.append(dict(metrics))  # copy: the trainer owns `metrics`


# --------------------------------------------------------------------------------------
# 4. Trainer
# --------------------------------------------------------------------------------------


def warmup_cosine_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0  # no decay horizon (yet): hold the peak lr
    progress = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        loss_fn: LossFn,
        callbacks: Sequence[Callback] = (),
    ) -> None:
        self.config = config
        self.device = get_device(config.device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.callbacks = list(callbacks)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        # fit() sets the real horizon (epochs * len(train_loader)). The lambda reads it lazily
        # on every scheduler.step(), so the scheduler can exist (and be loaded from a
        # checkpoint) before the loader is known. factor(0) does not depend on the horizon.
        self.total_steps = 0
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: warmup_cosine_factor(step, config.warmup_steps, self.total_steps),
        )
        self.epoch = 0
        self.global_step = 0
        self.should_stop = False

    def _autocast(self) -> torch.autocast:
        # bfloat16 autocast works on CPU and MPS in this torch version. Unlike float16, bf16 has
        # float32's exponent range, so no GradScaler is needed.
        return torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.config.use_amp
        )

    def _train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        x, y = x.to(self.device), y.to(self.device)
        # set_to_none: cheaper than zero-filling, and grads stay populated after step() so
        # on_step_end callbacks can inspect them.
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            loss = self.loss_fn(self.model(x), y)
        loss.backward()

        params = [p for p in self.model.parameters() if p.grad is not None]
        if self.config.grad_clip is not None:
            # Returns the total norm BEFORE clipping; grads are rescaled in place.
            grad_norm = nn.utils.clip_grad_norm_(params, self.config.grad_clip)
        else:
            # (the `is not None` repeat narrows the type for mypy)
            grad_norm = nn.utils.get_total_norm([p.grad for p in params if p.grad is not None])

        lr = self.optimizer.param_groups[0]["lr"]  # the lr THIS step uses
        self.optimizer.step()
        self.scheduler.step()  # per-step schedule: step after every optimizer.step()
        self.global_step += 1
        # .item() syncs with the device every step; fine at this scale, and callbacks want floats.
        return {
            "step": float(self.global_step),
            "loss": loss.item(),
            "lr": lr,
            "grad_norm": grad_norm.item(),
        }

    @staticmethod
    def _reseed_loader(loader: DataLoader[Batch], seed: int) -> None:
        # Make epoch e's shuffle a pure function of (seed, e), like DistributedSampler.set_epoch.
        # That is what lets a resumed run see exactly the batches the uninterrupted run saw.
        if loader.generator is not None:
            loader.generator.manual_seed(seed)

    def fit(
        self, train_loader: DataLoader[Batch], val_loader: DataLoader[Batch] | None = None
    ) -> list[dict[str, float]]:
        cfg = self.config
        self.total_steps = cfg.epochs * len(train_loader)
        self.should_stop = False
        history: list[dict[str, float]] = []
        logger.info(
            "starting training on %s: epochs %d..%d, %d steps/epoch",
            self.device,
            self.epoch + 1,
            cfg.epochs,
            len(train_loader),
        )
        for cb in self.callbacks:
            cb.on_train_start(self)

        # Start from self.epoch so a trainer restored from a checkpoint continues, not restarts.
        for epoch in range(self.epoch, cfg.epochs):
            self._reseed_loader(train_loader, cfg.seed + epoch)
            self.model.train()
            loss_sum, n_seen = 0.0, 0
            for x, y in train_loader:
                step_metrics = self._train_step(x, y)
                loss_sum += step_metrics["loss"] * x.shape[0]
                n_seen += x.shape[0]
                if self.global_step % cfg.log_every == 0:
                    # %-style args: the string is only formatted if the record is emitted.
                    logger.info(
                        "step %d loss=%.4f lr=%.3e grad_norm=%.3f",
                        self.global_step,
                        step_metrics["loss"],
                        step_metrics["lr"],
                        step_metrics["grad_norm"],
                    )
                for cb in self.callbacks:
                    cb.on_step_end(self, step_metrics)

            self.epoch = epoch + 1
            metrics = {
                "epoch": float(self.epoch),
                "train_loss": loss_sum / max(n_seen, 1),
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            if val_loader is not None:
                metrics["val_loss"] = self.evaluate(val_loader)["loss"]
            logger.info(
                "epoch %d/%d %s",
                self.epoch,
                cfg.epochs,
                " ".join(f"{k}={v:.4g}" for k, v in metrics.items() if k != "epoch"),
            )
            history.append(metrics)
            for cb in self.callbacks:
                cb.on_epoch_end(self, metrics)
            # Save AFTER callbacks so the checkpoint reflects their decisions for this epoch.
            if cfg.checkpoint_dir is not None:
                self.save_checkpoint(cfg.checkpoint_dir / "last.pt")
            if self.should_stop:
                logger.info("stopping early after epoch %d", self.epoch)
                break

        for cb in self.callbacks:
            cb.on_train_end(self)
        return history

    def evaluate(self, loader: DataLoader[Batch]) -> dict[str, float]:
        was_training = self.model.training  # restore the caller's mode, whatever it was
        self.model.eval()  # dropout off, batchnorm uses running stats
        loss_sum, n_seen = 0.0, 0
        try:
            # no_grad: don't record a graph (saves memory); inference_mode would also work but
            # its tensors can never be used in autograd later, which surprises some callers.
            with torch.no_grad(), self._autocast():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    loss = self.loss_fn(self.model(x), y)
                    loss_sum += loss.item() * x.shape[0]
                    n_seen += x.shape[0]
        finally:
            self.model.train(was_training)  # even if evaluation raised
        return {"loss": loss_sum / max(n_seen, 1)}

    def _rng_state(self) -> dict[str, Any]:
        # Dropout draws from the generator of the device it runs on: the CPU generator on CPU,
        # the MPS/CUDA generator on an accelerator. Save all that apply, or a resumed run
        # diverges. (Both device getters return CPU byte tensors, safe for weights_only.)
        rng: dict[str, Any] = {"torch": torch.get_rng_state(), "python": random.getstate()}
        if self.device.type == "mps":
            rng["mps"] = torch.mps.get_rng_state()
        elif self.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(self.device)
        return rng

    def _set_rng_state(self, rng: dict[str, Any]) -> None:
        torch.set_rng_state(rng["torch"])
        random.setstate(rng["python"])
        if "mps" in rng and self.device.type == "mps":
            torch.mps.set_rng_state(rng["mps"])
        elif "cuda" in rng and self.device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], self.device)

    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),  # Adam moments + current lr
            "scheduler": self.scheduler.state_dict(),  # last_epoch == global_step
            "epoch": self.epoch,
            "global_step": self.global_step,
            "config": self.config.to_dict(),
            "rng": self._rng_state(),
        }
        # Write to a temp file then rename: a process crash mid-save never leaves a truncated
        # checkpoint under the real name (os.replace is atomic on POSIX). Surviving power loss
        # would also need an fsync before the rename.
        tmp = path.with_name(path.name + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
        logger.info("saved checkpoint to %s (epoch %d)", path, self.epoch)

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        # weights_only=True (the default since torch 2.6) refuses arbitrary pickled objects, so
        # loading an untrusted file can't execute code. That's why the config is a plain dict.
        # map_location="cpu": RNG state must be a CPU tensor; load_state_dict moves the rest.
        state = torch.load(path, map_location="cpu", weights_only=True)
        saved_config = TrainConfig.from_dict(state["config"])
        if saved_config != self.config:
            logger.warning("checkpoint config differs from the trainer's config: %s", saved_config)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self._set_rng_state(state["rng"])
        logger.info("loaded checkpoint from %s (epoch %d)", path, self.epoch)
