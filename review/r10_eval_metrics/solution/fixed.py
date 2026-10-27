"""Reference fix for review/r10_eval_metrics/pr.py (same public API, bugs fixed, hardened).

See solution/REVIEW.md for the answer key; `# FIX(Bn)` comments map lines to it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import Literal

import torch
from torch import Tensor, nn

Batch = tuple[Tensor, Tensor]
LossFn = Callable[[Tensor, Tensor], Tensor]


class Metric(ABC):
    """Streaming metric: `update` once per batch, `compute` at the end, `reset` between runs."""

    @abstractmethod
    def update(self, preds: Tensor, target: Tensor) -> None: ...

    @abstractmethod
    def compute(self) -> float: ...

    @abstractmethod
    def reset(self) -> None: ...


def _check_logits(preds: Tensor, target: Tensor) -> None:
    if preds.ndim != 2 or target.ndim != 1 or preds.shape[0] != target.shape[0]:
        raise ValueError(
            f"expected logits (N, C) and target (N,), got {tuple(preds.shape)} "
            f"and {tuple(target.shape)}"
        )


class Accuracy(Metric):
    """Multiclass top-1 accuracy. `preds` are logits of shape (N, C), `target` is (N,)."""

    def __init__(self) -> None:
        self.reset()

    def update(self, preds: Tensor, target: Tensor) -> None:
        _check_logits(preds, target)
        predicted = preds.argmax(dim=-1)  # FIX(B1): class dim, not the batch dim
        self._correct += int((predicted == target).sum())
        self._total += target.numel()

    def compute(self) -> float:
        return self._correct / max(self._total, 1)

    def reset(self) -> None:
        self._correct = 0
        self._total = 0


class F1Score(Metric):
    """Multiclass F1 from logits (N, C). `average="macro"` weights every class equally.

    Every class in `range(num_classes)` is averaged, and a class whose F1 is undefined
    contributes 0, so the macro average never becomes NaN. This equals sklearn's
    `f1_score(..., labels=range(num_classes), average="macro", zero_division=0)`. (sklearn's
    default `labels=None` instead averages only classes present in y_true or y_pred.)
    """

    def __init__(self, num_classes: int, average: Literal["macro", "micro"] = "macro") -> None:
        if average not in ("macro", "micro"):
            raise ValueError(f"unknown average: {average!r}")
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def update(self, preds: Tensor, target: Tensor) -> None:
        _check_logits(preds, target)
        predicted = preds.detach().argmax(dim=-1).cpu()
        target = target.cpu()
        k = self.num_classes
        hits = predicted == target
        self._tp += torch.bincount(target[hits], minlength=k).double()
        self._fp += torch.bincount(predicted[~hits], minlength=k).double()
        self._fn += torch.bincount(target[~hits], minlength=k).double()

    def compute(self) -> float:
        tp, fp, fn = self._tp, self._fp, self._fn
        if self.average == "micro":
            tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
        # FIX(B4): F1 = 2TP / (2TP + FP + FN), defined as 0 when the denominator is 0.
        # The precision/recall form divides 0/0 for any class that is never predicted.
        denom = 2 * tp + fp + fn
        f1 = torch.where(denom > 0, 2 * tp / denom.clamp_min(1), torch.zeros_like(denom))
        return float(f1.mean())

    def reset(self) -> None:
        # FIX(B5): every accumulator is cleared (fn used to leak into the next evaluation).
        self._tp = torch.zeros(self.num_classes, dtype=torch.float64)
        self._fp = torch.zeros(self.num_classes, dtype=torch.float64)
        self._fn = torch.zeros(self.num_classes, dtype=torch.float64)


class BinaryAUROC(Metric):
    """Area under the ROC curve from raw scores (N,) and binary targets (N,).

    Uses the Mann-Whitney U statistic: AUROC is the probability that a random positive is scored
    above a random negative, counting ties as 1/2. Tied scores therefore get their average rank.
    """

    def __init__(self) -> None:
        self.reset()

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.numel() != target.numel():
            raise ValueError(f"preds has {preds.numel()} scores but target has {target.numel()}")
        self._scores.append(preds.detach().flatten().cpu())
        self._targets.append(target.detach().flatten().cpu())

    @property
    def scores(self) -> Tensor:
        return torch.cat(self._scores) if self._scores else torch.empty(0)

    @property
    def targets(self) -> Tensor:
        return torch.cat(self._targets) if self._targets else torch.empty(0)

    def compute(self) -> float:
        scores = self.scores.double()
        target = self.targets.bool()
        n_pos = int(target.sum())
        n_neg = target.numel() - n_pos
        if n_pos == 0 or n_neg == 0:
            return math.nan
        # FIX(B3): average ranks over ties. argsort ranks would order tied scores arbitrarily.
        _, inverse, counts = torch.unique(scores, return_inverse=True, return_counts=True)
        last_rank = counts.cumsum(0).double()
        avg_rank = last_rank - (counts.double() - 1) / 2
        ranks = avg_rank[inverse]
        u = ranks[target].sum() - n_pos * (n_pos + 1) / 2
        return float(u / (n_pos * n_neg))

    def reset(self) -> None:
        self._scores: list[Tensor] = []
        self._targets: list[Tensor] = []


class MeanLoss(Metric):
    """Per-sample average of `loss_fn(preds, target)` over the eval set.

    `loss_fn` returns a batch mean (the PyTorch default), so it is re-weighted by batch size.
    """

    def __init__(self, loss_fn: LossFn) -> None:
        self.loss_fn = loss_fn
        self._sum: Tensor | float = 0.0
        self._count = 0

    def update(self, preds: Tensor, target: Tensor) -> None:
        n = target.shape[0]
        # FIX(B2): weight each batch mean by its size; a mean of batch means over-weights the
        # (usually smaller) last batch. Accumulate as a tensor to avoid a host sync per batch.
        self._sum = self._sum + self.loss_fn(preds, target).detach().double() * n
        self._count += n

    def compute(self) -> float:
        return float(self._sum) / self._count if self._count else math.nan

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0


def _model_device(model: nn.Module) -> torch.device:
    param = next(model.parameters(), None)
    return param.device if param is not None else torch.device("cpu")


def evaluate(
    model: nn.Module, loader: Iterable[Batch], metrics: Mapping[str, Metric]
) -> dict[str, float]:
    """Run `model` over `loader` and return `{name: metric.compute()}` for every metric.

    The model's train/eval mode is restored afterwards, and no autograd graph is built.
    """
    device = _model_device(model)
    for metric in metrics.values():
        metric.reset()
    # Snapshot every module's flag: model.train(flag) would cascade and un-freeze submodules
    # the caller deliberately keeps in eval mode (e.g. frozen BatchNorm during fine-tuning).
    modes = {m: m.training for m in model.modules()}
    model.eval()
    try:
        # FIX(B8): no autograd graph during eval (memory/speed) ...
        with torch.no_grad():
            for inputs, target in loader:
                outputs = model(inputs.to(device, non_blocking=True))
                target = target.to(device, non_blocking=True)
                for metric in metrics.values():
                    metric.update(outputs, target)
    finally:
        for m, was_training in modes.items():  # FIX(B8): ... and the caller's modes restored
            m.training = was_training
    return {name: metric.compute() for name, metric in metrics.items()}


def f1_at_threshold(scores: Tensor, labels: Tensor, threshold: float) -> float:
    """Binary F1 when predicting positive for `scores >= threshold`."""
    predicted = scores >= threshold
    actual = labels.bool()
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def select_threshold(scores: Tensor, labels: Tensor) -> tuple[float, float]:
    """Return `(threshold, f1)` maximising binary F1 over the observed score values.

    Call this on VALIDATION scores only; reporting F1 on the same data it was tuned on is
    optimistically biased.
    """
    if scores.shape != labels.shape:
        raise ValueError(f"scores {tuple(scores.shape)} and labels {tuple(labels.shape)} differ")
    best_threshold, best_f1 = 0.5, -1.0
    for candidate in torch.unique(scores).tolist():
        f1 = f1_at_threshold(scores, labels, candidate)
        if f1 > best_f1:
            best_threshold, best_f1 = candidate, f1
    return best_threshold, best_f1


def collect_scores(model: nn.Module, loader: Iterable[Batch]) -> tuple[Tensor, Tensor]:
    """Concatenate the model's (N,) scores and the labels over a loader, on CPU."""
    auroc = BinaryAUROC()
    evaluate(model, loader, {"auroc": auroc})
    return auroc.scores, auroc.targets


def evaluate_binary(
    model: nn.Module, val_loader: Iterable[Batch], test_loader: Iterable[Batch]
) -> dict[str, float]:
    """Tune the decision threshold for F1 on val, then report val/test AUROC and test F1."""
    val_scores, val_labels = collect_scores(model, val_loader)
    test_scores, test_labels = collect_scores(model, test_loader)
    # FIX(B6): the threshold is a hyperparameter -> choose it on val, never on test.
    threshold, _ = select_threshold(val_scores, val_labels)
    val_auroc, test_auroc = BinaryAUROC(), BinaryAUROC()
    val_auroc.update(val_scores, val_labels)
    test_auroc.update(test_scores, test_labels)
    return {
        "threshold": threshold,
        "val_auroc": val_auroc.compute(),
        "test_auroc": test_auroc.compute(),
        "test_f1": f1_at_threshold(test_scores, test_labels, threshold),
    }


class EarlyStopping:
    """Stop training when `monitor` has not improved for `patience` consecutive epochs.

    `mode="min"` for losses, `mode="max"` for scores; an epoch counts as an improvement only if
    it beats the best value so far by more than `min_delta`.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        patience: int = 3,
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"unknown mode: {mode!r}")
        if patience < 1 or min_delta < 0:
            raise ValueError("patience must be >= 1 and min_delta >= 0")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf if mode == "min" else -math.inf
        self.num_bad_epochs = 0
        self.should_stop = False

    def _improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, metrics: Mapping[str, float]) -> bool:
        """Record one epoch's metrics; return True when training should stop."""
        if self.monitor not in metrics:
            raise KeyError(f"EarlyStopping monitors {self.monitor!r}, got {sorted(metrics)}")
        value = float(metrics[self.monitor])
        if math.isnan(value):
            raise ValueError(f"{self.monitor} is NaN")
        if self._improved(value):
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        # FIX(B7): stop after exactly `patience` epochs without improvement (was patience + 1).
        self.should_stop = self.num_bad_epochs >= self.patience
        return self.should_stop
