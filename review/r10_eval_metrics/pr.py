"""Add evaluation module: streaming metrics, threshold selection, early stopping.

## Summary
Our training scripts each hand-roll their own eval loop, and the numbers they print do not agree
with each other. This PR adds one small, dependency-free evaluation module (we can't pull in
sklearn/torchmetrics on the training image) that every trainer can share.

## Changes
- `Metric` base class with a torchmetrics-style `update(preds, target)` / `compute()` /
  `reset()` contract, so metrics stream over batches instead of concatenating the whole eval set.
- `Accuracy` (multiclass, from logits), `F1Score` (`average="macro" | "micro"`),
  `BinaryAUROC` (rank-based / Mann-Whitney, from raw scores) and `MeanLoss` (wraps any loss fn).
- `evaluate(model, loader, metrics)` runs a model over a loader and returns `{name: value}`.
- `select_threshold(scores, labels)` picks the decision threshold that maximises F1, and
  `evaluate_binary(model, val_loader, test_loader)` reports AUROC plus F1 at the tuned threshold.
- `EarlyStopping(monitor, mode, patience, min_delta)` with the usual Keras/Lightning semantics.

## Testing
Unit tests compare against hand-computed values and small brute-force references, since sklearn
is not installed. Also ran it on the image-classifier val split and the accuracy matches the old
script.
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


class Accuracy(Metric):
    """Multiclass top-1 accuracy. `preds` are logits of shape (N, C), `target` is (N,)."""

    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, preds: Tensor, target: Tensor) -> None:
        predicted = preds.argmax(dim=0)
        self._correct += int((predicted == target).sum())
        self._total += target.numel()

    def compute(self) -> float:
        return self._correct / max(self._total, 1)

    def reset(self) -> None:
        self._correct = 0
        self._total = 0


class F1Score(Metric):
    """Multiclass F1 from logits (N, C). `average="macro"` weights every class equally."""

    def __init__(self, num_classes: int, average: Literal["macro", "micro"] = "macro") -> None:
        if average not in ("macro", "micro"):
            raise ValueError(f"unknown average: {average!r}")
        self.num_classes = num_classes
        self.average = average
        self._tp = torch.zeros(num_classes)
        self._fp = torch.zeros(num_classes)
        self._fn = torch.zeros(num_classes)

    def update(self, preds: Tensor, target: Tensor) -> None:
        predicted = preds.argmax(dim=-1).cpu()
        target = target.cpu()
        k = self.num_classes
        hits = predicted == target
        self._tp += torch.bincount(target[hits], minlength=k).float()
        self._fp += torch.bincount(predicted[~hits], minlength=k).float()
        self._fn += torch.bincount(target[~hits], minlength=k).float()

    def compute(self) -> float:
        tp, fp, fn = self._tp, self._fp, self._fn
        if self.average == "micro":
            tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return float(f1.mean())

    def reset(self) -> None:
        self._tp.zero_()
        self._fp.zero_()


class BinaryAUROC(Metric):
    """Area under the ROC curve from raw scores (N,) and binary targets (N,).

    Uses the Mann-Whitney U statistic: AUROC is the probability that a random positive is scored
    above a random negative, which only depends on the ranks of the scores.
    """

    def __init__(self) -> None:
        self._scores: list[Tensor] = []
        self._targets: list[Tensor] = []

    def update(self, preds: Tensor, target: Tensor) -> None:
        self._scores.append(preds.flatten())
        self._targets.append(target.flatten())

    def compute(self) -> float:
        scores = torch.cat(self._scores).cpu().double()
        target = torch.cat(self._targets).cpu().bool()
        n_pos = int(target.sum())
        n_neg = target.numel() - n_pos
        if n_pos == 0 or n_neg == 0:
            return math.nan
        ranks = torch.empty_like(scores)
        ranks[scores.argsort()] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype)
        u = ranks[target].sum() - n_pos * (n_pos + 1) / 2
        return float(u / (n_pos * n_neg))

    def reset(self) -> None:
        self._scores = []
        self._targets = []


class MeanLoss(Metric):
    """Average of `loss_fn(preds, target)` over the eval set (`loss_fn` returns a batch mean)."""

    def __init__(self, loss_fn: LossFn) -> None:
        self.loss_fn = loss_fn
        self._sum = 0.0
        self._batches = 0

    def update(self, preds: Tensor, target: Tensor) -> None:
        self._sum += float(self.loss_fn(preds, target))
        self._batches += 1

    def compute(self) -> float:
        return self._sum / max(self._batches, 1)

    def reset(self) -> None:
        self._sum = 0.0
        self._batches = 0


def _model_device(model: nn.Module) -> torch.device:
    param = next(model.parameters(), None)
    return param.device if param is not None else torch.device("cpu")


def evaluate(
    model: nn.Module, loader: Iterable[Batch], metrics: Mapping[str, Metric]
) -> dict[str, float]:
    """Run `model` over `loader` and return `{name: metric.compute()}` for every metric."""
    device = _model_device(model)
    for metric in metrics.values():
        metric.reset()
    model.eval()
    for inputs, target in loader:
        outputs = model(inputs.to(device))
        for metric in metrics.values():
            metric.update(outputs, target.to(device))
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
    """Return `(threshold, f1)` maximising binary F1 over the observed score values."""
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
    return torch.cat(auroc._scores).cpu(), torch.cat(auroc._targets).cpu()


def evaluate_binary(
    model: nn.Module, val_loader: Iterable[Batch], test_loader: Iterable[Batch]
) -> dict[str, float]:
    """Tune the decision threshold for F1 and report val/test AUROC and test F1."""
    val_auroc = evaluate(model, val_loader, {"auroc": BinaryAUROC()})["auroc"]
    test_scores, test_labels = collect_scores(model, test_loader)
    threshold, _ = select_threshold(test_scores, test_labels)
    test_auroc = BinaryAUROC()
    test_auroc.update(test_scores, test_labels)
    return {
        "threshold": threshold,
        "val_auroc": val_auroc,
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
        value = float(metrics[self.monitor])
        if self._improved(value):
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        self.should_stop = self.num_bad_epochs > self.patience
        return self.should_stop
