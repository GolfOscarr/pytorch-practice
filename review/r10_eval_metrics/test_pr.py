import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from review.r10_eval_metrics.pr import (
    Accuracy,
    BinaryAUROC,
    EarlyStopping,
    F1Score,
    MeanLoss,
    evaluate,
    evaluate_binary,
    f1_at_threshold,
    select_threshold,
)

# --------------------------------------------------------------------------- helpers


def one_hot_logits(classes: list[int], num_classes: int) -> Tensor:
    """Logits whose argmax over the class dim is `classes`."""
    return F.one_hot(torch.tensor(classes), num_classes).float() * 5.0


def batches(x: Tensor, y: Tensor, sizes: list[int]) -> list[tuple[Tensor, Tensor]]:
    assert sum(sizes) == len(x)
    out, start = [], 0
    for size in sizes:
        out.append((x[start : start + size], y[start : start + size]))
        start += size
    return out


def bruteforce_auroc(scores: list[float], labels: list[int]) -> float:
    """P(score_pos > score_neg) + 0.5 * P(tie), over all positive/negative pairs."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def bruteforce_macro_f1(predicted: list[int], target: list[int], num_classes: int) -> float:
    """Per-class F1 = 2TP / (2TP + FP + FN), 0 when undefined; unweighted mean over classes."""
    scores = []
    for c in range(num_classes):
        tp = sum(p == c and t == c for p, t in zip(predicted, target, strict=True))
        fp = sum(p == c and t != c for p, t in zip(predicted, target, strict=True))
        fn = sum(p != c and t == c for p, t in zip(predicted, target, strict=True))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return sum(scores) / num_classes


class Linear3(nn.Module):
    """A fixed 4 -> 3 classifier with dropout, so train/eval mode is observable."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(4, 3)
        self.drop = nn.Dropout(0.5)
        self.grad_enabled_in_forward: list[bool] = []

    def forward(self, x: Tensor) -> Tensor:
        self.grad_enabled_in_forward.append(torch.is_grad_enabled())
        return self.fc(self.drop(x))


class Passthrough(nn.Module):
    """'Model' whose input already is the score; carries one parameter so it has a device."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.dummy


# --------------------------------------------------------------------------- Accuracy


def test_accuracy_counts_correct_rows_across_batches():
    metric = Accuracy()
    # batch 1: rows predict [0, 1, 2, 2] vs target [0, 1, 1, 2] -> 3/4 correct
    metric.update(one_hot_logits([0, 1, 2, 2], 3), torch.tensor([0, 1, 1, 2]))
    # batch 2: rows predict [1, 0] vs target [1, 1] -> 1/2 correct
    metric.update(one_hot_logits([1, 0], 3), torch.tensor([1, 1]))
    assert metric.compute() == pytest.approx(4 / 6)


def test_accuracy_square_logits_still_use_class_dimension():
    # N == C == 3: row-wise argmax is [2, 2, 0]; target matches rows 0 and 1 only
    logits = torch.tensor([[0.0, 1.0, 9.0], [0.0, 2.0, 3.0], [5.0, 0.0, 1.0]])
    metric = Accuracy()
    metric.update(logits, torch.tensor([2, 2, 1]))
    assert metric.compute() == pytest.approx(2 / 3)


def test_accuracy_reset_starts_from_scratch():
    metric = Accuracy()
    metric.update(one_hot_logits([0, 0], 3), torch.tensor([1, 1]))
    metric.reset()
    metric.update(one_hot_logits([0, 1], 3), torch.tensor([0, 1]))
    assert metric.compute() == pytest.approx(1.0)


# --------------------------------------------------------------------------- MeanLoss


def test_mean_loss_equal_batches_matches_full_dataset_loss():
    torch.manual_seed(1)
    logits, target = torch.randn(8, 3), torch.randint(0, 3, (8,))
    metric = MeanLoss(F.cross_entropy)
    for xb, yb in batches(logits, target, [4, 4]):
        metric.update(xb, yb)
    assert metric.compute() == pytest.approx(F.cross_entropy(logits, target).item(), rel=1e-6)


def test_mean_loss_is_per_sample_average_with_short_last_batch():
    torch.manual_seed(2)
    logits, target = torch.randn(9, 3), torch.randint(0, 3, (9,))
    metric = MeanLoss(F.cross_entropy)
    for xb, yb in batches(logits, target, [4, 4, 1]):
        metric.update(xb, yb)
    assert metric.compute() == pytest.approx(F.cross_entropy(logits, target).item(), rel=1e-6)


# --------------------------------------------------------------------------- BinaryAUROC


def test_auroc_hand_example_without_ties():
    # positives {0.8, 0.4}, negatives {0.6, 0.1}: pairs won = (0.8>0.6, 0.8>0.1, 0.4>0.1) = 3/4
    metric = BinaryAUROC()
    metric.update(torch.tensor([0.8, 0.6]), torch.tensor([1, 0]))
    metric.update(torch.tensor([0.4, 0.1]), torch.tensor([1, 0]))
    assert metric.compute() == pytest.approx(0.75)


def test_auroc_matches_pairwise_bruteforce_on_distinct_scores():
    g = torch.Generator().manual_seed(3)
    scores = torch.randn(40, generator=g)
    labels = (torch.rand(40, generator=g) < 0.4).long()
    metric = BinaryAUROC()
    metric.update(scores, labels)
    expected = bruteforce_auroc(scores.tolist(), labels.tolist())
    assert metric.compute() == pytest.approx(expected)


def test_auroc_counts_tied_scores_as_half_a_win():
    # Coarse scores (e.g. a calibrated model rounded to 1 decimal) produce many ties.
    g = torch.Generator().manual_seed(4)
    scores = torch.randint(0, 3, (30,), generator=g).float() / 2
    labels = (torch.rand(30, generator=g) < 0.5).long()
    metric = BinaryAUROC()
    metric.update(scores, labels)
    expected = bruteforce_auroc(scores.tolist(), labels.tolist())
    assert metric.compute() == pytest.approx(expected)


def test_auroc_of_constant_scores_is_one_half():
    metric = BinaryAUROC()
    metric.update(torch.full((6,), 0.3), torch.tensor([1, 1, 1, 0, 0, 0]))
    assert metric.compute() == pytest.approx(0.5)


# --------------------------------------------------------------------------- F1Score


def test_macro_f1_hand_example_every_class_predicted():
    # predicted [0,0,1,1,2,2] vs target [0,1,1,1,2,0]
    # class 0: tp1 fp1 fn1 -> 1/2; class 1: tp2 fp0 fn1 -> 4/5; class 2: tp1 fp1 fn0 -> 2/3
    metric = F1Score(num_classes=3, average="macro")
    metric.update(one_hot_logits([0, 0, 1, 1, 2, 2], 3), torch.tensor([0, 1, 1, 1, 2, 0]))
    assert metric.compute() == pytest.approx((1 / 2 + 4 / 5 + 2 / 3) / 3)


def test_macro_f1_class_never_predicted_scores_zero_not_nan():
    # Class 2 is never predicted (a common failure mode early in training).
    predicted, target = [0, 1, 1, 0, 0], [0, 1, 2, 2, 1]
    metric = F1Score(num_classes=3, average="macro")
    metric.update(one_hot_logits(predicted, 3), torch.tensor(target))
    value = metric.compute()
    assert math.isfinite(value)
    assert value == pytest.approx(bruteforce_macro_f1(predicted, target, 3))


def test_macro_f1_matches_bruteforce_on_random_predictions():
    g = torch.Generator().manual_seed(5)
    logits = torch.randn(50, 4, generator=g)
    target = torch.randint(0, 4, (50,), generator=g)
    metric = F1Score(num_classes=4, average="macro")
    for xb, yb in batches(logits, target, [20, 20, 10]):
        metric.update(xb, yb)
    expected = bruteforce_macro_f1(logits.argmax(-1).tolist(), target.tolist(), 4)
    assert metric.compute() == pytest.approx(expected)


def test_micro_f1_equals_accuracy_for_single_label_multiclass():
    g = torch.Generator().manual_seed(6)
    logits = torch.randn(30, 4, generator=g)
    target = torch.randint(0, 4, (30,), generator=g)
    metric = F1Score(num_classes=4, average="micro")
    metric.update(logits, target)
    accuracy = (logits.argmax(-1) == target).float().mean().item()
    assert metric.compute() == pytest.approx(accuracy)


def test_rejects_unknown_average():
    with pytest.raises(ValueError):
        F1Score(num_classes=3, average="weighted")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- evaluate


def make_classification_loader(seed: int, sizes: list[int]) -> list[tuple[Tensor, Tensor]]:
    g = torch.Generator().manual_seed(seed)
    n = sum(sizes)
    x = torch.randn(n, 4, generator=g)
    y = torch.randint(0, 3, (n,), generator=g)
    return batches(x, y, sizes)


def test_evaluate_matches_full_batch_computation_with_ragged_batches():
    model = Linear3()
    loader = make_classification_loader(seed=7, sizes=[5, 5, 2])
    result = evaluate(model, loader, {"acc": Accuracy(), "loss": MeanLoss(F.cross_entropy)})

    x = torch.cat([xb for xb, _ in loader])
    y = torch.cat([yb for _, yb in loader])
    model.eval()
    with torch.no_grad():
        logits = model(x)
    assert result["acc"] == pytest.approx((logits.argmax(-1) == y).float().mean().item())
    assert result["loss"] == pytest.approx(F.cross_entropy(logits, y).item(), rel=1e-6)


def test_evaluate_twice_with_same_metric_objects_gives_fresh_results():
    model = Linear3()
    epoch_1 = make_classification_loader(seed=8, sizes=[6, 6])
    epoch_2 = make_classification_loader(seed=9, sizes=[6, 6])
    metrics = {"f1": F1Score(num_classes=3, average="micro"), "acc": Accuracy()}
    evaluate(model, epoch_1, metrics)
    reused = evaluate(model, epoch_2, metrics)
    fresh = evaluate(model, epoch_2, {"f1": F1Score(3, average="micro"), "acc": Accuracy()})
    assert reused == pytest.approx(fresh)


def test_evaluate_restores_training_mode():
    model = Linear3()
    model.train()
    evaluate(model, make_classification_loader(seed=10, sizes=[4]), {"acc": Accuracy()})
    assert model.training
    assert model.drop.training

    model.eval()
    evaluate(model, make_classification_loader(seed=10, sizes=[4]), {"acc": Accuracy()})
    assert not model.training


def test_evaluate_does_not_build_autograd_graph():
    model = Linear3()
    evaluate(model, make_classification_loader(seed=11, sizes=[4, 4]), {"acc": Accuracy()})
    assert model.grad_enabled_in_forward == [False, False]


# --------------------------------------------------------------------------- thresholds


def test_select_threshold_matches_bruteforce_over_all_cutoffs():
    g = torch.Generator().manual_seed(12)
    scores = torch.rand(25, generator=g)
    labels = (scores + 0.3 * torch.randn(25, generator=g) > 0.5).long()
    threshold, f1 = select_threshold(scores, labels)
    best = max(f1_at_threshold(scores, labels, t) for t in scores.tolist())
    assert f1 == pytest.approx(best)
    assert f1_at_threshold(scores, labels, threshold) == pytest.approx(best)


def test_evaluate_binary_tunes_threshold_on_validation_split():
    # On val the F1-optimal cut is 0.6; on test it would be 0.3. Tuning must use val only.
    val_scores = torch.tensor([0.1, 0.3, 0.4, 0.6, 0.7, 0.9])
    val_labels = torch.tensor([0, 0, 0, 1, 1, 1])
    test_scores = torch.tensor([0.2, 0.3, 0.5, 0.65, 0.8, 0.95])
    test_labels = torch.tensor([0, 1, 1, 1, 1, 1])
    val_threshold, _ = select_threshold(val_scores, val_labels)
    test_oracle, _ = select_threshold(test_scores, test_labels)
    assert val_threshold == pytest.approx(0.6) and test_oracle == pytest.approx(0.3)

    result = evaluate_binary(
        Passthrough(),
        val_loader=batches(val_scores, val_labels, [3, 3]),
        test_loader=batches(test_scores, test_labels, [4, 2]),
    )
    assert result["threshold"] == pytest.approx(val_threshold)
    assert result["test_f1"] == pytest.approx(f1_at_threshold(test_scores, test_labels, 0.6))
    assert result["val_auroc"] == pytest.approx(1.0)
    assert result["test_auroc"] == pytest.approx(
        bruteforce_auroc(test_scores.tolist(), test_labels.tolist())
    )


# --------------------------------------------------------------------------- EarlyStopping


def run_early_stopping(stopper: EarlyStopping, values: list[float]) -> int | None:
    """Feed one value per epoch; return the 1-based epoch at which it asked to stop."""
    for epoch, value in enumerate(values, start=1):
        if stopper.step({stopper.monitor: value}):
            return epoch
    return None


def test_early_stopping_stops_after_exactly_patience_epochs_without_improvement():
    # best at epoch 1, then 2 epochs without improvement -> stop at epoch 3 with patience=2
    stopper = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    assert run_early_stopping(stopper, [1.0, 1.1, 1.2, 1.3, 1.4]) == 3
    assert stopper.best == pytest.approx(1.0)


def test_early_stopping_improvement_resets_the_counter():
    stopper = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    # 2 bad epochs, an improvement, then 3 bad epochs -> stop at epoch 7
    assert run_early_stopping(stopper, [1.0, 1.2, 1.1, 0.9, 0.95, 0.91, 1.0, 0.5]) == 7


def test_early_stopping_max_mode_and_min_delta():
    stopper = EarlyStopping(monitor="val_acc", mode="max", patience=1, min_delta=0.01)
    # 0.705 is not a big enough gain over 0.70, so it counts as a bad epoch
    assert run_early_stopping(stopper, [0.60, 0.70, 0.705, 0.80]) == 3
    assert stopper.best == pytest.approx(0.70)
