# r10_eval_metrics: answer key

PR: "Add evaluation module: streaming metrics, threshold selection, early stopping"

## Bugs

| ID | Severity | Location | What's wrong | Impact |
|----|----------|----------|--------------|--------|
| B1 | High | pr.py:59 | `Accuracy` takes `argmax(dim=0)` (over the batch) instead of the class dim | crash when N != C; silently wrong when N == C |
| B2 | High | pr.py:147-148 | `MeanLoss` averages per-batch means, so a short last batch is over-weighted | silently wrong |
| B3 | Medium | pr.py:128-129 | AUROC ranks come from `argsort`, so tied scores get arbitrary distinct ranks instead of their average rank | silently wrong |
| B4 | High | pr.py:96-98 | F1 is computed as `2PR/(P+R)` with no zero-division rule: a class that is never predicted (or has TP = 0) gives 0/0, and the macro mean becomes NaN | silently wrong (NaN) |
| B5 | Medium | pr.py:101-103 | `F1Score.reset()` clears `tp` and `fp` but not `fn`, so false negatives carry over between `evaluate()` calls | silently wrong |
| B6 | Critical | pr.py:212 | `evaluate_binary` tunes the threshold on the **test** scores and reports test F1 at that threshold (data snooping) | silently wrong (optimistic) |
| B7 | Medium | pr.py:260 | `EarlyStopping` checks `num_bad_epochs > patience`, so it stops one epoch late (patience + 1) | silently wrong |
| B8 | High | pr.py:170-174 | `evaluate()` runs with autograd on (no `no_grad`) and leaves the model in `eval()` mode after it returns | perf/memory + silently wrong training |

### B1: argmax over the batch dimension

Logits are `(N, C)`. The predicted class of each row is `argmax(dim=-1)` (equivalently `dim=1`).
`dim=0` gives, for each class, the row with the highest logit, which is a `(C,)` tensor. If N != C,
comparing it with `target` `(N,)` fails to broadcast and crashes. If N == C, for example a 10-class
model evaluated with batch size 10, it runs and returns nonsense. That second case is the dangerous
one.

```python
predicted = preds.argmax(dim=0)  # buggy
predicted = preds.argmax(dim=-1)  # fixed (+ shape validation via _check_logits)
```

Caught by `test_accuracy_counts_correct_rows_across_batches`,
`test_accuracy_square_logits_still_use_class_dimension` and every `evaluate` test that uses
`Accuracy`.

### B2: mean of batch means is not the dataset mean

`F.cross_entropy` returns the *batch* mean. Averaging B of those gives each batch equal weight, so
the 1 sample in a short final batch counts as much as the 4 in each full batch. The dataset mean
weights each batch mean by its size. Equal-size batches hide the bug, which is why
`test_mean_loss_equal_batches_matches_full_dataset_loss` passes on pr.py.

```python
self._sum += float(self.loss_fn(preds, target))  # buggy
self._batches += 1
...
self._sum = self._sum + self.loss_fn(preds, target).detach().double() * n  # fixed
self._count += n
```

Caught by `test_mean_loss_is_per_sample_average_with_short_last_batch` and
`test_evaluate_matches_full_batch_computation_with_ragged_batches`.

### B3: AUROC with tied scores

AUROC = P(score_pos > score_neg) + 0.5 * P(tie). In the Mann-Whitney form this means each group of
tied scores gets its **average rank**. `argsort` gives tied scores distinct ranks in whatever order
the sort leaves them, so the result depends on the input order. For example, constant scores with
labels `[1,1,1,0,0,0]` give 0.0 instead of 0.5. Ties are common in practice: quantized or rounded
scores, saturated sigmoids, tree models, or hard 0/1 predictions passed in as "scores".

```python
ranks[scores.argsort()] = torch.arange(1, n + 1, dtype=scores.dtype)  # buggy
_, inverse, counts = torch.unique(scores, return_inverse=True, return_counts=True)
ranks = (counts.cumsum(0) - (counts - 1) / 2).double()[inverse]  # fixed
```

Caught by `test_auroc_counts_tied_scores_as_half_a_win` and `test_auroc_of_constant_scores_is_one_half`.
The tests without ties pass on pr.py.

### B4: 0/0 in per-class F1 makes the macro average NaN

For a class with no predictions, precision = 0/0 = NaN. For any class with TP = 0, P = R = 0 and
`2PR/(P+R)` = 0/0. A single NaN class turns the macro mean into NaN. This happens all the time
early in training, when the model predicts only the majority class. NaN then propagates into
logging, checkpoint selection (`nan > best` is False), and early stopping. The usual convention
(sklearn `zero_division=0`) is that an undefined F1 is 0. Using `F1 = 2TP / (2TP + FP + FN)`
handles this with one guard.

```python
precision = tp / (tp + fp)  # buggy
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
denom = 2 * tp + fp + fn  # fixed
f1 = torch.where(denom > 0, 2 * tp / denom.clamp_min(1), torch.zeros_like(denom))
```

Caught by `test_macro_f1_class_never_predicted_scores_zero_not_nan`.

### B5: `reset()` leaks state across evaluations

`evaluate()` calls `reset()` on each metric so the same metric objects can be reused every epoch.
`F1Score.reset()` zeroes `tp` and `fp` but forgets `fn`, so epoch k's F1 still includes the false
negatives from every earlier epoch. The value stays plausible and slowly drifts, so nobody notices.
The fix has `__init__` call `reset()`, which is the single place where state is created. Then a new
accumulator cannot be added without also being reset.

```python
def reset(self) -> None:  # buggy
    self._tp.zero_()
    self._fp.zero_()


def reset(self) -> None:  # fixed (and __init__ calls reset())
    self._tp = torch.zeros(k, dtype=torch.float64)
    self._fp = torch.zeros(k, dtype=torch.float64)
    self._fn = torch.zeros(k, dtype=torch.float64)
```

Caught by `test_evaluate_twice_with_same_metric_objects_gives_fresh_results`.

### B6: threshold tuned on the test set

The decision threshold is a hyperparameter. Choosing it to maximize F1 on the test set and then
reporting test F1 at that threshold is an optimistically biased estimate, because the test set is
no longer held out. The bias is large on small or imbalanced test sets. `val_loader` is accepted
but only used to report `val_auroc`, which makes the PR look correct at a glance. The fix tunes on
val scores and only *applies* the threshold to test.

```python
threshold, _ = select_threshold(test_scores, test_labels)  # buggy
threshold, _ = select_threshold(val_scores, val_labels)  # fixed
```

Caught by `test_evaluate_binary_tunes_threshold_on_validation_split`.

### B7: early stopping is off by one

Keras and Lightning define patience as "number of epochs with no improvement after which training
is stopped". After `patience` bad epochs the counter equals `patience`, so the check is `>=`. With
`>` the run trains one extra epoch every time, and `patience=1` behaves like `patience=2`.

```python
self.should_stop = self.num_bad_epochs > self.patience  # buggy
self.should_stop = self.num_bad_epochs >= self.patience  # fixed
```

Caught by all three `test_early_stopping_*` tests.

### B8: `evaluate()` neither disables autograd nor restores train mode

There are two defects in the same 5 lines.
1. With no `torch.no_grad()` (or `inference_mode()`), every forward pass builds an autograd graph.
   `BinaryAUROC` stores the outputs without detaching them (`pr.py:118`), so every batch's graph
   stays alive until `compute()`: memory grows with the **whole eval set** (OOM on large eval
   sets). The other metrics (`MeanLoss` calls `float(...)`) let each batch's graph be freed, but
   every batch still pays for building it: extra compute and peak memory.
2. `model.eval()` is never undone. The usual loop (`train epoch -> evaluate -> train epoch`) then
   trains every epoch after the first with dropout off and BatchNorm frozen on running stats. No
   error is raised, and the model quietly trains worse. Restoring only the top-level flag with
   `model.train(was_training)` is not enough either: it cascades to every submodule, so a
   submodule the caller froze in eval (e.g. BatchNorm during fine-tuning) gets flipped back to
   train. fixed.py snapshots `m.training` for every module. The fix saves `model.training` and restores
   it in a `finally`.

```python
model.eval()  # buggy
for inputs, target in loader:
    outputs = model(inputs.to(device))
...
was_training = model.training  # fixed
model.eval()
try:
    with torch.no_grad():
        for inputs, target in loader:
            ...
finally:
    model.train(was_training)
```

Caught by `test_evaluate_does_not_build_autograd_graph` (no-grad part) and
`test_evaluate_restores_training_mode` (mode part).

## Production improvements

1. **Shape validation at the boundary** *(applied in fixed.py: `_check_logits`, AUROC numel check,
   `select_threshold` shape check).* With it, B1-class mistakes raise a clear error instead of
   failing to broadcast, or worse, broadcasting silently.
   ```python
   if preds.ndim != 2 or target.ndim != 1 or preds.shape[0] != target.shape[0]:
       raise ValueError(f"expected logits (N, C) and target (N,), got {preds.shape} / {target.shape}")
   ```
2. **Single source of state: `__init__` calls `reset()`** *(applied).* This removes the B5 class of
   bugs. A stronger version, the torchmetrics pattern, registers states declaratively and has the
   base class handle reset, sync, and device moves:
   ```python
   self.add_state("tp", default=torch.zeros(k), reduction="sum")
   ```
3. **No host sync per batch** *(applied for `MeanLoss`).* `float(loss)` / `.item()` inside
   `update` forces a CUDA/MPS synchronization every batch. Accumulate tensors on the device and
   call `.item()` once in `compute()`. Also `detach()` anything a metric stores *(applied)*.
4. **Fail loudly on undefined values** *(partly applied).* `EarlyStopping.step` raises on a missing
   `monitor` key (instead of a bare `KeyError` deep in training) and on NaN. `MeanLoss.compute()` on
   an empty run returns NaN, not a fake `0.0`. Consider `logging.warning` when AUROC is undefined
   (single-class eval set).
5. **Vectorize `select_threshold`** *(not applied).* It is currently O(N * U). Sort once and use
   cumulative TP/FP counts, taking only the last index of each tie group as a candidate cut. That is
   O(N log N) and matters for million-row eval sets:
   ```python
   order = scores.argsort(descending=True)
   s, y = scores[order], labels[order].float()
   tp = y.cumsum(0)
   fp = (1 - y).cumsum(0)
   last = torch.ones_like(s, dtype=torch.bool)
   last[:-1] = s[1:] != s[:-1]
   f1 = 2 * tp / (tp + fp + y.sum())  # 2TP / (2TP + FP + FN), FN = P - TP
   best = f1[last].argmax()
   ```
6. **Distributed / multi-process correctness** *(not applied).* For DDP the sufficient statistics
   (tp/fp/fn counts, loss sum and count) must be `all_reduce`d, and AUROC must `all_gather` its
   scores. Keeping metrics as sums and counts (as fixed.py does) is what makes that possible. A mean
   of per-rank means repeats B2 at the rank level.
7. **Typed config + checkpoint-aware early stopping** *(not applied).* A
   `@dataclass(frozen=True) EarlyStoppingConfig` that is logged with the run, plus storing
   `best_epoch` / `best_state_dict` so training can roll back to the best weights instead of the
   last ones.
8. **Property-based tests** *(not applied).* Use permutation invariance (shuffling or re-batching
   must not change any metric) and AUROC(-scores) = 1 - AUROC(scores). These two properties catch
   B2, B3 and most streaming bugs without any hand-computed values.

## Interview talking points

- **Streaming metrics must accumulate sufficient statistics, not per-batch results.** Use counts,
  sums, and confusion-matrix cells, then compute once at the end. A mean of means, a mean of
  per-batch F1, or a mean of per-rank values are all wrong as soon as batch sizes differ, and the
  last batch nearly always does (B2, and the same reasoning for DDP).
- **Define the edge cases explicitly.** F1 with zero denominators (B4) and AUROC with ties (B3) are
  where hand-rolled metrics disagree with sklearn. Know the conventions: `zero_division=0`, and a
  tie counts as half (the average-rank Mann-Whitney U).
- **Any choice made by looking at data is training.** Thresholds, temperature, and early-stopping
  epochs are hyperparameters. Tuning them on the test split (B6) leaks the test set and inflates
  the reported number. Monitor validation metrics, not training loss or test metrics.
- **Eval must not have side effects on the model.** Use `no_grad` / `inference_mode` for memory and
  speed, and restore the caller's `train()`/`eval()` mode. Leaving a model in eval mode silently
  disables dropout and BatchNorm updates for the rest of training (B8).
- **Test with cases that expose the bugs.** Use N == C batches, a ragged last batch, tied scores, a
  never-predicted class, and two evaluations in a row. The "nice" tests (equal batches, distinct
  scores) pass on this buggy PR.
