# r06_data_pipeline: answer key

## Bugs

| ID | Severity | Location | What's wrong | Impact |
|----|----------|----------|--------------|--------|
| B1 | High | pr.py:135 | Normalizer is fit on **all** records (val/test included), not on train | silently wrong (leakage, optimistic eval) |
| B2 | Critical | pr.py:132-134 | Oversampling runs **before** the split, so copies of one record land in train and val/test | silently wrong (leakage, inflated metrics, eval class balance distorted) |
| B3 | Critical | pr.py:98-102 | `group_key` only sorts records by group. The cut is still by record count, so groups straddle the split boundaries. Groups are also assigned in key order, not at random | silently wrong (user leakage, biased splits) |
| B4 | Medium | pr.py:121 | `std` is not guarded: a constant feature divides by 0 and gives `nan` | silently wrong (nan features, nan loss) |
| B5 | High | pr.py:178 (with pr.py:42) | Mask is `padded != PAD_ID`, but `PAD_ID == 0` is also a real token id | silently wrong (real tokens dropped) |
| B6 | High | pr.py:224 | `(per_step * mask).mean()` divides by `B*T`, padding included | silently wrong (loss scale depends on padding) |
| B7 | High (production / reproducibility) | pr.py:183-187 (with pr.py:154, 163) | `worker_init_fn` seeds the *global* NumPy RNG, but the dataset draws from its own `self.rng`, which is copied identically into every worker | silently wrong (augmentation repeats across workers) |
| B8 | Medium | pr.py:201, 204 | The val loader inherits `augment=True` and `shuffle=True` from the train config | silently wrong (noisy, non-reproducible val metrics) |

### B1: Normalization stats leak from val/test

z-scoring is a *learned* transform. Its parameters are part of the model and may only come from
training data. When mean and std are fit on the full dataset, val/test information goes into the
inputs, and the reported generalization is optimistic. The usual rule is "fit on train,
`transform` everywhere", the same as sklearn's `fit` vs `transform`.

```python
train, val, test = split_dataset(records, ...)
normalizer = fit_normalizer(records)  # buggy
```
```python
train, val, test = split_dataset(records, ...)
normalizer = fit_normalizer(train)  # fixed
```
Caught by `test_normalizer_statistics_come_from_training_records_only`.

### B2: Oversampling before splitting

Duplicating minority records before the split means a record and its copy can land on both sides.
The model is then evaluated on examples it trained on. The minority class is also inflated in
val/test, so the metrics no longer reflect the real class distribution. Resampling and
augmentation are training-time operations, so you **split first** and then oversample the train
split only.

```python
if oversample:
    records = oversample_minority(records, seed=seed)
train, val, test = split_dataset(records, ...)
```
```python
train, val, test = split_dataset(records, ...)
normalizer = fit_normalizer(train)
if oversample:
    train = oversample_minority(train, seed=seed)
```
Caught by `test_no_record_appears_in_both_train_and_eval_splits` and
`test_oversampling_balances_the_training_split`.

### B3: Group-aware split does not group

When the same user appears many times, a record-level split lets the model memorize users. Val
then measures "seen users", not new ones. Sorting makes each group contiguous, but the cut points
are still record indices, so the groups at the boundaries are split across two sets. Sorting by
key also makes the assignment deterministic: the lowest user ids always go to train, which biases
the splits if ids correlate with signup time or region. The fix is to shuffle the *group keys* and
assign whole groups (the same idea as sklearn's `GroupShuffleSplit`).

```python
order = list(records)
random.Random(seed).shuffle(order)
if group_key is not None:
    order.sort(key=lambda r: getattr(r, group_key))
n_train = int(ratios[0] * len(order))
...
```
```python
groups = defaultdict(list)
for i, r in enumerate(records):
    groups[getattr(r, group_key) if group_key is not None else i].append(r)
keys = list(groups)
rng.shuffle(keys)
for key in keys:  # whole groups only
    part = 0 if seen < targets[0] else 1 if seen < targets[1] else 2
    splits[part].extend(groups[key])
    seen += len(groups[key])
```
Caught by `test_group_split_puts_each_user_in_exactly_one_split[*]` and
`test_group_split_assigns_users_to_splits_at_random`.

### B4: Division by zero std

A feature that is constant in the training data (a one-hot column that never fires, or a
sensor stuck at one value) has `std == 0`. `(x - mean) / 0` gives `nan` (`0/0`) or `inf`, and a
single `nan` input makes the whole loss `nan`. Guard it: leave constant features unscaled, or
add an eps.

```python
return Normalizer(mean=x.mean(axis=0), std=x.std(axis=0))
```
```python
std = x.std(axis=0)
constant = std < STD_EPS
return Normalizer(mean=x.mean(axis=0), std=np.where(constant, 1.0, std))
```
Caught by `test_constant_feature_normalizes_to_finite_values`.

### B5: Padding id collides with a real token

The mask is computed from token *values*, but id 0 is a valid vocabulary entry
(`token ids are in [0, vocab_size)`). Every genuine token 0 is therefore treated as padding. The
model never attends to or gets loss from those positions. Nothing crashes, so this can go
unnoticed for a long time. Derive the mask from `lengths`, which is the ground truth, or reserve a
dedicated pad id outside the vocabulary.

```python
return Batch(..., mask=padded != PAD_ID, ...)
```
```python
lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
mask = torch.arange(padded.shape[1])[None, :] < lengths[:, None]
return Batch(..., lengths=lengths, mask=mask, ...)
```
Caught by `test_mask_marks_exactly_the_real_tokens_even_for_token_id_zero`.

### B6: Masked loss averages over padded positions

Zeroing the pad terms is only half the job: `.mean()` still divides by `B*T`. Padded positions
count in the denominator. The effective loss scale then depends on how much padding the batch
has, which in turn depends on the longest sequence in the batch. A short sequence batched with a
long one gets down-weighted. That interacts badly with bucketing, gradient accumulation and LR
tuning. Divide by the number of real positions.

```python
return (per_step * mask).mean()
```
```python
mask = mask.to(per_step.dtype)
return (per_step * mask).sum() / mask.sum().clamp_min(1.0)
```
Caught by `test_loss_of_a_sample_does_not_depend_on_padding` and
`test_loss_matches_mean_cross_entropy_over_real_positions`.

### B7: Every worker replays the same augmentation noise

Each DataLoader worker gets a *copy* of the dataset object (forked memory, or pickled under
spawn). `self.rng = np.random.default_rng(seed)` is copied with it, in the same state, so worker
k's i-th sample gets the same noise as worker 0's i-th sample. The `worker_init_fn` looks
correct, but it reseeds the *global* `np.random`, which `__getitem__` never uses. Since
PyTorch 1.9 the DataLoader itself seeds `torch`, Python `random` **and the global `np.random`**
in every worker, so those `seed` calls are redundant. What it cannot reach is an RNG object you
hold yourself, like `np.random.default_rng` on the dataset: that's your responsibility. Fix
by reseeding the dataset's own generator from `torch.initial_seed()`, which is `base_seed +
worker_id` and changes every epoch while staying reproducible through the loader's `generator`.

```python
def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
```
```python
def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    info = get_worker_info()
    if info is not None and isinstance(info.dataset, SequenceDataset):
        info.dataset.rng = np.random.default_rng(seed)
```
Caught by `test_each_worker_draws_different_augmentation_noise` (real DataLoader,
`num_workers=2`). The test uses the `fork` context because under `spawn` the workers would
re-import `pr.py` from disk and bypass the `--solution` redirect. The bug itself also happens
under spawn.

### B8: Validation loader shuffles and augments

The val/test pipeline must be deterministic. Otherwise the metric has noise from the augmentation
on top of model noise, two evaluations of the same checkpoint disagree, early stopping and model
selection chase that noise, and per-sample error analysis can't be lined up across runs.
Sharing one kwargs dict between the two loaders copied the train-only settings.

```python
val_ds = SequenceDataset(data.val, data.normalizer, augment, noise_std, seed)
common = {"batch_size": batch_size, "shuffle": True, ...}
return DataLoader(train_ds, **common), DataLoader(val_ds, **common)
```
```python
val_ds = SequenceDataset(data.val, data.normalizer, augment=False)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, ...)
```
Caught by `test_validation_loader_is_deterministic_and_unaugmented`.

## Production improvements

1. **Validate inputs** *(applied in fixed.py)*. Reject negative ratios and empty normalizer
   input instead of returning a degenerate split or a `nan` mean.
   ```python
   if not np.isclose(sum(ratios), 1.0) or min(ratios) < 0:
       raise ValueError(...)
   ```
2. **Logging** *(applied)*. Log split sizes and constant features, so a silent data problem shows
   up in the run log.
   ```python
   logger.info("split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))
   ```
3. **One generator per loader, persistent workers** *(applied)*. This one is a real (Low,
   reproducibility) defect in pr.py, not just a nicety: the PR shared one `torch.Generator`
   between both loaders, so iterating val changed train's shuffle order, which then depended on
   how often you validated.
   `persistent_workers=True` avoids respawning workers (and re-pickling the dataset) every epoch.
4. **Stratified split** *(not applied, API change)*. With a rare class, a random split can leave
   val with zero positives. Stratify on label inside each group bucket, e.g.
   `split_dataset(..., stratify_key="label")`, then assert every class appears in every split.
   ```python
   for label, bucket in by_label.items():
       tr, va, te = _split_groups(bucket, ratios, rng)
   ```
5. **Frozen config dataclass** *(not applied)*. Replace the long kwargs list with a `LoaderConfig`
   so train and eval settings are explicit and can't be mixed up again (the cause of B8).
   ```python
   @dataclass(frozen=True)
   class LoaderConfig:
       batch_size: int = 32
       num_workers: int = 4
       augment: bool = False
       shuffle: bool = False


   TRAIN = LoaderConfig(augment=True, shuffle=True)
   EVAL = LoaderConfig()
   ```
6. **Persist the normalizer with the checkpoint** *(not applied)*. Serving must use the *train*
   stats. Save `mean`/`std` (e.g. `np.savez` or as buffers on the model), not refit at inference.
7. **`pin_memory` + `prefetch_factor` for GPU training** *(not applied)*. Use
   `pin_memory=device.type == "cuda"` and `non_blocking=True` in `.to(device)`. Skip it on MPS, where
   pinning has no effect.
8. **Leakage assertions as a pipeline check** *(not applied)*. Cheap runtime guards such as
   `assert not train_users & val_users` in `prepare_datasets` catch regressions that unit
   tests on toy data might miss.

## Interview talking points

- **"Anything learned from data is fit on train only."** Normalizer stats, vocabularies,
  oversampling and augmentation all happen *after* the split (B1, B2). Leakage rarely crashes.
  It shows up as metrics that look too good and then don't hold in production.
- **Split by the unit you generalize over.** If the product predicts for *new users*, split by
  user. Check it with set intersections, not by eyeballing (B3). Mention `GroupShuffleSplit` /
  `StratifiedGroupKFold`.
- **Padding needs two things: the right mask and the right denominator.** Build masks from lengths
  and never from token values (B5). Normalize the loss by `mask.sum()`, not `numel()` (B6).
  Otherwise batch composition changes the gradient scale.
- **Worker RNG.** PyTorch (≥ 1.9) seeds `torch`, Python `random` and the global `np.random` in
  each worker. RNG objects held in the dataset (`np.random.default_rng`) are copied as-is, so
  seed them in `worker_init_fn` from `torch.initial_seed()` (B7). The famous "every worker
  returns the same augmentation" bug with global `np.random` only affects torch < 1.9.
- **Eval must be deterministic:** no shuffle, no augmentation, fixed order (B8). Otherwise you
  can't compare checkpoints or do per-example error analysis.
