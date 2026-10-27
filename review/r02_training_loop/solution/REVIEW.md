# r02_training_loop — answer key

PR: "Add train/validate loop with gradient accumulation, clipping and LR schedule"

## Bugs

| ID | Severity | Location | What's wrong | Impact |
|----|----------|----------|--------------|--------|
| B1 | Critical | pr.py:82 (and 89-92) | `optimizer.zero_grad()` is called once per epoch, never after an optimizer step | silently wrong (every step applies the sum of all gradients seen so far this epoch) |
| B2 | High | pr.py:87 | Loss is not divided by `accumulation_steps` before `backward()` (the docstring says it is) | silently wrong (effective LR is k times too large; results change with k) |
| B3 | High | pr.py:90-92 | `clip_grad_norm_` runs after `optimizer.step()` | silently wrong (clipping has no effect on the update) |
| B4 | Medium | pr.py:94-95 | `scheduler.step()` runs once per micro-batch, not once per optimizer step as documented | silently wrong (schedule runs k times too fast; warmup/decay ends early) |
| B5 | High | pr.py:98 | `running_loss += loss * batch_size` accumulates the live loss tensor, not a detached value | perf / memory leak (every batch's autograd graph stays alive until the epoch ends) |
| B6 | High | pr.py:105-113 | `evaluate` never calls `model.eval()` | silently wrong (dropout is on during validation, BatchNorm running stats are updated by validation data) |
| B7 | Medium | pr.py:110, 114, 116 | Validation loss is the mean of batch means, divided by the number of batches | silently wrong (biased when the last batch is smaller; not the dataset mean the PR promises) |
| B8 | High | pr.py:89 (nothing after the loop at 101) | A trailing partial accumulation window is never stepped | silently wrong (last `len % k` batches never influence the weights; their gradients are wiped by the next epoch's `zero_grad`) |

### B1: gradients are never cleared between optimizer steps

PyTorch's `.backward()` **adds** into `.grad`. Gradient accumulation relies on that: you sum
`k` micro-batches and then step. But once you step, you must clear the gradients, or the next
window starts from the old sum. Here `zero_grad` runs only once at the start of the epoch, so
step `n` applies the sum of all gradients since the start of the epoch. With momentum this gets
worse fast. It also fails for `accumulation_steps=1`, the default.

```python
optimizer.zero_grad(set_to_none=True)
for step, (inputs, targets) in enumerate(loader):
    ...
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
```

```python
def optimizer_step() -> None:
    ...
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)  # FIX(B1)
```

Caught by: `test_train_epoch_matches_reference_loop[1|2|3]`,
`test_gradient_clipping_matches_reference_and_bounds_update`,
`test_train_with_scheduler_and_accumulation_matches_reference`, `test_fit_matches_reference_train_then_validate`.

### B2: loss not scaled by 1/k under accumulation

With `reduction="mean"`, each micro-batch loss is already a mean over its samples. Summing the
gradients of `k` of them gives `k` times the gradient of the mean over the window. You must
divide each loss by `k`, so that `k` micro-batches of size `b` give the same step as one batch
of size `k*b`. Without the division the effective learning rate grows with `k`, and the LR you
tuned stops being valid when you change accumulation.

```python
loss = loss_fn(model(inputs), targets)
loss.backward()
```

```python
loss = loss_fn(model(inputs), targets)
(loss / accumulation_steps).backward()  # FIX(B2)
```

Keep the unscaled `loss` for logging. Only the tensor you backprop gets scaled.

Caught by: `test_accumulated_step_equals_one_big_batch_step` (the direct invariant),
`test_train_epoch_matches_reference_loop[2|3]`, `test_train_with_scheduler_and_accumulation_matches_reference`,
`test_fit_matches_reference_train_then_validate`.

### B3: gradient clipping after the optimizer step

`clip_grad_norm_` rescales `.grad` in place. The order has to be backward -> clip -> step. If
you clip after `step()`, you clip gradients that were already applied (and that get zeroed
next), so the clip does nothing. Spikes go straight through. Under AMP the order is
`scaler.unscale_(opt)` -> clip -> `scaler.step(opt)`.

```python
optimizer.step()
if max_grad_norm is not None:
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

```python
if max_grad_norm is not None:
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # FIX(B3)
optimizer.step()
```

Caught by: `test_gradient_clipping_matches_reference_and_bounds_update` (with plain SGD,
`||Δθ|| <= lr * max_norm` must hold), `test_fit_matches_reference_train_then_validate`.

### B4: per-iteration scheduler stepped per micro-batch

The PR says the scheduler is per-iteration (warmup, cosine, OneCycle), so it advances once per
**optimizer step**. Here `scheduler.step()` sits outside the `if`, so it runs on every
micro-batch. With `k=4` a 10k-step cosine schedule is done after 2.5k optimizer steps, and
OneCycle raises once it goes past `total_steps`. PyTorch also warns "Detected call of
`lr_scheduler.step()` before `optimizer.step()`" on the first batch. That warning is a hint.
(An epoch-level schedule such as `StepLR(step_size=epochs)` would be stepped in `fit`, once
per epoch. Pick one convention, write it down, and enforce it.)

```python
if (step + 1) % accumulation_steps == 0:
    ...
if scheduler is not None:
    scheduler.step()
```

```python
def optimizer_step() -> None:
    ...
    if scheduler is not None:
        scheduler.step()  # FIX(B4)
```

Caught by: `test_scheduler_advances_once_per_optimizer_step` (`last_epoch == 2` after 4
micro-batches with `k=2`), `test_train_with_scheduler_and_accumulation_matches_reference`,
`test_fit_matches_reference_train_then_validate`.

### B5: accumulating the live loss tensor keeps every graph alive

`loss` has a `grad_fn`. `running_loss += loss * batch_size` records an `AddBackward` node that
links `running_loss` to that batch's graph. After N batches, `running_loss` keeps N graphs'
worth of autograd nodes alive, and anything the nodes hold (custom-Function `ctx` state, and
saved activations when `backward` is skipped or uses `retain_graph`). Memory grows over the
epoch until the function returns. Keeping the accumulator on the device to avoid a per-step
`.item()` sync is a good idea. It only needs `.detach()`. The classic CPU-side variant of this
bug is `total += loss` without `.item()`.

```python
running_loss += loss * batch_size
```

```python
running_loss += loss.detach() * batch_size  # FIX(B5)
```

Caught by: `test_epoch_does_not_keep_autograd_graphs_of_earlier_batches_alive`. A custom
autograd Function stores a token in `ctx`, and the test counts how many earlier batches'
tokens are still reachable mid-epoch. It must be at most 1 (the previous `loss` local).

### B6: validation runs in training mode

`torch.no_grad()` and `model.eval()` do different jobs. `no_grad` turns off graph
recording (memory and speed). `eval()` switches module behaviour: Dropout becomes the
identity, and BatchNorm uses its running statistics instead of batch statistics **and stops
updating them**. Without `eval()`, the validation loss is noisy (random dropout masks), and
validation data leaks into the BatchNorm buffers that are saved with the checkpoint.

```python
@torch.no_grad()
def evaluate(model, loader, loss_fn):
    device = _model_device(model)
    ...
```

```python
was_training = model.training
model.eval()  # FIX(B6)
try:
    ...
finally:
    model.train(was_training)
```

Caught by: `test_evaluate_disables_dropout`, `test_evaluate_does_not_update_batchnorm_statistics`,
`test_fit_matches_reference_train_then_validate`.

### B7: mean of batch means is not the dataset mean

With 10 samples and `batch_size=4` the batches are 4, 4, 2. Averaging three batch means gives
the 2 samples of the last batch twice the weight of the others. The PR promises "sample-weighted
mean loss over the dataset", which needs `sum(mean_i * n_i) / sum(n_i)`. (Or use
`reduction="sum"` and divide by N. Or `drop_last=True`, but never for validation.) The error
shrinks as N grows, but it is real for small validation sets and makes runs with different
batch sizes impossible to compare.

```python
total_loss += loss_fn(model(inputs), targets).item()
num_batches += 1
...
return total_loss / num_batches
```

```python
batch_size = inputs.size(0)
total_loss += loss_fn(model(inputs), targets) * batch_size  # FIX(B7)
num_samples += batch_size
...
return total_loss.item() / num_samples
```

Caught by: `test_evaluate_is_sample_weighted_mean_with_ragged_last_batch`,
`test_fit_matches_reference_train_then_validate`.

### B8: the final partial accumulation window is dropped

`if (step + 1) % k == 0` only fires on full windows. With 5 batches and `k=2`, batch 5's
gradient is computed and then never applied: the next epoch's `zero_grad` (which runs at the top
of `train_one_epoch`) wipes it, so the work is wasted and those samples never influence the
weights. Fix: track pending
micro-batches and step once more after the loop. (Alternatives: `drop_last` on the sampler,
or compute `is_last = step + 1 == len(loader)` when the loader is `Sized`.)

Convention note: the tests follow the Hugging Face convention and divide the tail window's loss
by `k` too. Dividing by the real window size is also defensible, but it will fail the
reference-equivalence tests.

```python
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
...
return running_loss.item() / num_samples
```

```python
if pending:
    optimizer_step()  # FIX(B8)
```

Caught by: `test_every_micro_batch_contributes_to_an_optimizer_step[5-2-3|5-3-2]` (a counting
`SGD` subclass), `test_train_epoch_matches_reference_loop[2|3]`,
`test_train_with_scheduler_and_accumulation_matches_reference`, `test_fit_matches_reference_train_then_validate`.

## Production improvements

1. **Validate configuration at construction** (applied in fixed.py). Use `__post_init__` on
   `TrainConfig`, plus a guard in `train_one_epoch` for direct callers. `accumulation_steps=0`
   otherwise fails with `ZeroDivisionError` deep in the loop.
   ```python
   def __post_init__(self) -> None:
       if self.accumulation_steps < 1:
           raise ValueError(f"accumulation_steps must be >= 1, got {self.accumulation_steps}")
   ```
2. **Explicit empty-loader errors** (applied). Raise `ValueError("... yielded no batches")`
   instead of a bare `ZeroDivisionError`, which looks like a math bug.
3. **Restore train/eval mode in `evaluate`** (applied). Use `was_training` with `try/finally`
   so `evaluate` can be called from anywhere (callbacks, early stopping) without leaving the
   model in the wrong mode.
4. **Async-friendly host/device transfers** (applied). Keep the loss accumulators on the device
   and call `.item()` once per epoch. Use `.to(device, non_blocking=True)` together with
   `DataLoader(pin_memory=True)` on CUDA.
5. **Structured logging instead of silence/print** (applied). `logger.info` per epoch, using
   lazy `%`-formatting.
6. **Mixed precision + skipping the gradient sync during accumulation** (not applied). For
   CUDA/MPS, `torch.autocast(device.type, dtype=torch.bfloat16)`. With CUDA fp16, use a
   `GradScaler` and the order `scaler.scale(loss/k).backward()`, `scaler.unscale_(opt)`, clip,
   `scaler.step(opt)`, `scaler.update()`. Under DDP, wrap the non-final micro-batches in
   `model.no_sync()` so the all-reduce runs once per window instead of k times.
   ```python
   ctx = model.no_sync() if is_ddp and not window_end else contextlib.nullcontext()
   with ctx, torch.autocast(device.type, dtype=torch.bfloat16):
       loss = loss_fn(model(x), y)
       (loss / k).backward()
   ```
7. **Observability hooks** (not applied). Return or log the pre-clip gradient norm
   (`clip_grad_norm_` returns it), the LR per step, and a `torch.isfinite(loss)` check that
   aborts or skips the step on NaN/inf. Those signals are what you look at when a run
   diverges.
   ```python
   grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
   if not torch.isfinite(grad_norm):
       optimizer.zero_grad(set_to_none=True)
       return  # skip the bad step
   ```
8. **Exact last-window scaling** (not applied, it is a semantics choice). The fix divides the
   trailing partial window by `k` (the Hugging Face convention), so its step is slightly smaller.
   If that matters, divide by the actual window size `min(k, n_batches - window_start)`,
   which needs a `Sized` loader. Document whichever you pick.

## Interview talking points

- **The canonical step order:** `forward -> (loss / k).backward()` for each micro-batch, then
  once per window `clip -> optimizer.step() -> scheduler.step() -> zero_grad()`. Also cover the
  tail window. Know why each position matters: grads accumulate by design, clipping edits
  `.grad`, and the scheduler counts optimizer steps.
- **`eval()` vs `no_grad()`**: they are orthogonal. `no_grad` is about autograd (memory, speed).
  `eval` is about module semantics (Dropout, BatchNorm statistics). Validation needs both,
  and `torch.inference_mode()` is a stricter, faster `no_grad`.
- **Metric hygiene**: `.item()`/`.detach()` before accumulating, because otherwise you keep
  graphs alive. Weight by sample count, because a mean of batch means is biased with a ragged
  last batch. Keep accumulators on the device to avoid one sync per step.
- **How to test a training loop**: compare against a hand-written reference on tiny float64 CPU
  data. Check invariants (k micro-batches of size b == one batch of size k*b, `||Δθ|| <= lr*max_norm`
  for SGD), count optimizer steps with a subclass, read `scheduler.last_epoch`, and probe module
  modes with Dropout/BatchNorm.
- **Silent bugs are the dangerous ones**: every bug here still "trains" and the loss
  still goes down, which is why the PR's manual MNIST check didn't catch them. Equivalence
  tests against a reference do.
