# r07 — Mixed precision training + "numerically safer" losses: answer key

## Bugs

| ID | Severity | Location | What's wrong | Impact |
|----|----------|----------|--------------|--------|
| B1 | High | pr.py:77-78 | `clip_grad_norm_` runs before `scaler.unscale_`, so it clips the *scaled* gradients | silently wrong (clipping ~always fires, real update shrunk by the loss scale; logged grad norm is ×scale) |
| B2 | Medium | pr.py:72 | `torch.autocast(device_type="cuda")` hard-coded | silently wrong / perf (autocast is disabled with a warning on CPU/MPS: no mixed precision at all) |
| B3 | High | pr.py:103-104 | `F.cross_entropy` is fed softmax probabilities instead of logits | silently wrong (double softmax: loss and gradients squashed, training stalls) |
| B4 | High | pr.py:117-118 | `sigmoid` applied before `binary_cross_entropy_with_logits` | silently wrong (double sigmoid: loss floor ~0.31 for positives / ~0.69 for negatives, gradients vanish) |
| B5 | High | pr.py:132 | `gather(-1, tokens)` with a `(B, T)` index on a `(B, T, V)` tensor | crash (`RuntimeError: Index tensor must have the same number of dimensions`) |
| B6 | Medium | pr.py:131 | `torch.log(torch.softmax(x))` instead of `log_softmax` | silently wrong (softmax underflows to 0 -> `-inf` log-probs, NaN gradients for large logits) |
| B7 | Medium | pr.py:148 | `x / (norm + eps)` computed in the input dtype; `1e-8` rounds to 0 in fp16 | silently wrong (zero vectors become NaN in fp16; NaN poisons the whole batch) |
| B8 | Medium | pr.py:142-143 | `masked_mean` reduces in the input dtype | silently wrong (fp16 sum overflows past 65504 -> `inf` loss / metric) |

### B1: Gradient clipping on scaled gradients

GradScaler multiplies the loss by `S` (e.g. 2^16) before `backward()`, so every `.grad` is `S×`
too large until `scaler.unscale_(optimizer)` divides it back. Clipping before unscaling compares
`S·||g||` against `grad_clip`: clipping fires on essentially every step, and after the (late)
`unscale_` divides by `S`, the applied gradient norm is `grad_clip / S`, so learning
effectively stops. The reported
`grad_norm` is also `S×` too large. The documented AMP recipe is
`scale(loss).backward() -> unscale_ -> clip -> step -> update`. (`unscale_` is also where the
scaler records inf/NaN so `step()` can skip; calling it explicitly is what makes clipping legal.)

```python
grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
scaler.unscale_(optimizer)
```
```python
scaler.unscale_(optimizer)
grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```
Caught by: `test_train_step_reports_unscaled_gradient_norm`, `test_train_step_clipped_update_has_norm_grad_clip`.

### B2: autocast hard-coded to CUDA

`torch.autocast("cuda")` on a machine without CUDA emits `UserWarning: CUDA is not available...
Disabling autocast` and runs everything in fp32. The PR's whole point (speed / memory) silently
disappears on MPS/CPU, and the "matches fp32 baseline" claim in the PR is then trivially true.
Derive the device type from the model.

```python
with torch.autocast(device_type="cuda", dtype=dtype):
```
```python
device_type = next(model.parameters()).device.type
with torch.autocast(device_type=device_type, dtype=dtype):
```
Caught by: `test_train_step_forward_runs_in_requested_low_precision[float16|bfloat16]`.

### B3: cross-entropy on probabilities

`F.cross_entropy` = `log_softmax` + NLL. Passing probabilities applies softmax twice; inputs in
`[0, 1]` produce a nearly uniform second softmax, so the loss is stuck near `log(C)` and
gradients are tiny. Classic "loss doesn't go down" bug that never crashes.

```python
probs = torch.softmax(logits.float(), dim=-1)
return F.cross_entropy(probs, targets, label_smoothing=smoothing)
```
```python
return F.cross_entropy(logits.float(), targets, label_smoothing=smoothing)
```
Caught by: `test_smoothed_cross_entropy_matches_reference[*]`, `test_smoothed_cross_entropy_is_finite_and_correct_for_extreme_logits`.

### B4: double sigmoid in BCE

`binary_cross_entropy_with_logits` applies the sigmoid internally (via the stable
`max(x,0) - x·y + log1p(exp(-|x|))` form). Feeding it `sigmoid(x)` ∈ (0,1) means the effective
probability is `sigmoid(sigmoid(x))` ∈ (0.5, 0.73): a perfectly confident correct prediction still
has loss ≈ 0.31 (positive label) or ≈ 0.69 (negative label), and gradients are scaled by `sigmoid'(x)`, which vanishes for large |x|.

```python
probs = torch.sigmoid(logits.float())
return F.binary_cross_entropy_with_logits(probs, targets.float(), pos_weight=pos_weight)
```
```python
return F.binary_cross_entropy_with_logits(logits.float(), targets.float(), pos_weight=pos_weight)
```
Caught by: `test_multilabel_bce_matches_reference[*]`, `test_multilabel_bce_confident_correct_predictions_have_near_zero_loss`.

### B5: gather index has the wrong rank

`torch.gather(input, dim, index)` requires `index.ndim == input.ndim`; the output has the
index's shape. Tokens must be `(B, T, 1)` and the trailing dim squeezed afterwards. (The
`.squeeze(-1)` already in the PR is the tell that the author meant to unsqueeze.)

```python
return log_probs.gather(-1, tokens).squeeze(-1)
```
```python
return log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
```
Caught by: `test_sequence_log_probs_shape_and_values`, `test_sequence_log_probs_finite_for_unlikely_tokens_under_extreme_logits`.

### B6: log(softmax) instead of log_softmax

`softmax` of a logit 1e4 below the max is `exp(-1e4)` = 0 in any float dtype, so `log` gives
`-inf` (and the backward of `log` at 0 is `inf·0 = NaN`). `log_softmax` computes
`x - max - log(sum(exp(x - max)))` directly and returns the finite value `-1e4`.

```python
log_probs = torch.log(torch.softmax(logits.float(), dim=-1))
```
```python
log_probs = F.log_softmax(logits.float(), dim=-1)
```
Caught by: `test_sequence_log_probs_finite_for_unlikely_tokens_under_extreme_logits`.

### B7: eps underflows in fp16

The smallest positive fp16 subnormal is ~6e-8, so `fp16_tensor + 1e-8` adds exactly 0. A zero
vector gives `0 / 0 = NaN`. Under autocast, embeddings arrive as fp16, so a single padding
row turns the batch's similarity matrix (and loss) into NaN. Fix: compute the norm in float32
and `clamp_min(eps)`, then cast back. The fp32 upcast is the part that matters: `clamp_min`
alone isn't enough, and `F.normalize` on fp16 input still returns NaN for a zero vector
(its eps rounds to 0 in fp16 too).

```python
return x / (x.norm(dim=dim, keepdim=True) + eps)
```
```python
norm = x.float().norm(dim=dim, keepdim=True).clamp_min(eps)
return (x.float() / norm).to(x.dtype)
```
Caught by: `test_l2_normalize_zero_vector_stays_zero_in_float16`, `test_cosine_similarity_with_zero_vector_is_zero_in_float16`.

### B8: fp16 reduction overflow

fp16 max is 65504. 8192 tokens with loss 10 sum to 81920 -> `inf`, and `inf / 8192 = inf`.
This is why autocast itself runs `sum`, `softmax`, `log_softmax`, losses etc. in float32:
reductions should accumulate in float32 even when the elementwise data is fp16.

```python
mask = mask.to(values.dtype)
return (values * mask).sum() / mask.sum()
```
```python
weights = mask.to(torch.float32)
total = (values.float() * weights).sum()
return total / weights.sum().clamp_min(1.0)
```
Caught by: `test_masked_mean_of_many_fp16_values_does_not_overflow`.

## Production improvements

1. **Validate hyper-parameters early** — *applied in fixed.py* (`grad_clip <= 0`,
   `temperature <= 0` raise `ValueError`). A temperature of 0 otherwise yields NaN distillation
   targets far from the call site.
   ```python
   if temperature <= 0:
       raise ValueError(f"temperature must be positive, got {temperature}")
   ```
2. **Empty-mask guard in `masked_mean`** — *applied* (`clamp_min(1.0)` on the count): a batch
   that is all padding returns 0 instead of NaN.
3. **Make the scaler optional for bf16** — bf16 has fp32's
   exponent range, so loss scaling is unnecessary; accept `scaler=None` or
   `GradScaler(enabled=(dtype == torch.float16))`:
   ```python
   scaler = torch.amp.GradScaler(device_type, enabled=dtype is torch.float16)
   ```
4. **Avoid per-step host syncs** — `loss.item()` / `grad_norm.item()` / `get_scale()` each force a
   device sync. Return detached tensors and `.item()` only every `log_every` steps:
   ```python
   return StepStats(loss=loss.detach(), grad_norm=grad_norm.detach(), ...)  # tensor fields
   ```
5. **Use the scaler's own skip signal** — comparing `get_scale()` before/after works but is
   indirect; also track a counter of skipped steps and alert when it exceeds a few % (persistent
   overflow means a real NaN bug, not a scale problem):
   ```python
   if stats.skipped:
       metrics["amp/skipped_steps"] += 1
       logger.warning("AMP skipped step %d (scale=%.0f)", step, stats.scale)
   ```
6. **Gradient accumulation support** — expose `accum_steps`, divide the loss, and only
   `unscale_/clip/step/update/zero_grad` on the boundary step; wrap non-boundary micro-batches in
   `model.no_sync()` under DDP.
7. **Checkpoint the scaler** — `scaler.state_dict()` must be saved/restored with the optimizer,
   otherwise resumed fp16 runs restart at `init_scale` and skip a burst of steps.
8. **Config object** — group `grad_clip`, `dtype`, `accum_steps` in a frozen `@dataclass AmpConfig`
   so the trainer, logging and checkpoint metadata share one source of truth.

## Interview talking points

- **AMP step order:** `scale(loss).backward() -> unscale_(opt) -> clip -> step(opt) -> update()`.
  Anything that reads gradient *values* (clipping, norm logging, custom grad ops) must come after
  `unscale_`; `step()` skips the update if `unscale_` found inf/NaN, and `update()` then lowers
  the scale.
- **Fused "with logits" APIs exist for stability:** `cross_entropy`, `log_softmax`,
  `binary_cross_entropy_with_logits` all use log-sum-exp / softplus identities. Never compose
  `log(softmax)` or `sigmoid -> BCEWithLogits` / `softmax -> cross_entropy`; the latter two don't
  crash, they just quietly ruin training.
- **fp16 vs bf16:** fp16 has 10 mantissa bits and max 65504 (overflow/underflow risk; needs
  loss scaling); bf16 has fp32's 8-bit exponent but only 7 mantissa bits (no scaling needed,
  less precision). Constants like `eps=1e-8` and reductions must be done in fp32.
- **Why autocast keeps some ops in fp32:** reductions and softmax/loss ops are on autocast's
  fp32 list for exactly the overflow reason in B8 — helpers that run *outside* autocast (or on
  pre-cast fp16 tensors) have to upcast themselves.
- **Device-agnostic code:** derive the device from the model/tensors; hard-coded `"cuda"` in
  autocast silently turns into fp32 on Apple/CPU, which a "loss curves match" test won't notice.
