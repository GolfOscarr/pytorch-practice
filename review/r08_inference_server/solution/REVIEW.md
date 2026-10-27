# r08 answer key: batched generation + async inference service

## Bugs

| ID | Severity | Location | What's wrong | Impact |
|----|----------|----------|--------------|--------|
| B1 | Critical | pr.py:207-208 | `pad_prompts` right-pads, so `logits[:, -1]` is read at a pad slot for shorter prompts, and new tokens are appended after the pads | silently wrong |
| B2 | High | pr.py:234, pr.py:252 | `position_ids` are `arange(prompt_len)` / `prompt_len + step` for every row, ignoring padding, so a padded row's positions don't match what it would get alone | silently wrong |
| B3 | High | pr.py:182-186 | `temperature=0` divides by zero *before* the greedy branch. Every positive logit becomes `+inf`, every negative one `-inf`, and a `0` logit becomes `nan`. `argmax` treats `nan` as the max, so it returns the first `nan` or the first `+inf` instead of the best token | silently wrong |
| B4 | Medium | pr.py:173 | top-p uses the inclusive cumsum with `>`: it drops the token that crosses the threshold and can drop the top token itself (then the row is all `-inf`, which gives `nan` probs and a crash in `multinomial`) | silently wrong / crash |
| B5 | Medium | pr.py:212 (`generate`) | no `torch.inference_mode()`/`no_grad` and no `model.eval()`: it builds an autograd graph for every step (memory grows with the KV cache), and dropout stays active if the model is in train mode | perf + silently wrong |
| B6 | Critical | pr.py:384 | the worker calls the blocking `_run_batch` directly inside the event loop. While a batch runs, no other coroutine runs: no new requests are accepted, timeouts don't fire, health checks stall | perf / availability |
| B7 | High | pr.py:367 | prompts are sorted by length but outputs are zipped back to the *unsorted* `batch`, so callers get someone else's completion | silently wrong + data leak between users |
| B8 | High | pr.py:385-387 | a failed batch is logged and dropped, and its futures are never resolved. Callers hang until their timeout (forever if `timeout=None`) and see `TimeoutError` instead of the real error | hang |
| B9 | High | pr.py:335-338, pr.py:368 | `_validate` checks `len(prompt) + max_new_tokens <= max_seq_len` per request, but the batch decodes `max(max_new_tokens)` steps for every row. A long-prompt/short-budget request batched with a short-prompt/long-budget one runs past the position table (`IndexError` in `pos_emb`), and the whole batch fails | crash: one user's request fails other users' requests |

### B1: right padding in batched decoding

A decoder-only model generates by appending to the end of the sequence. With right padding, a
short row looks like `[t1 t2 PAD PAD]`, so `logits[:, -1]` for that row is computed at a PAD
position, and generated tokens land after the pads. Left-pad (`[PAD PAD t1 t2]`) so every row's
last real token is in the final column. The attention mask then hides the pads.

```python
ids[i, : len(p)] = torch.tensor(p, dtype=torch.long)
mask[i, : len(p)] = 1
```
```python
ids[i, max_len - len(p) :] = torch.tensor(p, dtype=torch.long)
mask[i, max_len - len(p) :] = 1
```
Caught by: `test_padded_batch_puts_every_prompts_last_token_in_final_column`,
`test_batched_generation_matches_each_prompt_alone[*]`, `test_each_caller_gets_its_own_completion`,
`test_concurrent_requests_that_each_fit_are_all_served`.

Left padding is the expected fix. Keeping right padding and gathering each row's logits at its own
last real index (plus per-row write positions in the cache) is also correct, but
`test_padded_batch_puts_every_prompts_last_token_in_final_column` pins the left-padded layout of
`pad_prompts`, so that variant fails that one test.

### B2: position ids ignore left padding / KV-cache step

With learned (or rotary) positions, a left-padded row must still start at position 0. Otherwise
its tokens get the same embeddings as a longer prompt's tail and the output changes with batch
composition. Derive positions from the mask, both at prefill and at each decode step.

```python
position_ids = torch.arange(prompt_len, device=device).expand(batch, prompt_len)
...
position_ids = torch.full((batch, 1), prompt_len + step, device=device)
```
```python
position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
...
position_ids = attention_mask.sum(dim=-1, keepdim=True) - 1  # after appending the new 1
```
Caught by: `test_batched_generation_matches_each_prompt_alone[*]`, `test_generation_stops_after_eos`,
`test_each_caller_gets_its_own_completion`. B1 and B2 are both needed: in pr.py, right padding plus
`arange` positions is *self-consistent at prefill*, which is why it looks plausible.

### B3: temperature = 0

`T -> 0` is the greedy limit, but you can't compute it as `logits / 0`: positive logits become
`+inf`, negative ones `-inf`, and a logit of exactly `0` becomes `0 / 0 = nan`. `torch.argmax`
treats `nan` as the maximum, so one zero logit wins outright; otherwise the first `+inf` wins, and
ties between infinities say nothing about which token was best. Branch to `argmax` on the raw
logits before scaling.

```python
logits = logits.float() / params.temperature
...
if params.temperature == 0.0:
    return logits.argmax(dim=-1)
```
```python
logits = logits.float()
...
if params.temperature == 0.0:
    return logits.argmax(dim=-1)
logits = logits / params.temperature
```
Caught by: `test_temperature_zero_is_greedy`.

### B4: top-p removes the threshold-crossing (or the top) token

Nucleus = the smallest prefix of the sorted distribution whose mass is `>= p`. You drop a token
only if the mass *before* it already reaches `p` (exclusive cumsum), or the equivalent: shift the
mask right by one and always keep index 0. With `cumsum > p`, `[0.5, 0.3, ...]`, p=0.6 keeps only
0.5, and `[0.9, ...]`, p=0.5 removes everything.

```python
cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
remove = cumulative > top_p
```
```python
probs = sorted_logits.softmax(dim=-1)
remove = (probs.cumsum(dim=-1) - probs) >= top_p
```
Caught by: `test_top_p_keeps_smallest_set_reaching_p`, `test_top_p_always_keeps_most_likely_token`,
`test_top_p_preserves_original_token_order` (at p of about 1, float cumsum `> p` drops the tail token).

### B5: generation without inference mode / eval

Without `no_grad`, every forward pass keeps activations for backward. Generation memory then
grows with every token, and the KV cache tensors carry `grad_fn`s. Without `eval()`, a model
handed over in train mode samples with dropout, so it's non-deterministic and lower quality.
`InferenceService.__init__` calls `model.eval()`, which hides this for the service but not for
direct `generate()` callers. The fix must also *restore* the caller's mode afterwards (`try` /
`finally`): `generate()` must not silently switch a model that is mid-training into eval mode.
`test_generate_restores_training_flag` checks this, so a bare `model.eval()` is an incomplete fix.

```python
def generate(...):
    input_ids, attention_mask = pad_prompts(...)
```
```python
@torch.inference_mode()
def generate(...):
    was_training = model.training
    model.eval()
    try:
        ...
    finally:
        model.train(was_training)
```
Caught by: `test_generate_does_not_track_gradients`,
`test_generate_is_deterministic_even_if_model_left_in_train_mode`; the mode restore by
`test_generate_restores_training_flag`.

### B6: blocking model call inside the event loop

`async def` doesn't make code non-blocking. A synchronous torch forward pass holds the loop's only
thread, so nothing else is scheduled until it returns. A heartbeat stalls for the whole batch, and
`asyncio.wait_for` timeouts can't fire (the result arrives first, so the "timeout" request
succeeds late). Offload with `asyncio.to_thread` / `loop.run_in_executor` (torch releases the GIL
in kernels). One worker means one batch in flight, so the model isn't used concurrently.

```python
outputs = self._run_batch(batch)
```
```python
outputs = await asyncio.to_thread(self._run_batch, batch)
```
Caught by: `test_event_loop_stays_responsive_during_generation`,
`test_slow_request_times_out_and_service_keeps_working`.

### B7: results routed to the wrong futures

`sorted(...)` reorders the prompts, but `zip(batch, outputs)` still uses the original order. With
`strict=True` the lengths match, so nothing crashes: user A gets user B's completion. That's a
correctness bug and a privacy incident. Sorting also buys nothing, because every row is padded to
the batch max anyway. If you do sort (for example to bucket across batches), carry the index:
`order = sorted(range(n), key=...)` and scatter results back.

```python
prompts = sorted((p.request.prompt_ids for p in batch), key=len)
```
```python
prompts = [p.request.prompt_ids for p in batch]
```
Caught by: `test_each_caller_gets_its_own_completion`.

### B8: worker swallows exceptions; callers hang

Every future the worker took ownership of must be resolved: result, exception, or cancel. Logging
and `continue` keeps the worker alive but orphans the futures.

```python
except Exception:
    logger.exception("inference batch of %d failed", len(batch))
    continue
```
```python
except Exception as exc:
    logger.exception("inference batch of %d failed", len(batch))
    for p in batch:
        if not p.future.done():
            p.future.set_exception(exc)
    continue
```
Caught by: `test_model_failure_is_raised_to_callers`, `test_service_recovers_after_a_failed_batch`.

### B9: per-request length check, per-batch decode length

Validation is per request, but decoding is per batch: every row runs `max(max_new_tokens)` steps,
and row `i` reaches position `len(prompt_i) + max_new - 1`. With `max_seq_len=64`, request A
(60-token prompt, 2 new tokens) and request B (2-token prompt, 40 new tokens) each pass
`_validate`, but batched together A is decoded up to position 99 and `pos_emb` raises
`IndexError`. Because errors fail the whole batch, B fails too: whether a request succeeds depends
on who else was in the batch, which is very hard to reproduce from a bug report.

```python
max_new = max(p.request.max_new_tokens for p in batch)
outputs = generate(self.model, prompts, max_new, ...)
return [out[: p.request.max_new_tokens] for p, out in zip(batch, outputs, strict=True)]
```
```python
budgets = [p.request.max_new_tokens for p in batch]
return generate(self.model, prompts, budgets, ...)  # generate takes one budget per row

# inside generate's decode loop:
finished |= budget <= step + 1  # a row is done once it has its own budget
attention_mask = torch.cat([attention_mask, (~finished)[:, None].long()], dim=1)
```
fixed.py chooses per-row budgets over a batch-level fit check (split or defer rows that don't fit
`max(len) + max(budget)`), because it is one mechanism, never delays a valid request, and also
stops the batch as soon as every row has its tokens. A finished row appends a masked token, so its
position stops advancing; its extra outputs are discarded. `generate` also validates
`len(prompt) + budget <= max_seq_len` per row, so a direct caller gets a `ValueError` instead of
an `IndexError` deep in the model.

Caught by: `test_concurrent_requests_that_each_fit_are_all_served`.

## Production improvements

1. **Config validation at construction** (applied). Put `__post_init__` on `SamplingParams`
   (`temperature >= 0`, `top_k >= 1`, `0 < top_p <= 1`) and `ServiceConfig`, so a bad config fails at
   startup, not as a `nan` deep inside a batch.
   ```python
   def __post_init__(self) -> None:
       if not 0.0 < self.top_p <= 1.0:
           raise ValueError("top_p must be in (0, 1]")
   ```
2. **Graceful shutdown** (applied). `stop()` cancels the worker, then fails with
   `RuntimeError("inference service stopped")` every request the worker had already taken off
   the queue (mid-`_collect_batch` or running in `to_thread`, tracked in `_in_flight`) and every
   request still queued. Without the in-flight part, a caller with `timeout=None` whose batch was
   running at shutdown waits forever. Caught by `test_stop_fails_in_flight_requests_instead_of_hanging`.
   The cancelled batch's thread still runs to completion in the background (threads can't be
   cancelled). If you'd rather finish that batch, use a stop event the worker checks between
   batches and `await` the worker instead of cancelling it.
   ```python
   stranded, self._in_flight = self._in_flight, []
   while not self._queue.empty():
       stranded.append(self._queue.get_nowait())
   for pending in stranded:
       if not pending.future.done():
           pending.future.set_exception(RuntimeError("inference service stopped"))
   ```
3. **Skip dead requests** (applied). Drop futures that are already done (timed out or cancelled
   by the client) before running a batch: `batch = [p for p in batch if not p.future.done()]`.
   This saves compute.
4. **Backpressure** (not applied). Use a bounded queue (`asyncio.Queue(maxsize=...)`) and reject
   with a 429-style error when full, instead of letting latency grow without limit.
   ```python
   try:
       self._queue.put_nowait(pending)
   except asyncio.QueueFull:
       raise ServiceOverloaded from None
   ```
5. **Per-request sampling params / seeds** (not applied). Today sampling is service-wide. Carry
   `SamplingParams` per request and apply temperature/top-k/top-p per row (vectorized: a
   `[B, 1]` temperature tensor), or group batches by params.
6. **Observability** (partly: `enqueued_at` exists but isn't used). Record queue wait, batch size,
   prefill/decode latency and tokens/s, for example via `logger.info` or a metrics client.
   ```python
   wait_ms = (time.perf_counter() - p.enqueued_at) * 1000
   ```
7. **Smarter batching** (not applied). Use continuous/iteration-level batching (admit new
   requests between decode steps, evict finished rows) instead of keeping finished rows in the
   batch until the longest request is done. Short requests currently pay for the longest one.
8. **Device/dtype handling** (not applied). Move `generate` inputs with `non_blocking=True`, run
   `torch.autocast` / bf16 on GPU, and call `torch.cuda.synchronize` before timing.

## Interview talking points

- **Padding side is a property of the architecture.** Decoder-only models append at the end, so
  batched generation must left-pad, mask the pads, and derive position ids from the mask
  (`cumsum - 1`), both at prefill and for each cached decode step. The gold test: batched output
  equals each prompt generated alone.
- **Sampling edge cases.** Treat `temperature=0` as an explicit greedy branch, not a limit you
  compute. Top-p uses the *exclusive* cumulative sum (or shift right by one), so the top token is
  always kept and the threshold-crossing token is included.
- **`async` is not parallel.** CPU/GPU-bound work in a coroutine blocks the whole loop. Offload it
  with `asyncio.to_thread`/an executor, keep a single consumer so the model isn't hit
  concurrently, and prove it with a heartbeat or latency test.
- **Future ownership in a micro-batcher.** Every enqueued future must be resolved exactly once
  (result, exception, or skipped if the client already timed out), and results must map back by
  identity, never by a reordered position. Those rules are what keep requests from hanging and
  users from seeing each other's outputs.
- **Inference hygiene.** Use `inference_mode` plus `eval()` (and restore the mode) for memory,
  speed and determinism. Validate requests up front so one bad request can't fail a whole batch,
  and check limits against what the batch actually runs (per-row budgets), not only per request.
