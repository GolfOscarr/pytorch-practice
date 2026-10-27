# r12 — Async checkpointing: answer key

## Bugs

| ID | Severity | Location | What's wrong | Impact |
|----|----------|----------|--------------|--------|
| B1 | Critical | pr.py:81-85 | Background worker serializes the live `state_dict()` (references to the parameter / Adam moment tensors), not a snapshot | silently wrong (checkpoint mixes later steps) |
| B2 | High | pr.py:98 | `torch.save` writes straight to the final `step_N.pt`; a crash mid-write leaves a truncated file that `latest()` picks | crash on resume / lost run |
| B3 | High | pr.py:125 | `wait()` waits on futures but never checks them, so background exceptions vanish | silently wrong (no checkpoints, no error) |
| B4 | High | pr.py:116 | Retention sorts paths lexicographically (`step_10 < step_9`) and deletes the *newest* checkpoints | silently wrong (data loss) |
| B5 | High | pr.py:156 | `torch.load(..., weights_only=False)` unpickles arbitrary objects | security (RCE from a checkpoint file) |
| B6 | Medium | pr.py:94 | `async def save_async` calls the blocking `Future.result()` and freezes the event loop for the whole write | perf / liveness |
| B7 | Medium | pr.py:129 | `close()` is `shutdown(wait=False)`: returns before pending saves finish (and never surfaces their errors) | lost last checkpoint on exit |
| B8 | Medium | pr.py:110 | Upload retry catches `Exception`: permanent errors (403, bad bucket, bugs in the uploader) are retried with backoff | perf (wasted retries, seconds to minutes of delay), noisy failure |

### B1: Checkpoint aliases live training state

`nn.Module.state_dict()` and `Optimizer.state_dict()` return **references** to the same storage
the model trains on (detached tensors that share memory). `optimizer.step()` updates parameters
and `exp_avg`/`exp_avg_sq` *in place*. Once the write runs on another thread, the training loop
moves on and keeps mutating those tensors while `torch.save` is still pickling them. The file
ends up with values from a later step, or a mix of steps (some tensors pickled before the
update, some after). Nothing crashes; the resume just quietly comes from the wrong state.

The fix is to take a snapshot on the caller's thread: a deep copy of every tensor to CPU. That
copy is the only synchronous cost of a save (a D2H memcpy, cheap next to serialization and
upload). This is how PyTorch DCP's `async_save` and most production checkpointers work.

```python
# buggy
state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
```
```python
# fixed
def _snapshot(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _snapshot(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return type(obj)(_snapshot(v) for v in obj)
    return obj


state = _snapshot({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()})
```
Caught by `test_checkpoint_holds_values_from_save_time_even_if_training_continues`. A gated
`save_fn` holds the write until the test has run 3 more optimizer steps, which forces the race
every time.

### B2: Non-atomic write

A write can be interrupted: OOM kill, preemption, disk full, an exception in the serializer. A
partial `step_N.pt` is then the file with the highest step, so `latest()`/`load_latest()` pick
it and resume fails, or worse, the job restarts from step 0 if the caller treats load errors as
"no checkpoint". The standard pattern is to write to a temp file in the **same directory** (so
the rename stays on one filesystem), then `os.replace` it onto the final name. On POSIX the
rename is atomic, so readers see either the old file or the complete new one. Use a unique
temp name so two writers never share one, and delete the temp file on failure. For full
durability, also `fsync` the file and the directory.

```python
# buggy
path = self.directory / f"step_{step}.pt"
self._save_fn(state, path)
```
```python
# fixed
tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
try:
    self._save_fn(state, tmp)
    os.replace(tmp, path)
except BaseException:
    tmp.unlink(missing_ok=True)
    raise
```
Caught by `test_failed_write_does_not_become_the_latest_checkpoint`.

### B3: Swallowed background exceptions

A `concurrent.futures.Future` stores the worker's exception, and nobody sees it unless someone
calls `.result()` or `.exception()`. `concurrent.futures.wait()` only waits; it never raises. So
in the PR, a full disk or failed upload produces no checkpoints and no error, and you find out
when you try to resume. "Fire and forget" needs an explicit error channel: re-raise in
`wait()`/`close()` (as fixed.py does), or attach `add_done_callback` to log or flag failures
right away.

```python
# buggy
pending, self._pending = self._pending, []
wait_futures(pending)
```
```python
# fixed
wait_futures(pending)
for future in pending:
    exc = future.exception()
    if exc is not None:
        raise exc
```
Caught by `test_wait_reraises_background_save_failure`.

### B4: Retention deletes the newest checkpoints

`sorted()` on paths compares strings: `"step_10.pt" < "step_11.pt" < "step_8.pt" < "step_9.pt"`.
Once the step count gains a digit, `checkpoints[:-keep_last]` drops the *newest* files and keeps
old ones. Worse, `latest()` sorts numerically (`max(key=_step_of)`), so the two disagree and
the bug never shows up in a short run. Fix: sort with the same numeric key, or zero-pad the
filename (`step_{step:08d}.pt`) so string order matches numeric order.

```python
# buggy
checkpoints = sorted(self.directory.glob("step_*.pt"))
```
```python
# fixed
checkpoints = sorted(self.directory.glob("step_*.pt"), key=_step_of)
```
Caught by `test_retention_keeps_the_newest_checkpoints` (steps 5..12, keep 3 → must be 10, 11, 12).

### B5: Unsafe deserialization

`torch.load` with `weights_only=False` is `pickle.load`. A checkpoint can carry a `__reduce__`
payload that runs arbitrary code on load. Checkpoint directories and buckets are shared, synced,
and downloaded from model hubs, so this is a real attack surface. Model and optimizer state
dicts contain only tensors, dicts, lists, and primitives, so `weights_only=True` (the default
since torch 2.6, but here the PR explicitly opted out) loses nothing. If you need custom types,
allow-list them with `torch.serialization.add_safe_globals`.

```python
# buggy
state = torch.load(path, map_location="cpu", weights_only=False)
```
```python
# fixed
state = torch.load(path, map_location="cpu", weights_only=True)
```
Caught by `test_load_latest_refuses_to_execute_pickled_code`.

### B6: Blocking call inside `async def`

`async def` does not make a function non-blocking. `Future.result()` blocks the thread, and here
that thread is the event loop thread, so every other coroutine (heartbeats, health checks,
request handlers) stalls for the whole write and upload. Bridge the thread-pool future into an
awaitable with `asyncio.wrap_future`, or run blocking work with `asyncio.to_thread` /
`loop.run_in_executor`.

```python
# buggy
return self.save(step, model, optimizer).result()
```
```python
# fixed
return await asyncio.wrap_future(self.save(step, model, optimizer))
```
Caught by `test_save_async_keeps_the_event_loop_responsive`. The write waits up to 1 s for a
heartbeat coroutine. It sees the heartbeat only if the loop was free to run it.

### B7: `close()` doesn't drain pending saves

`shutdown(wait=False)` returns right away. Queued work still runs (on a normal interpreter exit
CPython even joins the worker threads, so the write usually finishes), but `close()` returning
no longer means the checkpoint is durable:
- anything that runs right after `close()` (an upload, a copy, `load_latest`, reporting "done"
  to an orchestrator) races the writer;
- a hard kill (SIGKILL, preemption, `os._exit`, a launcher tearing the process down after `main`
  returns) loses the final checkpoint, which is the one you care about most;
- failures are never surfaced, because nothing ever looks at the futures. This is the strongest
  argument. Fix: `wait()` (which re-raises) and then `shutdown(wait=True)`, making `close()`
idempotent.

```python
# buggy
def close(self) -> None:
    self._executor.shutdown(wait=False)
```
```python
# fixed
def close(self) -> None:
    if self._closed:
        return
    self._closed = True
    try:
        self.wait()
    finally:
        self._executor.shutdown(wait=True)
```
Caught by `test_close_flushes_pending_saves`.

### B8: Retrying non-retryable errors

Retries are for *transient* failures (connection reset, timeout, throttling). A 403, a missing
bucket, or a `TypeError` in the uploader fails the same way every time. Retrying it burns
`sum(backoff)` seconds (here 0.5+1+2 = 3.5 s; with realistic settings, minutes) on the single
checkpoint worker, delaying every queued save behind it, and buries the real error in retry
logs. Classify errors and retry only the retryable set. The backoff itself is correctly capped
in the PR. Adding jitter would be a further improvement.

```python
# buggy
except Exception:
```
```python
# fixed
RETRYABLE_ERRORS = (ConnectionError, TimeoutError)
...
except RETRYABLE_ERRORS as exc:
```
Caught by `test_upload_does_not_retry_permanent_errors` (`PermissionError` must be tried exactly once).

## Production improvements

1. **Validate config** *(applied in fixed.py)*: `keep_last=0` makes `checkpoints[:-0]` an empty
   slice, so nothing is ever deleted. Negative `max_retries` silently skips uploads.
   ```python
   if keep_last < 1:
       raise ValueError(f"keep_last must be >= 1, got {keep_last}")
   ```
2. **Logging / observability** *(applied: logs on write and each retry)*: also emit metrics
   for snapshot time (blocking), write time, upload time, and queue depth. Queue depth is the
   number to alert on: if saves pile up faster than they drain, host RAM fills with snapshots.
   ```python
   logger.info("checkpoint step=%d written to %s", step, path)
   ```
3. **Reject use after close** *(applied)*: `save()` after `close()` raises a clear
   `RuntimeError` instead of the executor's generic one, and `close()` is idempotent.
4. **Bound in-flight snapshots / back-pressure** *(not applied)*: each pending save holds a full
   CPU copy of model + optimizer (about 3x model size for Adam). Cap with a semaphore, or skip
   or coalesce if the previous save hasn't finished:
   ```python
   self._slots = threading.BoundedSemaphore(max_in_flight)
   self._slots.acquire()  # in save(), before snapshotting
   future.add_done_callback(lambda _: self._slots.release())
   ```
5. **Report failures as they happen** *(not applied)*: add `future.add_done_callback(self._on_done)` so
   failures are logged when they happen, not only at the next `wait()`. Optionally fail the
   next `save()` if the previous one failed.
6. **Durability** *(not applied)*: after the temp write, `os.fsync` the file and the directory before
   relying on the rename; write a small `latest` manifest (step, sha256, size) so `latest()` can
   verify integrity instead of trusting file names; zero-pad names (`step_{step:08d}.pt`) so
   `ls` order matches numeric order.
7. **Retry policy as data** *(partially applied: `RETRYABLE_ERRORS` constant)*: a frozen
   `@dataclass RetryPolicy(max_retries, base, cap, jitter, retry_on)` with full jitter
   (`random.uniform(0, min(cap, base * 2**n))`) to avoid synchronized retry storms across
   ranks.
8. **Distributed awareness** *(not applied)*: under DDP/FSDP only rank 0 (or each rank, for its
   shard) should write. Use `torch.distributed.checkpoint.async_save` for sharded state rather
   than rolling your own.

## Interview talking points

- **"Async" moves the problem from *time* to *consistency*.** Once the write is off the critical
  path, the data must not change under the writer. Snapshot (clone to CPU) synchronously, then
  serialize and upload in the background. Know the cost: one D2H copy plus host RAM per
  in-flight save.
- **Crash consistency = temp file + atomic rename (+ fsync).** Readers should only ever see
  complete files. Pair it with numeric (not lexicographic) ordering, so "latest" and
  "retention" agree on what newest means.
- **Background errors need an explicit channel.** A Future holds its exception until someone
  asks. `wait()`/`close()` must re-raise, and `close()` must drain the queue, or you lose the
  last checkpoint, the one that matters.
- **`async def` with a blocking call is still blocking.** Use `asyncio.wrap_future` /
  `to_thread`. A heartbeat coroutine is the standard way to test it.
- **Retry only what's transient, with capped, jittered backoff. Never unpickle untrusted data**
  (`weights_only=True`). Both are the kind of "production judgment" signal interviewers
  listen for.
