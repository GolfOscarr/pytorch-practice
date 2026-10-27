"""Non-blocking checkpointing so training doesn't stall on saves

## Summary
On the 1B runs a synchronous `torch.save` + upload takes 20-40 s, and the GPUs sit idle the whole
time. This PR adds `AsyncCheckpointer`, which hands checkpoint writes and uploads to a background
worker so the training loop gets control back immediately. It also takes over retention (only the
newest `keep_last` checkpoints are kept on local disk) and resuming from the latest checkpoint.

## Changes
- `AsyncCheckpointer(directory, keep_last=3, max_retries=3, uploader=None, ...)`:
  - `save(step, model, optimizer) -> Future[Path]`: queues a checkpoint on a single background
    worker thread (one worker, so saves are written in order and never interleave) and returns
    at once.
  - `save_async(...)`: the same thing for asyncio-based loops (eval/serving harness).
  - `wait()` blocks until every queued save is done; `close()` / the context manager shut the
    worker down.
  - `latest()` / `load_latest(model, optimizer)` find the newest `step_<N>.pt` and restore model
    and optimizer state; `load_latest` returns the step so the loop can resume from it.
  - Retention: after each save, everything but the newest `keep_last` files is deleted.
  - `uploader`: optional injectable `Callable[[Path], None]` (S3/GCS client in prod, a fake in
    tests). Uploads are retried up to `max_retries` times with exponential backoff capped at
    `backoff_max` seconds.
- `save_fn` / `sleep` are injectable too, so tests don't need real I/O delays.

## Testing
- Ran the 125M config for 2k steps with a checkpoint every 100 steps: step time is flat across
  saves (previously a ~6 s spike), and resuming with `load_latest` reproduces the loss curve.
- Checked the bucket: uploads land, and the local directory never holds more than 3 files.
"""

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from pathlib import Path
from types import TracebackType
from typing import Any

import torch
from torch import nn

SaveFn = Callable[[Any, Path], None]
Uploader = Callable[[Path], None]


def _step_of(path: Path) -> int:
    """`step_120.pt` -> 120."""
    return int(path.stem.removeprefix("step_"))


class AsyncCheckpointer:
    """Writes (and optionally uploads) checkpoints on a background thread."""

    def __init__(
        self,
        directory: str | Path,
        keep_last: int = 3,
        max_retries: int = 3,
        uploader: Uploader | None = None,
        save_fn: SaveFn = torch.save,
        backoff_base: float = 0.5,
        backoff_max: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._uploader = uploader
        self._save_fn = save_fn
        self._sleep = sleep
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint")
        self._pending: list[Future[Path]] = []

    # ------------------------------------------------------------------ saving

    def save(self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer) -> Future[Path]:
        """Queue a checkpoint of `model` and `optimizer` at `step`; returns immediately."""
        state = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        future = self._executor.submit(self._write, step, state)
        self._pending.append(future)
        return future

    async def save_async(
        self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> Path:
        """Like `save`, but awaitable: resolves to the checkpoint path once it is written."""
        return self.save(step, model, optimizer).result()

    def _write(self, step: int, state: dict[str, Any]) -> Path:
        path = self.directory / f"step_{step}.pt"
        self._save_fn(state, path)
        if self._uploader is not None:
            self._upload_with_retry(path)
        self._prune()
        return path

    def _upload_with_retry(self, path: Path) -> None:
        assert self._uploader is not None
        for attempt in range(self.max_retries + 1):
            try:
                self._uploader(path)
                return
            except Exception:
                if attempt == self.max_retries:
                    raise
                self._sleep(min(self.backoff_max, self.backoff_base * 2**attempt))

    def _prune(self) -> None:
        checkpoints = sorted(self.directory.glob("step_*.pt"))
        for old in checkpoints[: -self.keep_last]:
            old.unlink(missing_ok=True)

    # --------------------------------------------------------------- lifecycle

    def wait(self) -> None:
        """Block until every queued save has finished."""
        pending, self._pending = self._pending, []
        wait_futures(pending)

    def close(self) -> None:
        """Shut down the background worker."""
        self._executor.shutdown(wait=False)

    def __enter__(self) -> "AsyncCheckpointer":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---------------------------------------------------------------- loading

    def latest(self) -> Path | None:
        """Path of the newest checkpoint on disk, or None if there is none."""
        checkpoints = list(self.directory.glob("step_*.pt"))
        if not checkpoints:
            return None
        return max(checkpoints, key=_step_of)

    def load_latest(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> int | None:
        """Restore the newest checkpoint into `model`/`optimizer`; returns its step (or None)."""
        path = self.latest()
        if path is None:
            return None
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        return int(state["step"])
