"""Reference solution for review/r12_async_checkpoint: non-blocking checkpointing.

Same public API as pr.py. Lines changed for a planted bug carry `# FIX(Bn)` comments that map
to solution/REVIEW.md; `# IMPROVE` marks production hardening beyond the bug fixes.
"""

import asyncio
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from pathlib import Path
from types import TracebackType
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

SaveFn = Callable[[Any, Path], None]
Uploader = Callable[[Path], None]

# FIX(B8): only transient failures are worth retrying; auth/validation errors never heal.
RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)


_CHECKPOINT_NAME = re.compile(r"step_(\d+)\.pt")


def _step_of(path: Path) -> int:
    """`step_120.pt` -> 120."""
    return int(path.stem.removeprefix("step_"))


def _checkpoint_files(directory: Path) -> list[Path]:
    """Checkpoints in `directory`, ignoring stray files like `step_best.pt` that would make
    `int()` raise."""
    return [p for p in directory.glob("step_*.pt") if _CHECKPOINT_NAME.fullmatch(p.name)]


def _snapshot(obj: Any) -> Any:
    """Deep-copy every tensor in a (nested) state dict to CPU memory owned by the snapshot."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _snapshot(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return type(obj)(_snapshot(v) for v in obj)
    return obj


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
        # IMPROVE: validate config up front (keep_last=0 would make `[:-0]` keep everything).
        if keep_last < 1:
            raise ValueError(f"keep_last must be >= 1, got {keep_last}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
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
        self._closed = False

    # ------------------------------------------------------------------ saving

    def save(self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer) -> Future[Path]:
        """Queue a checkpoint of `model` and `optimizer` at `step`; returns immediately."""
        if self._closed:  # IMPROVE: fail loudly instead of RuntimeError from the executor
            raise RuntimeError("AsyncCheckpointer is closed")
        # FIX(B1): snapshot on the caller's thread. state_dict() returns *references* to the live
        # parameters/moments, which the next optimizer.step() mutates in place while the worker
        # is still pickling them. Copying to CPU here is the only synchronous cost of a save.
        state = _snapshot(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
        )
        future = self._executor.submit(self._write, step, state)
        self._pending.append(future)
        return future

    async def save_async(
        self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> Path:
        """Like `save`, but awaitable: resolves to the checkpoint path once it is written."""
        # FIX(B6): never block the event loop on Future.result(); bridge it to an awaitable.
        return await asyncio.wrap_future(self.save(step, model, optimizer))

    def _write(self, step: int, state: dict[str, Any]) -> Path:
        path = self.directory / f"step_{step}.pt"
        # FIX(B2): write to a unique temp file, then atomically rename. A crash mid-write leaves
        # only a `*.tmp` file (ignored by latest()), never a truncated `step_N.pt`.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            self._save_fn(state, tmp)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        if self._uploader is not None:
            self._upload_with_retry(path)
        self._prune()
        logger.info("checkpoint step=%d written to %s", step, path)  # IMPROVE: observability
        return path

    def _upload_with_retry(self, path: Path) -> None:
        assert self._uploader is not None
        for attempt in range(self.max_retries + 1):
            try:
                self._uploader(path)
                return
            except RETRYABLE_ERRORS as exc:  # FIX(B8): was `except Exception`
                if attempt == self.max_retries:
                    raise
                delay = min(self.backoff_max, self.backoff_base * 2**attempt)
                logger.warning("upload of %s failed (%r); retry in %.1fs", path, exc, delay)
                self._sleep(delay)

    def _prune(self) -> None:
        # FIX(B4): sort by the parsed step number. Lexicographic order puts step_10 before
        # step_9, so the newest checkpoints were the ones being deleted.
        checkpoints = sorted(_checkpoint_files(self.directory), key=_step_of)
        for old in checkpoints[: -self.keep_last]:
            old.unlink(missing_ok=True)

    # --------------------------------------------------------------- lifecycle

    def wait(self) -> None:
        """Block until every queued save has finished; re-raise the first failure."""
        pending, self._pending = self._pending, []
        wait_futures(pending)
        # FIX(B3): concurrent.futures stores worker exceptions on the Future; nobody sees them
        # unless someone calls .result()/.exception(). Surface them here.
        for future in pending:
            exc = future.exception()
            if exc is not None:
                raise exc

    def close(self) -> None:
        """Flush pending saves (re-raising failures) and shut down the background worker."""
        if self._closed:
            return
        self._closed = True
        # FIX(B7): drain the queue before shutting down; shutdown(wait=False) returned while the
        # last checkpoint was still being written, and its errors were never surfaced.
        try:
            self.wait()
        finally:
            self._executor.shutdown(wait=True)

    def __enter__(self) -> "AsyncCheckpointer":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is None:
            self.close()
            return
        # The body already failed: don't let a background save error replace that exception.
        try:
            self.close()
        except Exception:
            logger.exception("background checkpoint save failed while handling another error")

    # ---------------------------------------------------------------- loading

    def latest(self) -> Path | None:
        """Path of the newest checkpoint on disk, or None if there is none."""
        checkpoints = _checkpoint_files(self.directory)
        if not checkpoints:
            return None
        return max(checkpoints, key=_step_of)

    def load_latest(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> int | None:
        """Restore the newest checkpoint into `model`/`optimizer`; returns its step (or None)."""
        path = self.latest()
        if path is None:
            return None
        # FIX(B5): weights_only=True restricts unpickling to tensors/containers/primitives, so a
        # tampered checkpoint cannot execute code. Our state dicts need nothing more.
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        return int(state["step"])
