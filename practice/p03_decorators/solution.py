"""Reference solution for p03_decorators. Try decorators.py yourself before reading this."""

# ruff: noqa: UP047
# Classic TypeVar/ParamSpec on purpose: decorator FACTORIES (`retry(...)`, `timed(...)`) return
# `Callable[[Callable[P, R]], Callable[P, R]]`, where P/R must be solved when the returned
# decorator is applied. PEP 695 `def f[**P, R]` scopes them to the factory call instead, so one
# style (module-level P/R) is used throughout. UP047 would push half the file to PEP 695.

import functools
import inspect
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar, cast, overload

import torch
from torch import nn

# ParamSpec captures a function's whole parameter list, so `Callable[P, R] -> Callable[P, R]`
# tells mypy the decorated function keeps its exact signature (not just `Callable[..., Any]`).
P = ParamSpec("P")
R = TypeVar("R")
M = TypeVar("M", bound=nn.Module)
F = TypeVar("F", bound=Callable[..., Any])


def _synchronize_device() -> None:
    # GPU kernels are launched asynchronously: without a sync, the clock stops when the
    # work is *queued*, not when it is *done*, and you measure almost nothing.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# 1. timed
# ---------------------------------------------------------------------------


class TimedFunction(Protocol[P, R]):
    """What `timed` returns: the original callable plus timing metadata."""

    __name__: str
    __wrapped__: Callable[P, R]
    last_duration: float | None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


@overload
def timed(fn: Callable[P, R], /) -> TimedFunction[P, R]: ...
@overload
def timed(
    *, logger: logging.Logger | None = None, sync: bool = False
) -> Callable[[Callable[P, R]], TimedFunction[P, R]]: ...
def timed(
    fn: Callable[P, R] | None = None,
    /,
    *,
    logger: logging.Logger | None = None,
    sync: bool = False,
) -> TimedFunction[P, R] | Callable[[Callable[P, R]], TimedFunction[P, R]]:
    def decorate(func: Callable[P, R]) -> TimedFunction[P, R]:
        # Resolve the logger once, at decoration time. Default to the *decorated* function's
        # module logger so log records are attributed to the code that was timed.
        log = logger if logger is not None else logging.getLogger(func.__module__)

        @functools.wraps(func)  # copies __name__, __qualname__, __doc__, sets __wrapped__
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if sync:
                _synchronize_device()  # don't charge previously queued work to this call
            start = time.perf_counter()  # monotonic, high resolution; time.time() can jump
            try:
                return func(*args, **kwargs)
            finally:
                # finally: a failing call still gets timed (often the one you care about).
                if sync:
                    _synchronize_device()
                elapsed = time.perf_counter() - start
                timed_wrapper.last_duration = elapsed
                # Lazy %-formatting: the string is only built if the level is enabled.
                log.info("%s took %.3f ms", func.__qualname__, elapsed * 1e3)

        # cast: tell mypy the plain function now also carries `last_duration`.
        timed_wrapper = cast(TimedFunction[P, R], wrapper)
        timed_wrapper.last_duration = None
        return timed_wrapper

    if fn is not None:
        return decorate(fn)  # bare form: @timed  -> timed(fn)
    return decorate  # parameterized form: @timed(...) -> timed(...)(fn)


# ---------------------------------------------------------------------------
# 2. retry
# ---------------------------------------------------------------------------


def retry(
    times: int,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    delay: float = 0.0,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    # Validate eagerly, at decoration time: a bad config should fail at import, not at 3 a.m.
    if times < 1:
        raise ValueError(f"times must be >= 1, got {times}")
    if delay < 0:
        raise ValueError(f"delay must be >= 0, got {delay}")

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions:  # anything not listed propagates immediately
                    if attempt == times:
                        raise  # bare raise keeps the original traceback
                    if delay > 0:
                        time.sleep(delay)
            raise AssertionError("unreachable")  # keeps mypy happy about the return type

        return wrapper

    return decorate


# ---------------------------------------------------------------------------
# 3 + 6a. eval_mode / inference
# ---------------------------------------------------------------------------


@contextmanager
def eval_mode(model: M) -> Iterator[M]:
    # Snapshot EVERY submodule: models often mix modes (e.g. a frozen BatchNorm kept in eval
    # while the rest trains). model.train(True) on exit would silently unfreeze it.
    modes = {m: m.training for m in model.modules()}
    model.eval()
    try:
        yield model
    finally:
        # finally: restore even if the body raised.
        for m, was_training in modes.items():
            m.training = was_training  # set the flag directly; m.train() would recurse


def inference(fn: Callable[Concatenate[M, P], R]) -> Callable[Concatenate[M, P], R]:
    @functools.wraps(fn)
    def wrapper(model: M, /, *args: P.args, **kwargs: P.kwargs) -> R:
        # no_grad: skip building the autograd graph (saves memory and time).
        # eval_mode: dropout off, BatchNorm uses running stats. Both are restored on exit.
        with torch.no_grad(), eval_mode(model):
            return fn(model, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# 4. validate_shapes
# ---------------------------------------------------------------------------


def validate_shapes(**specs: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    # Parse "B,T,D" -> ["B", "T", "D"] once, not on every call.
    parsed = {arg: [d.strip() for d in spec.split(",")] for arg, spec in specs.items()}

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(func)
        unknown = set(parsed) - set(sig.parameters)
        if unknown:  # typo in a spec name -> fail at decoration time
            raise ValueError(f"{func.__qualname__} has no parameter(s) {sorted(unknown)}")

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # bind() maps positional AND keyword args onto parameter names, exactly like the
            # real call would (and raises TypeError for a bad call, same as Python would).
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            seen: dict[str, tuple[int, str]] = {}  # dim name -> (size, arg that fixed it)
            for arg, dims in parsed.items():
                value = bound.arguments[arg]
                if value is None:
                    continue  # optional tensors (e.g. mask=None) are allowed to be absent
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{arg}: expected a Tensor, got {type(value).__name__}")
                spec = ",".join(dims)
                if value.ndim != len(dims):
                    raise ValueError(
                        f"{arg}: expected rank {len(dims)} ({spec}), got shape {tuple(value.shape)}"
                    )
                for dim, size in zip(dims, value.shape, strict=True):
                    if dim.isdigit():  # literal size, e.g. "B,3"
                        if size != int(dim):
                            raise ValueError(
                                f"{arg}: expected size {dim} in ({spec}), "
                                f"got shape {tuple(value.shape)}"
                            )
                    elif dim in seen and seen[dim][0] != size:
                        prev_size, prev_arg = seen[dim]
                        raise ValueError(
                            f"dimension {dim!r} mismatch: {prev_arg} has {dim}={prev_size} "
                            f"but {arg} has {dim}={size}"
                        )
                    elif dim not in seen:
                        seen[dim] = (size, arg)  # first time we see this dim: remember it
            return func(*args, **kwargs)

        return wrapper

    return decorate


# ---------------------------------------------------------------------------
# 5. Registry
# ---------------------------------------------------------------------------


class Registry:
    """Maps string names to classes/factories so configs can say `model: "mlp"`."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[F], F]:
        def decorate(obj: F) -> F:
            if name in self._entries:
                # Two plugins claiming the same name is a bug; silently overwriting hides it.
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._entries[name] = obj
            return obj  # return it unchanged so the class/function stays usable directly

        return decorate

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._entries[name]
        except KeyError:
            # Listing the valid choices turns a config typo into a 5-second fix.
            raise KeyError(
                f"unknown {self.kind} {name!r}; available: {sorted(self._entries)}"
            ) from None

    def build(self, name: str, **kwargs: Any) -> Any:
        return self.get(name)(**kwargs)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def names(self) -> list[str]:
        return sorted(self._entries)


# ---------------------------------------------------------------------------
# 6b. Timer
# ---------------------------------------------------------------------------


class Timer:
    """`with Timer() as t: ...` then read `t.elapsed` (seconds)."""

    def __init__(self, sync: bool = False) -> None:
        self.sync = sync
        self.elapsed: float | None = None
        self._start = 0.0

    def __enter__(self) -> "Timer":
        if self.sync:
            _synchronize_device()
        self._start = time.perf_counter()
        return self  # this is what `as t` binds to

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.sync:
            _synchronize_device()
        self.elapsed = time.perf_counter() - self._start
        # Returning None (falsy) means "don't swallow the exception" -- it propagates.
