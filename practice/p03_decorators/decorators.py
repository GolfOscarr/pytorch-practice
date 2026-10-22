"""Production Python for ML: decorators and context managers.

Real training/serving codebases (Lightning, HF transformers, detectron2, mmcv, timm) lean on a
handful of small decorators and context managers. They look trivial, but each one has a classic
bug that ships to production. Implement the pieces below, then run:

    uv run pytest practice/p03_decorators          # add -rA to list each test
    uv run mypy practice/p03_decorators            # your typing is part of the exercise

What to build, and why each one matters
---------------------------------------
1. `timed` -- a decorator usable BOTH bare (`@timed`) and parameterized
   (`@timed(logger=..., sync=True)`). It measures wall time with `time.perf_counter()`, logs
   `"<qualname> took <x> ms"` at INFO, and stores the last duration (seconds) on
   `wrapper.last_duration` (None before the first call).
   Why: GPU work is ASYNC. `y = model(x)` returns as soon as kernels are queued, so naive timing
   measures ~0. With `sync=True`, synchronize the device before starting AND before stopping
   the clock: call `torch.cuda.synchronize()` if `torch.cuda.is_available()`, and
   `torch.mps.synchronize()` if `torch.backends.mps.is_available()`. Look the functions up at
   call time (`torch.mps.synchronize()`), don't `from torch.mps import synchronize`.
   Also why: without `functools.wraps`, every decorated function is called "wrapper" in logs,
   profilers and stack traces, its docstring vanishes, and `inspect.signature` breaks. `wraps`
   copies `__name__`/`__qualname__`/`__doc__` and sets `__wrapped__`.
   Rules: default logger is `logging.getLogger(fn.__module__)`; record the duration even when
   the function raises (then re-raise).

2. `retry(times, exceptions=(Exception,), delay=0.0)` -- a decorator FACTORY.
   Why: dataloaders reading from S3/NFS, checkpoint uploads and metric pushes fail transiently.
   Rules: `times` is the TOTAL number of attempts; raise `ValueError` immediately (at
   `retry(...)` time, before decorating anything) if `times < 1`. Only exceptions that are
   instances of `exceptions` are retried -- anything else propagates on the first failure
   (retrying a `KeyError` from a bug just hides the bug three times). After the final attempt,
   re-raise the LAST exception with its original traceback. Sleep `delay` seconds between
   attempts (not after the last one); a negative `delay` is a `ValueError` at `retry(...)` time.
   Call it as `time.sleep(...)` after `import time` (not `from time import sleep`), so tests
   can patch it.

3. `inference(fn)` -- decorates `fn(model, *args, **kwargs)` (first argument is an `nn.Module`,
   which also covers methods of a Module, where it is `self`). Runs fn under `torch.no_grad()`
   with the model in eval mode, and restores the previous modes afterward, EVEN IF fn RAISES.
   Why: forgetting `.eval()` leaves dropout on and BatchNorm updating its running stats during
   validation -- silently wrong metrics. Forgetting to restore `.train()` silently trains the
   rest of the epoch in eval mode. Reuse `eval_mode` (below) to implement it. Preserve the
   function's metadata (`functools.wraps`).

4. `validate_shapes(**specs)`, e.g. `@validate_shapes(x="B,T,D", mask="B,T")`.
   Each spec is a comma-separated list of dims: a NAME (any identifier) or an integer literal
   (fixed size, e.g. "B,3"). On every call: bind the arguments with `inspect.signature(fn)`
   (so positional and keyword calls both work; apply defaults), then for each spec'd argument:
     - `None` is allowed and skipped (optional tensors like `mask=None`);
     - not a `torch.Tensor`                -> `TypeError`;
     - wrong rank                          -> `ValueError` naming the argument, the spec and
                                              the actual shape;
     - literal size mismatch               -> `ValueError` naming the argument, the spec and
                                              the actual shape;
     - a named dim disagrees across args   -> `ValueError` naming the dim AND both arguments,
                                              e.g. "dimension 'T' mismatch: x has T=5 but
                                              mask has T=4".
   Format shapes as `tuple(t.shape)`, e.g. `(2, 5, 1)` (not `torch.Size([2, 5, 1])`), and quote
   dim names with `{dim!r}`, e.g. `'T'`. Preserve the function's metadata (`functools.wraps`).
   A spec for a parameter that `fn` doesn't have is a typo: raise `ValueError` when decorating.
   Why: shape bugs in PyTorch often don't crash -- broadcasting turns a (B,T) vs (B,1,T) mix-up
   into a wrong-but-valid tensor. Failing early with names beats debugging a NaN loss.

5. `Registry` -- `MODELS = Registry("model")`, then `@MODELS.register("mlp")` on a class or
   factory function; `MODELS.get("mlp")` returns it, `MODELS.build("mlp", **kwargs)` calls it.
   Why: configs say `model: mlp`; the registry maps that string to code without an if/elif
   chain. Rules: `register` returns the object unchanged; registering a name twice raises
   `ValueError` whose message includes the name (silent overwrite = two plugins fighting);
   `get` of an unknown name raises `KeyError` whose message includes the requested name and
   the available names; support `name in registry` and
   `registry.names()` (sorted list).

6. Context managers, one written each way:
   - `eval_mode(model)` with `@contextlib.contextmanager`: puts the model in eval mode, yields
     it, and on exit restores the training flag of EVERY submodule to what it was before.
     Hint: `model.train()` on exit is wrong if a submodule (e.g. a frozen BatchNorm) was
     deliberately in eval mode -- snapshot `m.training` for all `model.modules()`.
     Hint: put the restore in a `finally`, or an exception in the body skips it.
   - `Timer` as a class with `__enter__`/`__exit__`: `with Timer() as t: ...` sets `t.elapsed`
     (seconds, via `time.perf_counter()`) on exit -- also when the body raises, and it must NOT
     swallow the exception. Optional `sync=True` synchronizes like `timed`.

Typing: use `ParamSpec`/`TypeVar` (already declared below) so a decorated function keeps its
exact signature for mypy. `Concatenate[M, P]` expresses "first argument is a Module".
"""

# ruff: noqa: UP047
# Classic TypeVar/ParamSpec on purpose: decorator FACTORIES (`retry(...)`, `timed(...)`) return
# `Callable[[Callable[P, R]], Callable[P, R]]`, where P/R must be solved when the returned
# decorator is applied. PEP 695 `def f[**P, R]` scopes them to the factory call instead, so one
# style (module-level P/R) is used throughout. UP047 would push half the file to PEP 695.

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar, overload

from torch import nn

P = ParamSpec("P")
R = TypeVar("R")
M = TypeVar("M", bound=nn.Module)
F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# 1. timed
# ---------------------------------------------------------------------------


class TimedFunction(Protocol[P, R]):
    """What `timed` returns: the original callable plus timing metadata.

    Hint: build a normal `functools.wraps` wrapper, then `typing.cast` it to this type
    so mypy lets you set `last_duration` on it.
    """

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
    """Time each call of a function, log it, and remember the last duration.

    Usable as `@timed` or `@timed(logger=..., sync=...)`.

    Args:
        fn: the function, when used bare (`@timed`). None in the parameterized form.
        logger: where to log; defaults to `logging.getLogger(fn.__module__)`.
        sync: synchronize CUDA/MPS before starting and before stopping the clock.

    Returns:
        Bare form: the wrapped function. Parameterized form: a decorator.
        The wrapped function has `.last_duration` (seconds, None before the first call).
    """
    raise NotImplementedError("Your code here")


# ---------------------------------------------------------------------------
# 2. retry
# ---------------------------------------------------------------------------


def retry(
    times: int,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    delay: float = 0.0,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator factory: call the function up to `times` times until it succeeds.

    Args:
        times: total number of attempts (>= 1).
        exceptions: exception types that trigger a retry; others propagate immediately.
        delay: seconds to sleep between attempts.

    Returns:
        A decorator that preserves the function's signature and metadata.

    Raises:
        ValueError: if `times < 1` or `delay < 0` (raised by `retry(...)` itself).
    """
    raise NotImplementedError("Your code here")


# ---------------------------------------------------------------------------
# 3 + 6a. eval_mode / inference
# ---------------------------------------------------------------------------


@contextmanager
def eval_mode(model: M) -> Iterator[M]:
    """Context manager: model in eval mode inside, every submodule's mode restored on exit.

    Args:
        model: any nn.Module.

    Yields:
        The same model.
    """
    raise NotImplementedError("Your code here")


def inference(fn: Callable[Concatenate[M, P], R]) -> Callable[Concatenate[M, P], R]:
    """Decorator: run `fn(model, ...)` under `torch.no_grad()` and `eval_mode(model)`.

    Args:
        fn: a function whose first positional argument is an nn.Module.

    Returns:
        The wrapped function (same signature and metadata). Grad mode and training modes are
        restored afterward, even if `fn` raises.
    """
    raise NotImplementedError("Your code here")


# ---------------------------------------------------------------------------
# 4. validate_shapes
# ---------------------------------------------------------------------------


def validate_shapes(**specs: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator factory checking tensor ranks and named-dim agreement across arguments.

    Args:
        **specs: parameter name -> comma-separated dims, e.g. `x="B,T,D", mask="B,T"`.

    Returns:
        A decorator that validates on every call, then calls the function. The wrapped
        function keeps the original's signature and metadata.

    Raises:
        ValueError: when decorating, if a spec names a parameter `fn` doesn't have;
            when calling, on rank / fixed-size / named-dim mismatch.
        TypeError: when calling, if a spec'd argument is neither a Tensor nor None.
    """
    raise NotImplementedError("Your code here")


# ---------------------------------------------------------------------------
# 5. Registry
# ---------------------------------------------------------------------------


class Registry:
    """Maps string names to classes/factories so configs can say `model: "mlp"`."""

    def __init__(self, kind: str) -> None:
        self.kind = kind  # used in error messages, e.g. "unknown model 'mpl'"
        self._entries: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[F], F]:
        """Return a decorator that stores the decorated object under `name`.

        Raises:
            ValueError: (from the decorator) if `name` is already registered; the message
                includes `name`.
        """
        raise NotImplementedError("Your code here")

    def get(self, name: str) -> Callable[..., Any]:
        """Return the object registered under `name`.

        Raises:
            KeyError: if unknown; the message includes `name` and the available names.
        """
        raise NotImplementedError("Your code here")

    def build(self, name: str, **kwargs: Any) -> Any:
        """Look up `name` and call it with `**kwargs`."""
        raise NotImplementedError("Your code here")

    def __contains__(self, name: object) -> bool:
        raise NotImplementedError("Your code here")

    def names(self) -> list[str]:
        """Sorted list of registered names."""
        raise NotImplementedError("Your code here")


# ---------------------------------------------------------------------------
# 6b. Timer
# ---------------------------------------------------------------------------


class Timer:
    """`with Timer() as t: ...` then read `t.elapsed` (seconds)."""

    def __init__(self, sync: bool = False) -> None:
        self.sync = sync
        self.elapsed: float | None = None  # set on exit

    def __enter__(self) -> "Timer":
        """Start the clock and return the object bound by `as`."""
        raise NotImplementedError("Your code here")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the clock and set `self.elapsed`. Must not suppress exceptions."""
        raise NotImplementedError("Your code here")
