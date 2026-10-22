"""Custom autograd functions: write forward AND backward by hand.

Build three `torch.autograd.Function` subclasses plus thin functional wrappers, then run:
    uv run pytest practice/p02_autograd          (add -rA to list each test)

1. `LogSumExp.apply(x, dim)` -> log(sum(exp(x), dim)), with `dim` removed from the shape.
   - Must be numerically stable: exp(1000.) is inf in float32/float64. Use the max-shift trick:
         logsumexp(x) = m + log(sum(exp(x - m))),   m = max(x, dim)
     so the largest exponent is exp(0) = 1.
   - Edge case: if a whole slice is -inf, m is -inf and x - m is nan (-inf - -inf). The correct
     answer is -inf (log of 0). Replace a non-finite m by 0 before shifting.
   - Backward: d logsumexp / dx = softmax(x, dim) = exp(x - out). The upstream gradient has
     `dim` removed, so unsqueeze it (and the saved output) back before broadcasting.
   - `dim` is a plain int, not a tensor: backward returns one gradient per forward input, so
     return `None` in its slot.

2. `StraightThroughRound.apply(x)` -> torch.round(x) in forward, but backward passes the
   upstream gradient through unchanged (straight-through estimator, used for quantization-aware
   training). Real round() has zero gradient almost everywhere, which would stop learning.

3. `SiLU.apply(x)` -> x * sigmoid(x), with a hand-derived backward:
         d/dx [x * s(x)] = s(x) * (1 + x * (1 - s(x))),   s = sigmoid
   Use torch.sigmoid (it is stable for large |x|). Avoid exp(x) / (1 + exp(x)): for large x it
   becomes inf / inf = nan.

Rules / hints:
- forward and backward are @staticmethods; the first argument is `ctx`.
- Tensors needed by backward go through `ctx.save_for_backward(...)` and come back from
  `ctx.saved_tensors`. Non-tensors (like `dim`) are stored as plain attributes: `ctx.dim = dim`.
- What to save is a design choice. SiLU needs its INPUT x. LogSumExp can save its OUTPUT too,
  since softmax = exp(x - out) reuses the forward result instead of recomputing it.
- Don't call torch.logsumexp / torch.nn.functional.silu / torch.softmax inside your functions;
  build them from basic ops (max, exp, log, sum, sigmoid, ...).
- Verify gradients with `torch.autograd.gradcheck` on CPU float64 inputs. It compares your
  backward against finite differences; float32 is too imprecise for that.
"""

from typing import Any

import torch


class LogSumExp(torch.autograd.Function):
    """Numerically stable log-sum-exp over one dimension."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute log(sum(exp(x), dim)) without overflow.

        Args:
            ctx: autograd context for stashing what backward needs.
            x: input tensor (floating dtype).
            dim: dimension to reduce (negative values allowed). It is removed from the output.

        Returns:
            Tensor with the shape of x minus `dim`.
        """
        raise NotImplementedError("Your code here")

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Return (grad wrt x, None for dim).

        Args:
            ctx: the same context forward filled.
            grad_output: dLoss/dOut, with the shape of forward's output.

        Returns:
            (grad_output broadcast over `dim` times softmax(x, dim), None).
        """
        raise NotImplementedError("Your code here")


class StraightThroughRound(torch.autograd.Function):
    """round() in forward, identity in backward (straight-through estimator)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        """Round x to the nearest integer (same semantics as torch.round).

        Args:
            ctx: autograd context (unused here).
            x: input tensor.

        Returns:
            Rounded tensor, same shape/dtype as x.
        """
        raise NotImplementedError("Your code here")

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """Pass the upstream gradient through unchanged.

        Args:
            ctx: autograd context.
            grad_output: dLoss/dOut.

        Returns:
            dLoss/dx, equal to grad_output.
        """
        raise NotImplementedError("Your code here")


class SiLU(torch.autograd.Function):
    """SiLU / swish: x * sigmoid(x), with a hand-written backward."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        """Compute x * sigmoid(x).

        Args:
            ctx: autograd context; save what backward needs.
            x: input tensor.

        Returns:
            Tensor with the same shape as x.
        """
        raise NotImplementedError("Your code here")

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """Return grad_output * d silu(x) / dx.

        Args:
            ctx: the same context forward filled.
            grad_output: dLoss/dOut.

        Returns:
            dLoss/dx, same shape as x.
        """
        raise NotImplementedError("Your code here")


def logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Functional wrapper around LogSumExp.apply.

    Args:
        x: input tensor.
        dim: dimension to reduce (default: last).

    Returns:
        log(sum(exp(x), dim)) with `dim` removed.
    """
    raise NotImplementedError("Your code here")


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Functional wrapper around StraightThroughRound.apply.

    Args:
        x: input tensor.

    Returns:
        torch.round(x) in value, identity in gradient.
    """
    raise NotImplementedError("Your code here")


def silu(x: torch.Tensor) -> torch.Tensor:
    """Functional wrapper around SiLU.apply.

    Args:
        x: input tensor.

    Returns:
        x * sigmoid(x).
    """
    raise NotImplementedError("Your code here")
