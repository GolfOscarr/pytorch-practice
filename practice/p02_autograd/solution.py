"""Reference solution for p02_autograd. Try functions.py yourself before reading this."""

from typing import Any

import torch


class LogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int) -> torch.Tensor:
        m = x.amax(dim=dim, keepdim=True)
        # If a slice is all -inf, m is -inf and x - m would be nan. Shifting by 0 instead
        # gives log(sum(exp(-inf))) = log(0) = -inf, the correct answer.
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        # After the shift the largest exponent is exp(0) = 1: no overflow, and the sum is >= 1
        # (when the max is finite), so log() never sees 0 for ordinary inputs either.
        out = m + torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True))
        out = out.squeeze(dim)
        # Save the OUTPUT as well as x: softmax = exp(x - out) reuses the forward result.
        ctx.save_for_backward(x, out)
        # dim is a python int, not a tensor: store it as a plain attribute.
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        x, out = ctx.saved_tensors
        dim = ctx.dim
        # Put the reduced dim back (size 1) so both broadcast against x.
        # For an all -inf slice this is exp(-inf - -inf) = nan, same as torch.logsumexp's grad.
        softmax = torch.exp(x - out.unsqueeze(dim))
        # One return value per forward input; dim is not differentiable -> None.
        return grad_output.unsqueeze(dim) * softmax, None


class StraightThroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        # Nothing to save: the backward does not depend on x.
        return torch.round(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        # Pretend round() was the identity. The true gradient (0 almost everywhere) would stop
        # anything upstream from learning.
        return grad_output


class SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        # Save the INPUT: the derivative needs x itself, not just x * sigmoid(x).
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)  # torch.sigmoid is stable for large |x| (no exp overflow)
        # Product rule: d/dx [x * s] = s + x * s * (1 - s) = s * (1 + x * (1 - s)).
        return grad_output * s * (1 + x * (1 - s))


def logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return LogSumExp.apply(x, dim)


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return StraightThroughRound.apply(x)


def silu(x: torch.Tensor) -> torch.Tensor:
    return SiLU.apply(x)
