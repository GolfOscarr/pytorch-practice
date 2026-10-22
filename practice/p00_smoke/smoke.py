"""Smoke problem: minimize a function with gradient descent using autograd.

Implement `gradient_descent` below, then run:
    uv run pytest practice/p00_smoke

Rules:
- Use autograd (`.backward()` / `x.grad`) to get the gradient. No torch.optim.
- Don't modify `x0` in place. Return a new tensor.
"""

from collections.abc import Callable

import torch

from common import get_device


def gradient_descent(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    lr: float = 0.1,
    steps: int = 100,
) -> torch.Tensor:
    """Run `steps` iterations of x <- x - lr * grad f(x), starting from x0.

    Args:
        f: scalar-valued function of a tensor.
        x0: starting point (any shape, float dtype).
        lr: learning rate.
        steps: number of update steps.

    Returns:
        The final x, detached, with the same shape/device/dtype as x0.
    """
    raise NotImplementedError("Your code here")


if __name__ == "__main__":
    device = get_device()
    target = torch.tensor([3.0, -2.0], device=device)
    x = gradient_descent(lambda x: ((x - target) ** 2).sum(), torch.zeros(2, device=device))
    print(f"found {x.tolist()}, expected {target.tolist()}")
