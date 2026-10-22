"""Reference solution for p00_smoke. Try smoke.py yourself before reading this."""

from collections.abc import Callable

import torch

from common import get_device


def gradient_descent(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    lr: float = 0.1,
    steps: int = 100,
) -> torch.Tensor:
    # detach(): cut x off from whatever graph x0 belongs to.
    # clone():  get new memory, so updating x never touches x0.
    # requires_grad_(True): make x a leaf tensor that autograd tracks.
    x = x0.detach().clone().requires_grad_(True)

    for _ in range(steps):
        loss = f(x)
        loss.backward()  # fills x.grad with df/dx
        assert x.grad is not None  # x.grad is typed Optional; this narrows it for mypy

        # The update must not be recorded in the graph. Without no_grad(),
        # an in-place op on a leaf that requires grad raises a RuntimeError.
        with torch.no_grad():
            x -= lr * x.grad

        # backward() ADDS to .grad, so reset it or gradients pile up across steps.
        x.grad = None

    # Return a plain tensor (requires_grad=False) to the caller.
    return x.detach()


if __name__ == "__main__":
    device = get_device()
    target = torch.tensor([3.0, -2.0], device=device)
    x = gradient_descent(lambda x: ((x - target) ** 2).sum(), torch.zeros(2, device=device))
    print(f"found {x.tolist()}, expected {target.tolist()}")
