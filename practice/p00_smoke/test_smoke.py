import pytest
import torch

from common import get_device
from practice.p00_smoke.smoke import gradient_descent

DEVICE = get_device()


def test_quadratic_1d():
    x = gradient_descent(lambda x: (x - 3.0) ** 2, torch.tensor(0.0, device=DEVICE))
    assert x.item() == pytest.approx(3.0, abs=1e-4)


def test_quadratic_vector():
    target = torch.tensor([1.0, -2.0, 0.5], device=DEVICE)
    x = gradient_descent(lambda x: ((x - target) ** 2).sum(), torch.zeros(3, device=DEVICE))
    torch.testing.assert_close(x, target, atol=1e-4, rtol=0)


def test_returns_same_shape_device_dtype_and_detached():
    x0 = torch.zeros(2, 3, device=DEVICE)
    x = gradient_descent(lambda x: (x**2).sum(), x0, steps=1)
    assert x.shape == x0.shape
    assert x.device == x0.device
    assert x.dtype == x0.dtype
    assert not x.requires_grad


def test_does_not_mutate_input():
    x0 = torch.ones(4, device=DEVICE)
    gradient_descent(lambda x: (x**2).sum(), x0, steps=10)
    torch.testing.assert_close(x0, torch.ones(4, device=DEVICE))


def test_zero_steps_returns_start():
    x0 = torch.tensor([5.0, 7.0], device=DEVICE)
    x = gradient_descent(lambda x: (x**2).sum(), x0, steps=0)
    torch.testing.assert_close(x, x0)


def test_rosenbrock_moves_downhill():
    def rosenbrock(p: torch.Tensor) -> torch.Tensor:
        return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2

    x0 = torch.tensor([-1.0, 1.0], device=DEVICE)
    x = gradient_descent(rosenbrock, x0, lr=1e-3, steps=2000)
    assert rosenbrock(x).item() < 0.1
