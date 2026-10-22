import torch

from common import get_device


def test_device_runs_tensor_ops():
    device = get_device()
    t = torch.ones(3, device=device) * 2
    assert t.sum().item() == 6.0
