"""Sanity check: verifies PyTorch, MPS, autograd and a tiny training loop work."""

import platform
import time

import torch

from common import get_device, seed_everything


def main() -> None:
    seed_everything(0)
    device = get_device()
    print(f"python   {platform.python_version()}")
    print(f"torch    {torch.__version__}")
    mps = torch.backends.mps
    print(f"mps      built={mps.is_built()} available={mps.is_available()}")
    print(f"device   {device}")

    # Autograd check: d/dx (x^2) = 2x
    x = torch.tensor(3.0, requires_grad=True)
    (x**2).backward()
    assert x.grad is not None and x.grad.item() == 6.0
    print("autograd ok")

    # Tiny linear regression on the selected device
    w_true = torch.tensor([[2.0], [-3.0]], device=device)
    X = torch.randn(512, 2, device=device)
    y = X @ w_true + 0.5
    model = torch.nn.Linear(2, 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(200):
        loss = torch.nn.functional.mse_loss(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"train ok  loss={loss.item():.2e}")

    # Quick matmul throughput comparison
    for dev in {torch.device("cpu"), device}:
        a = torch.randn(2048, 2048, device=dev)
        _ = a @ a  # warmup
        if dev.type == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            _ = a @ a
        if dev.type == "mps":
            torch.mps.synchronize()
        print(f"matmul    {dev.type:4s} {(time.perf_counter() - t) / 10 * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
