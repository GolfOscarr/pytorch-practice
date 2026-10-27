"""Make our trainer multi-GPU with DistributedDataParallel

## Summary
Ports the single-process trainer to `torch.distributed` + DDP so a run can use every GPU on a
node, while staying runnable on laptops (CPU + gloo) for smoke tests. Everything is split into
small functions so each piece can be tested on its own and reused by the upcoming eval job.

## Changes
- `select_backend` / `setup_process_group` / `cleanup_process_group`: process-group lifecycle,
  backend chosen from the device (gloo on CPU, NCCL on accelerators).
- `build_dataloader`: shards the training set with `DistributedSampler`, reshuffled per epoch.
- `build_eval_dataloader`: deterministic (unshuffled) sharding for validation.
- `train_one_epoch` / `evaluate`: return local loss sums + sample counts.
- `reduce_metrics`: all-reduces metric sums so every rank logs the same global number.
- `save_checkpoint` / `load_checkpoint`: model + optimizer + epoch in a single file.
- `main(rank, world_size, cfg, train_set, val_set)`: the per-rank entry; `launch` wraps
  `mp.spawn`.

## Testing
- `test_pr.py` spawns 2 CPU ranks over gloo with a `file://` rendezvous in `tmp_path`, trains a
  few epochs on synthetic regression data and checks losses, checkpoints and teardown.
- Sampler / backend helpers are covered by plain unit tests (no process group needed).
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, TensorDataset

from common import get_device, seed_everything

Batch = tuple[torch.Tensor, ...]


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 8
    lr: float = 0.05
    seed: int = 0
    device: str | None = None
    init_method: str = "env://"
    checkpoint_path: str | None = None


def select_backend(device: torch.device) -> str:
    """Pick the collective backend for ``device``: gloo on CPU, NCCL on accelerators."""
    return "gloo" if device.type == "cpu" else "nccl"


def resolve_device(rank: int, prefer: str | None = None) -> torch.device:
    """The device this rank trains on; CUDA ranks get one GPU each."""
    device = get_device(prefer)
    if device.type == "cuda":
        return torch.device("cuda", rank % torch.cuda.device_count())
    return device


def setup_process_group(
    rank: int, world_size: int, device: torch.device, init_method: str = "env://"
) -> None:
    """Join the default process group with the backend that matches ``device``."""
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend=select_backend(device),
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )


def cleanup_process_group() -> None:
    """Tear down the default process group if this process joined one."""
    if dist.is_initialized():
        dist.destroy_process_group()


def build_model(in_features: int = 4, hidden: int = 16) -> nn.Module:
    """Small MLP regressor used by the trainer."""
    return nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def build_dataloader(
    dataset: Dataset[Batch],
    rank: int,
    world_size: int,
    epoch: int,
    batch_size: int,
    seed: int = 0,
) -> DataLoader[Batch]:
    """Training loader over this rank's 1/world_size shard, reshuffled for ``epoch``."""
    sampler: DistributedSampler[Batch] = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed + rank
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def build_eval_dataloader(
    dataset: Dataset[Batch], rank: int, world_size: int, batch_size: int
) -> DataLoader[Batch]:
    """Validation loader over this rank's shard, in a fixed order."""
    sampler: DistributedSampler[Batch] = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Batch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, int]:
    """One pass over ``loader``. Returns (sum of per-sample losses, number of samples)."""
    model.train()
    loss_sum, seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        seen += x.size(0)
    return loss_sum, seen


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader[Batch], device: torch.device
) -> tuple[float, int]:
    """Returns (sum of per-sample losses, number of samples) over this rank's shard."""
    model.eval()
    loss_sum, seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss_sum += F.mse_loss(model(x), y, reduction="sum").item()
        seen += x.size(0)
    return loss_sum, seen


def reduce_metrics(sums: dict[str, float], count: int, device: torch.device) -> dict[str, float]:
    """Turn per-rank metric sums into global per-sample means, identical on every rank."""
    keys = sorted(sums)
    totals = torch.tensor([sums[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    global_count = count * dist.get_world_size()
    return {k: (totals[i] / global_count).item() for i, k in enumerate(keys)}


def save_checkpoint(
    path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int
) -> None:
    """Persist model + optimizer state for ``epoch``."""
    state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """Load a checkpoint into a plain (unwrapped) model. Returns the next epoch to run."""
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return int(state["epoch"]) + 1


def main(
    rank: int,
    world_size: int,
    cfg: TrainConfig,
    train_set: Dataset[Batch],
    val_set: Dataset[Batch],
) -> list[dict[str, float]]:
    """Per-rank training entry point. Returns the per-epoch metric history."""
    device = resolve_device(rank, cfg.device)
    setup_process_group(rank, world_size, device, cfg.init_method)
    seed_everything(cfg.seed)

    model = build_model().to(device)
    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=cfg.lr, momentum=0.9)
    val_loader = build_eval_dataloader(val_set, rank, world_size, cfg.batch_size)

    history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        train_loader = build_dataloader(
            train_set, rank, world_size, epoch, cfg.batch_size, cfg.seed
        )
        train_sum, n_train = train_one_epoch(ddp_model, train_loader, optimizer, device)
        val_sum, n_val = evaluate(ddp_model, val_loader, device)
        train_loss = reduce_metrics({"loss": train_sum}, n_train, device)["loss"]
        val_loss = reduce_metrics({"loss": val_sum}, n_val, device)["loss"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if cfg.checkpoint_path is not None:
            save_checkpoint(cfg.checkpoint_path, ddp_model, optimizer, epoch)

    cleanup_process_group()
    return history


def launch(
    world_size: int, cfg: TrainConfig, train_set: Dataset[Batch], val_set: Dataset[Batch]
) -> None:
    """Run ``main`` on ``world_size`` local processes."""
    mp.spawn(main, args=(world_size, cfg, train_set, val_set), nprocs=world_size, join=True)


def _synthetic_regression(n: int, seed: int) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5], [3.0]]) + 0.1 * torch.randn(n, 1, generator=g)
    return TensorDataset(x, y)


if __name__ == "__main__":
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    world = max(torch.cuda.device_count(), 2)
    train, val = _synthetic_regression(1024, seed=0), _synthetic_regression(256, seed=1)
    launch(world, TrainConfig(checkpoint_path="ddp_ckpt.pt"), train, val)
