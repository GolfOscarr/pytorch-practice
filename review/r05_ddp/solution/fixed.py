"""Reference fix for review/r05_ddp/pr.py: DDP trainer with all planted bugs fixed.

Same public API as pr.py. `# FIX(Bn)` marks the lines that correspond to the answer key in
REVIEW.md; `# IMPROVED` marks production improvements that are not bug fixes.
"""

import logging
import os
from collections.abc import Sized
from dataclasses import dataclass
from typing import cast

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset, TensorDataset

from common import get_device, seed_everything

log = logging.getLogger(__name__)

Batch = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    batch_size: int = 8
    lr: float = 0.05
    seed: int = 0
    device: str | None = None
    init_method: str = "env://"
    checkpoint_path: str | None = None

    def __post_init__(self) -> None:  # IMPROVED: fail fast on nonsense configs
        if self.epochs < 1 or self.batch_size < 1 or self.lr <= 0:
            raise ValueError(f"invalid TrainConfig: {self}")


def select_backend(device: torch.device) -> str:
    """Pick the collective backend for ``device``: NCCL for CUDA, gloo for everything else."""
    # FIX(B1): NCCL is CUDA-only; everything else uses gloo. (MPS tensors aren't supported by
    # any backend, so resolve_device keeps non-CUDA ranks on CPU.)
    return "nccl" if device.type == "cuda" else "gloo"


def resolve_device(rank: int, prefer: str | None = None) -> torch.device:
    """The device this rank trains on; CUDA ranks get one GPU each.

    Off CUDA, ranks train on CPU: no c10d backend (gloo included) supports MPS tensors.
    """
    device = get_device(prefer)
    if device.type == "cuda":
        return torch.device("cuda", rank % torch.cuda.device_count())
    if device.type != "cpu":
        log.warning("%s has no distributed backend; rank %d trains on CPU", device, rank)
    return torch.device("cpu")


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
    # FIX(B3): the seed must be identical on every rank. DistributedSampler shuffles the full
    # index list with this seed and then each rank takes a strided slice; different seeds mean
    # different permutations, so shards overlap and some samples are never trained on.
    sampler: DistributedSampler[Batch] = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    )
    sampler.set_epoch(epoch)  # FIX(B2): without this every epoch replays the same order
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def build_eval_dataloader(
    dataset: Dataset[Batch], rank: int, world_size: int, batch_size: int
) -> DataLoader[Batch]:
    """Validation loader over this rank's shard, in a fixed order, each sample exactly once."""
    # FIX(B5): DistributedSampler pads the index list (repeating samples) so every rank gets the
    # same length. That is required for training (equal step counts) but double-counts samples
    # in a reduced metric. A strided, unpadded Subset gives uneven but exact shards.
    n = len(cast(Sized, dataset))
    shard = Subset(dataset, range(rank, n, world_size))
    return DataLoader(shard, batch_size=batch_size, shuffle=False)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Batch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, int]:
    """One pass over ``loader``. Returns (sum of per-sample losses, number of samples)."""
    model.train()
    # IMPROVED: accumulate on device; one host sync per epoch instead of one `.item()` per step.
    loss_sum = torch.zeros((), device=device)
    seen = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach() * x.size(0)
        seen += x.size(0)
    return loss_sum.item(), seen


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader[Batch], device: torch.device
) -> tuple[float, int]:
    """Returns (sum of per-sample losses, number of samples) over this rank's shard.

    Pass the *unwrapped* module: eval shards are uneven, and a DDP forward may issue
    collectives (buffer broadcast) that would deadlock when ranks run different step counts.
    """
    model.eval()
    loss_sum = torch.zeros((), device=device)
    seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss_sum += F.mse_loss(model(x), y, reduction="sum")
        seen += x.size(0)
    return loss_sum.item(), seen


def reduce_metrics(sums: dict[str, float], count: int, device: torch.device) -> dict[str, float]:
    """Turn per-rank metric sums into global per-sample means, identical on every rank."""
    keys = sorted(sums)
    # FIX(B4): reduce the sample count together with the sums (one collective). Ranks do not
    # see the same number of samples (uneven eval shards, last partial batch), so
    # `count * world_size` is not the global count.
    packed = torch.tensor([*(sums[k] for k in keys), count], dtype=torch.float64, device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    global_count = packed[-1].item()
    if global_count == 0:  # IMPROVED: avoid silent NaN on an empty split
        raise ValueError("reduce_metrics called with zero samples on every rank")
    return {k: (packed[i] / global_count).item() for i, k in enumerate(keys)}


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(
    path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int
) -> None:
    """Persist model + optimizer state for ``epoch``. Call on one rank only (see ``main``)."""
    # FIX(B6): save the inner module's state_dict. DDP's own keys carry a `module.` prefix that
    # makes the file unloadable into a plain model (inference, eval jobs, non-DDP resume).
    state = {
        "epoch": epoch,
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    # IMPROVED: atomic write, so a crash mid-save never leaves a truncated "latest" checkpoint.
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """Load a checkpoint into a model (plain or DDP-wrapped). Returns the next epoch to run."""
    state = torch.load(path, map_location=map_location, weights_only=True)
    _unwrap(model).load_state_dict(state["model"])
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
    try:  # FIX(B8): always tear the group down, also when training raises
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
            val_sum, n_val = evaluate(model, val_loader, device)  # unwrapped: uneven shards
            train_loss = reduce_metrics({"loss": train_sum}, n_train, device)["loss"]
            val_loss = reduce_metrics({"loss": val_sum}, n_val, device)["loss"]
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if rank == 0:  # IMPROVED: log once, not world_size times
                log.info("epoch %d train %.4f val %.4f", epoch, train_loss, val_loss)
            if cfg.checkpoint_path is not None:
                # FIX(B7): one writer. Concurrent torch.save to one path can interleave and
                # corrupt the file; the barrier keeps other ranks from racing ahead (e.g. to
                # read it) before it is complete.
                if rank == 0:
                    save_checkpoint(cfg.checkpoint_path, ddp_model, optimizer, epoch)
                dist.barrier()
        return history
    finally:
        cleanup_process_group()


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
    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    world = max(torch.cuda.device_count(), 2)
    train, val = _synthetic_regression(1024, seed=0), _synthetic_regression(256, seed=1)
    launch(world, TrainConfig(checkpoint_path="ddp_ckpt.pt"), train, val)
