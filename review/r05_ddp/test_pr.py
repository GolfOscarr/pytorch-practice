"""Tests for the DDP trainer: pure helpers in-process, collectives on 2 spawned CPU/gloo ranks."""

import importlib.util
import inspect
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset

from review.r05_ddp.pr import (
    build_dataloader,
    build_eval_dataloader,
    build_model,
    load_checkpoint,
    select_backend,
)

WORLD_SIZE = 2
EPOCHS = 3
N_TRAIN, N_VAL = 32, 9
SPAWN_TIMEOUT_S = 60.0

# Spawned ranks start a fresh interpreter, so they re-load the code under test from the exact
# file this process imported (pr.py, or solution/fixed.py under --solution).
TARGET_FILE = inspect.getfile(select_backend)


# ----------------------------------------------------------------------------- helpers


def _regression_data(n: int, seed: int) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5], [3.0]]) + 0.1 * torch.randn(n, 1, generator=g)
    return TensorDataset(x, y)


def _index_dataset(n: int) -> TensorDataset:
    return TensorDataset(torch.arange(n))


def _indices(loader: Any) -> list[int]:
    return [int(i) for (batch,) in loader for i in batch]


class _BrokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A dataset whose samples fail to load, to simulate a crash mid-epoch."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(f"corrupt sample {idx}")


def _load_target(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("r05_ddp_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_result(workdir: str, name: str, rank: int, payload: dict[str, Any]) -> None:
    (Path(workdir) / f"{name}_rank{rank}.json").write_text(json.dumps(payload))


def _read_results(workdir: Path, name: str) -> list[dict[str, Any]]:
    return [json.loads((workdir / f"{name}_rank{r}.json").read_text()) for r in range(WORLD_SIZE)]


def _run_spawned(fn: Callable[..., None], *args: Any) -> None:
    ctx = mp.spawn(fn, args=args, nprocs=WORLD_SIZE, join=False)
    assert ctx is not None
    deadline = time.monotonic() + SPAWN_TIMEOUT_S
    while not ctx.join(timeout=1.0):
        if time.monotonic() > deadline:
            for proc in ctx.processes:
                proc.kill()
            pytest.fail(f"spawned ranks did not finish within {SPAWN_TIMEOUT_S}s (deadlock?)")


# ----------------------------------------------------------------------------- spawned workers


def _train_worker(rank: int, world_size: int, target: str, workdir: str) -> None:
    pr = _load_target(target)
    cfg = pr.TrainConfig(
        epochs=EPOCHS,
        batch_size=8,
        lr=0.05,
        device="cpu",
        init_method=f"file://{workdir}/train_rendezvous",
        checkpoint_path=f"{workdir}/ckpt.pt",
    )
    with mock.patch.object(torch, "save", wraps=torch.save) as save_spy:
        history = pr.main(
            rank, world_size, cfg, _regression_data(N_TRAIN, 0), _regression_data(N_VAL, 1)
        )
    _write_result(
        workdir,
        "train",
        rank,
        {
            "history": history,
            "save_calls": save_spy.call_count,
            "pg_alive": dist.is_initialized(),
        },
    )


def _collectives_worker(rank: int, world_size: int, target: str, workdir: str) -> None:
    pr = _load_target(target)
    cpu = torch.device("cpu")

    pr.setup_process_group(rank, world_size, cpu, f"file://{workdir}/reduce_rendezvous")
    backend = dist.get_backend()
    # Rank 0 saw 2 samples, rank 1 saw 6: global means are (10+2)/8 and (2+4)/8.
    sums, count = ({"loss": 10.0, "acc": 2.0}, 2) if rank == 0 else ({"loss": 2.0, "acc": 4.0}, 6)
    reduced = pr.reduce_metrics(sums, count, cpu)
    pr.cleanup_process_group()

    cfg = pr.TrainConfig(
        epochs=1, batch_size=4, device="cpu", init_method=f"file://{workdir}/crash_rendezvous"
    )
    error = None
    try:
        pr.main(rank, world_size, cfg, _BrokenDataset(16), _regression_data(4, 1))
    except RuntimeError as exc:
        error = str(exc)
    _write_result(
        workdir,
        "collectives",
        rank,
        {
            "backend": backend,
            "reduced": reduced,
            "error": error,
            "pg_alive_after_error": dist.is_initialized(),
        },
    )


@pytest.fixture(scope="module")
def train_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict[str, Any]]]:
    workdir = tmp_path_factory.mktemp("ddp_train")
    _run_spawned(_train_worker, WORLD_SIZE, TARGET_FILE, str(workdir))
    return workdir, _read_results(workdir, "train")


@pytest.fixture(scope="module")
def collectives_run(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    workdir = tmp_path_factory.mktemp("ddp_collectives")
    _run_spawned(_collectives_worker, WORLD_SIZE, TARGET_FILE, str(workdir))
    return _read_results(workdir, "collectives")


# ----------------------------------------------------------------------------- backend


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cpu", "gloo"), ("mps", "gloo"), ("cuda", "nccl"), ("cuda:1", "nccl")],
)
def test_backend_matches_device(device: str, expected: str) -> None:
    assert select_backend(torch.device(device)) == expected


def test_spawned_ranks_join_a_gloo_group_on_cpu(collectives_run: list[dict[str, Any]]) -> None:
    assert [r["backend"] for r in collectives_run] == ["gloo"] * WORLD_SIZE


# ----------------------------------------------------------------------------- data sharding


@pytest.mark.parametrize("epoch", [0, 1, 2])
def test_train_shards_partition_the_dataset(epoch: int) -> None:
    ds = _index_dataset(16)
    shards = [
        _indices(build_dataloader(ds, r, WORLD_SIZE, epoch, batch_size=4, seed=7))
        for r in range(WORLD_SIZE)
    ]
    assert all(len(s) == 8 for s in shards)
    assert sorted(shards[0] + shards[1]) == list(range(16))


def test_train_order_is_deterministic_for_a_given_epoch() -> None:
    ds = _index_dataset(16)
    first = _indices(build_dataloader(ds, 0, WORLD_SIZE, epoch=3, batch_size=4, seed=7))
    again = _indices(build_dataloader(ds, 0, WORLD_SIZE, epoch=3, batch_size=4, seed=7))
    assert first == again


def test_train_order_changes_between_epochs() -> None:
    ds = _index_dataset(16)
    orders = [
        _indices(build_dataloader(ds, 0, WORLD_SIZE, epoch=e, batch_size=4, seed=7))
        for e in range(3)
    ]
    assert orders[0] != orders[1] != orders[2]


def test_eval_shards_cover_every_sample_exactly_once() -> None:
    ds = _index_dataset(N_VAL)
    seen = [
        i for r in range(WORLD_SIZE) for i in _indices(build_eval_dataloader(ds, r, WORLD_SIZE, 4))
    ]
    assert sorted(seen) == list(range(N_VAL))


# ----------------------------------------------------------------------------- metric reduction


def test_reduce_metrics_weights_each_rank_by_its_sample_count(
    collectives_run: list[dict[str, Any]],
) -> None:
    for result in collectives_run:
        assert result["reduced"] == pytest.approx({"loss": 1.5, "acc": 0.75})


def test_training_loss_decreases(train_run: tuple[Path, list[dict[str, Any]]]) -> None:
    _, results = train_run
    history = results[0]["history"]
    assert len(history) == EPOCHS
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_every_rank_reports_the_same_metrics(train_run: tuple[Path, list[dict[str, Any]]]) -> None:
    _, results = train_run
    assert results[0]["history"] == pytest.approx(results[1]["history"])


# ----------------------------------------------------------------------------- checkpoints


def test_checkpoint_loads_into_a_plain_model(train_run: tuple[Path, list[dict[str, Any]]]) -> None:
    workdir, _ = train_run
    model = build_model()
    next_epoch = load_checkpoint(str(workdir / "ckpt.pt"), model)
    assert next_epoch == EPOCHS


def test_reported_val_loss_equals_single_process_loss_on_full_val_set(
    train_run: tuple[Path, list[dict[str, Any]]],
) -> None:
    workdir, results = train_run
    model = build_model()
    load_checkpoint(str(workdir / "ckpt.pt"), model)
    x, y = _regression_data(N_VAL, 1).tensors
    with torch.no_grad():
        expected = F.mse_loss(model(x), y).item()
    for result in results:
        assert result["history"][-1]["val_loss"] == pytest.approx(expected, rel=1e-5)


def test_only_rank_zero_writes_checkpoints(train_run: tuple[Path, list[dict[str, Any]]]) -> None:
    _, results = train_run
    assert results[0]["save_calls"] >= 1
    assert all(r["save_calls"] == 0 for r in results[1:])


# ----------------------------------------------------------------------------- teardown


def test_process_group_is_destroyed_after_training(
    train_run: tuple[Path, list[dict[str, Any]]],
) -> None:
    _, results = train_run
    assert not any(r["pg_alive"] for r in results)


def test_training_error_propagates_and_process_group_is_destroyed(
    collectives_run: list[dict[str, Any]],
) -> None:
    for result in collectives_run:
        assert result["error"] is not None and "corrupt sample" in result["error"]
        assert result["pg_alive_after_error"] is False
