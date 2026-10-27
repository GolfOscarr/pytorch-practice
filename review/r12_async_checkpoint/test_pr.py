import asyncio
import contextlib
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from review.r12_async_checkpoint.pr import AsyncCheckpointer

PWNED_ENV = "R12_CHECKPOINT_PAYLOAD_RAN"


def make_model_and_optimizer(seed: int = 0) -> tuple[nn.Linear, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    model = nn.Linear(4, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    return model, optimizer


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad()
    model(torch.ones(2, 4)).pow(2).sum().backward()
    optimizer.step()


def no_sleep(_: float) -> None:
    pass


def saved_steps(directory: Path) -> list[int]:
    return sorted(int(p.stem.removeprefix("step_")) for p in directory.glob("step_*.pt"))


class GatedSave:
    """save_fn that blocks until `gate` is set, then writes for real."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.started = threading.Event()

    def __call__(self, obj: Any, path: Path) -> None:
        self.started.set()
        assert self.gate.wait(timeout=5.0), "test never opened the gate"
        torch.save(obj, path)


# ----------------------------------------------------------------- basics


def test_save_then_load_latest_round_trips_model_and_optimizer(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()
    train_step(model, optimizer)
    with AsyncCheckpointer(tmp_path) as ckpt:
        ckpt.save(7, model, optimizer)
        ckpt.wait()

        fresh_model, fresh_opt = make_model_and_optimizer(seed=1)
        assert ckpt.load_latest(fresh_model, fresh_opt) == 7

    for a, b in zip(model.parameters(), fresh_model.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(
        fresh_opt.state_dict()["state"][0]["exp_avg"], optimizer.state_dict()["state"][0]["exp_avg"]
    )


def test_load_latest_without_checkpoints_returns_none(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()
    with AsyncCheckpointer(tmp_path) as ckpt:
        assert ckpt.latest() is None
        assert ckpt.load_latest(model, optimizer) is None


def test_save_returns_before_the_write_finishes(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()
    gated = GatedSave()
    with AsyncCheckpointer(tmp_path, save_fn=gated) as ckpt:
        future = ckpt.save(1, model, optimizer)
        assert not future.done()
        gated.gate.set()
        assert future.result(timeout=5.0) == tmp_path / "step_1.pt"


# --------------------------------------------------------- snapshot semantics


def test_checkpoint_holds_values_from_save_time_even_if_training_continues(
    tmp_path: Path,
) -> None:
    model, optimizer = make_model_and_optimizer()
    train_step(model, optimizer)
    expected_weight = model.weight.detach().clone()
    expected_exp_avg = optimizer.state_dict()["state"][0]["exp_avg"].clone()

    gated = GatedSave()
    with AsyncCheckpointer(tmp_path, save_fn=gated) as ckpt:
        ckpt.save(1, model, optimizer)
        assert gated.started.wait(timeout=5.0)
        # Training keeps going while the background write is still in flight.
        for _ in range(3):
            train_step(model, optimizer)
        gated.gate.set()
        ckpt.wait()

        restored, restored_opt = make_model_and_optimizer(seed=1)
        ckpt.load_latest(restored, restored_opt)

    torch.testing.assert_close(restored.weight, expected_weight)
    torch.testing.assert_close(restored_opt.state_dict()["state"][0]["exp_avg"], expected_exp_avg)


# ------------------------------------------------------------ crash safety


def test_failed_write_does_not_become_the_latest_checkpoint(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()

    def write_half_then_crash(obj: Any, path: Path) -> None:
        Path(path).write_bytes(b"PK\x03\x04 truncated")
        raise OSError("No space left on device")

    with AsyncCheckpointer(tmp_path) as good:
        good.save(1, model, optimizer)

    ckpt = AsyncCheckpointer(tmp_path, save_fn=write_half_then_crash)
    ckpt.save(2, model, optimizer)
    with contextlib.suppress(OSError):
        ckpt.wait()
    with contextlib.suppress(OSError):
        ckpt.close()

    assert ckpt.latest() == tmp_path / "step_1.pt"
    restored, restored_opt = make_model_and_optimizer(seed=1)
    assert ckpt.load_latest(restored, restored_opt) == 1


def test_wait_reraises_background_save_failure(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()

    def broken_save(obj: Any, path: Path) -> None:
        raise OSError("disk unavailable")

    ckpt = AsyncCheckpointer(tmp_path, save_fn=broken_save)
    ckpt.save(1, model, optimizer)
    with pytest.raises(OSError, match="disk unavailable"):
        ckpt.wait()
    with contextlib.suppress(OSError):
        ckpt.close()


def test_close_flushes_pending_saves(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()

    def slow_save(obj: Any, path: Path) -> None:
        time.sleep(0.2)
        torch.save(obj, path)

    ckpt = AsyncCheckpointer(tmp_path, save_fn=slow_save)
    ckpt.save(1, model, optimizer)
    ckpt.save(2, model, optimizer)
    ckpt.close()

    assert saved_steps(tmp_path) == [1, 2]


# ---------------------------------------------------------------- retention


def test_retention_keeps_the_newest_checkpoints(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()
    with AsyncCheckpointer(tmp_path, keep_last=3) as ckpt:
        for step in range(5, 13):
            ckpt.save(step, model, optimizer)
        ckpt.wait()
        assert saved_steps(tmp_path) == [10, 11, 12]
        assert ckpt.latest() == tmp_path / "step_12.pt"


# ------------------------------------------------------------------ security


def test_load_latest_refuses_to_execute_pickled_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PWNED_ENV, raising=False)

    class Payload:
        def __reduce__(self) -> tuple[Any, tuple[str]]:
            return (exec, (f"import os; os.environ[{PWNED_ENV!r}] = '1'",))

    torch.save(Payload(), tmp_path / "step_99.pt")
    model, optimizer = make_model_and_optimizer()
    with AsyncCheckpointer(tmp_path) as ckpt, pytest.raises(pickle.UnpicklingError):
        ckpt.load_latest(model, optimizer)
    assert PWNED_ENV not in os.environ


# --------------------------------------------------------------------- async


def test_save_async_keeps_the_event_loop_responsive(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()
    heartbeat = threading.Event()
    observed: list[bool] = []

    def save_waiting_for_heartbeat(obj: Any, path: Path) -> None:
        observed.append(heartbeat.wait(timeout=1.0))
        torch.save(obj, path)

    async def beat() -> None:
        heartbeat.set()

    async def main(ckpt: AsyncCheckpointer) -> Path:
        task = asyncio.create_task(beat())
        path = await ckpt.save_async(3, model, optimizer)
        await task
        return path

    with AsyncCheckpointer(tmp_path, save_fn=save_waiting_for_heartbeat) as ckpt:
        path = asyncio.run(main(ckpt))

    assert path == tmp_path / "step_3.pt"
    assert observed == [True], "event loop was blocked while the checkpoint was written"


# -------------------------------------------------------------------- upload


def test_upload_retries_transient_errors_with_capped_backoff(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()
    calls: list[Path] = []
    delays: list[float] = []

    def flaky_upload(path: Path) -> None:
        calls.append(path)
        if len(calls) <= 4:
            raise ConnectionError("connection reset by peer")

    with AsyncCheckpointer(
        tmp_path,
        max_retries=5,
        uploader=flaky_upload,
        backoff_base=1.0,
        backoff_max=4.0,
        sleep=delays.append,
    ) as ckpt:
        ckpt.save(1, model, optimizer)
        ckpt.wait()

    assert calls == [tmp_path / "step_1.pt"] * 5
    assert delays == [1.0, 2.0, 4.0, 4.0]


def test_upload_gives_up_after_max_retries(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()
    calls: list[Path] = []

    def down(path: Path) -> None:
        calls.append(path)
        raise TimeoutError("upload timed out")

    ckpt = AsyncCheckpointer(tmp_path, max_retries=2, uploader=down, sleep=no_sleep)
    future = ckpt.save(1, model, optimizer)
    assert isinstance(future.exception(timeout=5.0), TimeoutError)
    assert len(calls) == 3
    with contextlib.suppress(TimeoutError):
        ckpt.close()


def test_upload_does_not_retry_permanent_errors(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()
    calls: list[Path] = []

    def forbidden(path: Path) -> None:
        calls.append(path)
        raise PermissionError("403: bucket access denied")

    ckpt = AsyncCheckpointer(tmp_path, max_retries=3, uploader=forbidden, sleep=no_sleep)
    future = ckpt.save(1, model, optimizer)
    assert isinstance(future.exception(timeout=5.0), PermissionError)
    assert len(calls) == 1
    with contextlib.suppress(PermissionError):
        ckpt.close()
