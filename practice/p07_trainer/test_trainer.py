import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from common import get_device, seed_everything
from practice.p07_trainer.trainer import (
    ArrayDataset,
    Callback,
    EarlyStopping,
    HistoryCallback,
    TrainConfig,
    Trainer,
    make_dataloaders,
)

CPU = "cpu"  # exact-comparison tests run on CPU: deterministic kernels, float32 everywhere


def regression_data(n: int = 64, seed: int = 0, scale: float = 1.0) -> ArrayDataset:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=g)
    w = torch.tensor([[1.5], [-2.0], [0.5]])
    y = (x @ w + 0.3 + 0.05 * torch.randn(n, 1, generator=g)) * scale
    return ArrayDataset(x, y)


def mlp() -> nn.Module:
    # Dropout makes the checkpoint test depend on restoring the global RNG state too.
    return nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1))


# ----------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "bad",
    [
        {"lr": 0.0},
        {"lr": -1e-3},
        {"lr": math.nan},
        {"epochs": 0},
        {"batch_size": 0},
        {"grad_clip": 0.0},
        {"weight_decay": -0.1},
        {"warmup_steps": -1},
        {"log_every": 0},
        {"device": "not_a_device"},
    ],
)
def test_config_validation_rejects_bad_values(bad: dict[str, Any]):
    with pytest.raises(ValueError):
        TrainConfig(**bad)


def test_config_frozen_and_dict_round_trip(tmp_path: Path):
    cfg = TrainConfig(lr=3e-4, epochs=5, grad_clip=1.0, checkpoint_dir=tmp_path, device=CPU)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.lr = 1.0  # type: ignore[misc]

    d = cfg.to_dict()
    json.dumps(d)  # must be JSON-serializable (no Path objects)
    assert d["checkpoint_dir"] == str(tmp_path)
    assert TrainConfig.from_dict(d) == cfg
    assert TrainConfig.from_dict(TrainConfig().to_dict()) == TrainConfig()

    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig.from_dict({**d, "learning_rate": 0.1})


# ----------------------------------------------------------------------------- data


def test_array_dataset_basics():
    ds = regression_data(n=10)
    assert len(ds) == 10
    x, y = ds[3]
    assert x.shape == (3,) and y.shape == (1,)
    with pytest.raises(ValueError):
        ArrayDataset(torch.zeros(5, 2), torch.zeros(4))


def test_seeded_dataloader_order_is_reproducible():
    ds = ArrayDataset(torch.arange(32).float().unsqueeze(1), torch.arange(32).float())

    def order(seed: int) -> list[list[int]]:
        loader, _ = make_dataloaders(ds, batch_size=8, seed=seed)
        # Two epochs: the second epoch must also be reproducible (and differ from the first).
        return [y.long().tolist() for _ in range(2) for _, y in loader]

    a, b, c = order(0), order(0), order(1)
    assert a == b
    assert a != c
    assert a[:4] != a[4:]  # epoch 2 is reshuffled
    assert sorted(i for batch in a[:4] for i in batch) == list(range(32))

    _, val = make_dataloaders(ds, ds, batch_size=8, seed=0)
    assert val is not None
    assert [y.long().tolist() for _, y in val][0] == list(range(8))  # val is not shuffled


# ----------------------------------------------------------------------------- trainer


def test_fit_reduces_loss_and_returns_history():
    seed_everything(0)
    device = str(get_device())
    train, val = make_dataloaders(
        regression_data(128, seed=0), regression_data(64, seed=1), batch_size=16, seed=0
    )
    trainer = Trainer(mlp(), TrainConfig(lr=1e-2, epochs=15, device=device), F.mse_loss)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)

    assert val is not None
    before = trainer.evaluate(val)["loss"]
    history = trainer.fit(train, val)
    after = trainer.evaluate(val)["loss"]

    assert len(history) == 15
    assert [h["epoch"] for h in history] == list(range(1, 16))
    assert {"train_loss", "val_loss", "lr"} <= set(history[0])
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert after < 0.2 * before
    assert trainer.epoch == 15 and trainer.global_step == 15 * len(train)
    assert next(trainer.model.parameters()).device.type == torch.device(device).type


def test_evaluate_restores_mode_and_builds_no_graph():
    seed_everything(0)
    model = mlp()
    seen: list[tuple[bool, bool]] = []
    model.register_forward_hook(lambda m, i, out: seen.append((m.training, out.requires_grad)))
    trainer = Trainer(model, TrainConfig(device=CPU), F.mse_loss)
    ds = regression_data(20)
    loader, _ = make_dataloaders(ds, batch_size=8, seed=0)

    model.train()
    result = trainer.evaluate(loader)
    assert model.training  # restored to train mode
    assert seen and all(not training and not req for training, req in seen)

    model.eval()
    trainer.evaluate(loader)
    assert not model.training  # evaluate restores the PREVIOUS mode, it doesn't force train()

    expected = F.mse_loss(model(ds.features), ds.targets).item()  # eval mode, full batch
    assert result["loss"] == pytest.approx(expected, rel=1e-5)


class GradNormProbe(Callback):
    """Records the actual gradient norm after clipping (grads are still set in on_step_end)."""

    def __init__(self) -> None:
        self.post_clip: list[float] = []
        self.reported: list[float] = []

    def on_step_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        self.post_clip.append(
            torch.linalg.vector_norm(torch.cat([g.flatten() for g in grads])).item()
        )
        self.reported.append(metrics["grad_norm"])


def test_grad_clipping_bounds_the_norm():
    seed_everything(0)
    loader, _ = make_dataloaders(regression_data(32, scale=100.0), batch_size=8, seed=0)
    probe = GradNormProbe()
    cfg = TrainConfig(lr=1e-3, epochs=2, grad_clip=0.5, device=CPU)
    Trainer(mlp(), cfg, F.mse_loss, callbacks=[probe]).fit(loader)

    assert len(probe.post_clip) == 8
    assert max(probe.reported) > 10 * 0.5  # pre-clip norms were large: clipping really happened
    assert all(n <= 0.5 * (1 + 1e-4) for n in probe.post_clip)
    assert probe.post_clip[0] == pytest.approx(0.5, rel=1e-4)


class LrProbe(Callback):
    def __init__(self) -> None:
        self.lrs: list[float] = []

    def on_step_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        self.lrs.append(metrics["lr"])


def test_scheduler_linear_warmup_then_cosine():
    seed_everything(0)
    loader, _ = make_dataloaders(regression_data(16), batch_size=4, seed=0)  # 4 steps/epoch
    probe = LrProbe()
    peak, warmup, epochs = 0.1, 4, 3
    cfg = TrainConfig(lr=peak, epochs=epochs, warmup_steps=warmup, device=CPU)
    trainer = Trainer(mlp(), cfg, F.mse_loss, callbacks=[probe])
    trainer.fit(loader)

    total = epochs * 4
    assert len(probe.lrs) == total
    # Linear warmup: lr used at step s is peak * s / warmup (so step 0 uses lr 0).
    assert probe.lrs[:warmup] == pytest.approx([0.0, 0.025, 0.05, 0.075])
    # Then cosine from peak to 0 over the remaining steps.
    for s in range(warmup, total):
        expected = peak * 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total - warmup)))
        assert probe.lrs[s] == pytest.approx(expected), f"step {s}"
    assert probe.lrs[warmup] == pytest.approx(peak)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


def test_early_stopping_stops_on_plateau():
    seed_everything(0)
    train, val = make_dataloaders(regression_data(16), regression_data(16, 1), batch_size=8, seed=0)
    history_cb = HistoryCallback()
    early = EarlyStopping(patience=2, min_delta=1e-3)
    cfg = TrainConfig(lr=1e-9, epochs=20, device=CPU)  # lr ~ 0: val_loss plateaus immediately
    history = Trainer(mlp(), cfg, F.mse_loss, callbacks=[early, history_cb]).fit(train, val)

    # Epoch 1 sets `best`; epochs 2 and 3 fail to improve by min_delta -> stop after epoch 3.
    assert len(history) == 3
    assert history_cb.history == history


def test_early_stopping_logic_directly():
    class FakeTrainer:
        should_stop = False

    fake: Any = FakeTrainer()
    early = EarlyStopping(patience=2, min_delta=0.1, monitor="val_loss")
    early.on_train_start(fake)
    for loss in [1.0, 0.5, 0.45]:  # 0.45 is not a min_delta improvement over 0.5 -> wait=1
        early.on_epoch_end(fake, {"val_loss": loss})
    assert not fake.should_stop
    early.on_epoch_end(fake, {"val_loss": 0.3})  # real improvement resets patience
    early.on_epoch_end(fake, {"val_loss": 0.3})
    assert not fake.should_stop
    early.on_epoch_end(fake, {"val_loss": 0.29})
    assert fake.should_stop
    with pytest.raises(ValueError):
        EarlyStopping(patience=0)


class StopAfter(Callback):
    def __init__(self, epochs: int) -> None:
        self.epochs = epochs

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        if trainer.epoch >= self.epochs:
            trainer.should_stop = True


def test_checkpoint_resume_matches_uninterrupted_run(tmp_path: Path):
    cfg = TrainConfig(lr=1e-2, epochs=4, warmup_steps=3, grad_clip=1.0, seed=7, device=CPU)
    ds = regression_data(40)

    def loader() -> Any:
        return make_dataloaders(ds, batch_size=8, seed=cfg.seed)[0]

    # Uninterrupted reference: 4 epochs straight.
    seed_everything(0)
    straight = Trainer(mlp(), cfg, F.mse_loss)
    straight_hist = straight.fit(loader())

    # Interrupted: 2 epochs (same config, so the same lr horizon), save, then a FRESH trainer.
    seed_everything(0)
    first = Trainer(mlp(), cfg, F.mse_loss, callbacks=[StopAfter(2)])
    first_hist = first.fit(loader())
    assert len(first_hist) == 2
    ckpt = tmp_path / "ckpt" / "run.pt"
    first.save_checkpoint(ckpt)

    seed_everything(123)  # different init and RNG state: everything must come from the file
    resumed = Trainer(mlp(), cfg, F.mse_loss)
    resumed.load_checkpoint(ckpt)
    assert resumed.epoch == 2 and resumed.global_step == 2 * len(loader())
    resumed_hist = resumed.fit(loader())

    assert [h["epoch"] for h in resumed_hist] == [3.0, 4.0]
    for a, b in zip(first_hist + resumed_hist, straight_hist, strict=True):
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-6)
    for p, q in zip(resumed.model.parameters(), straight.model.parameters(), strict=True):
        torch.testing.assert_close(p, q)
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(
        straight.optimizer.param_groups[0]["lr"]
    )


def test_checkpoint_file_contents_and_checkpoint_dir(tmp_path: Path):
    seed_everything(0)
    cfg = TrainConfig(epochs=2, checkpoint_dir=tmp_path / "out", device=CPU)
    loader, _ = make_dataloaders(regression_data(16), batch_size=8, seed=0)
    Trainer(mlp(), cfg, F.mse_loss).fit(loader)

    path = tmp_path / "out" / "last.pt"
    assert path.exists()  # fit() saves to checkpoint_dir / "last.pt" every epoch
    # Must load with weights_only=True: no arbitrary pickled objects in the checkpoint.
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert {"model", "optimizer", "scheduler", "epoch", "global_step", "config", "rng"} <= set(
        state
    )
    assert state["epoch"] == 2 and state["global_step"] == 4
    assert TrainConfig.from_dict(state["config"]) == cfg
    assert not list((tmp_path / "out").glob("*.tmp"))  # no leftover temp files


def test_logging_uses_logging_module_not_print(caplog: pytest.LogCaptureFixture, capsys):
    caplog.set_level(logging.INFO)
    seed_everything(0)
    loader, _ = make_dataloaders(regression_data(32), batch_size=4, seed=0)  # 8 steps/epoch
    Trainer(mlp(), TrainConfig(epochs=2, log_every=3, device=CPU), F.mse_loss).fit(loader)

    records = [r for r in caplog.records if r.name.endswith("p07_trainer.trainer")]
    messages = [r.getMessage() for r in records]
    assert sum(m.startswith("step ") for m in messages) == 16 // 3  # global steps 3, 6, ..., 15
    assert sum(m.startswith("epoch ") for m in messages) == 2
    assert all(r.levelno == logging.INFO for r in records)
    assert capsys.readouterr().out == ""


class Recorder(Callback):
    def __init__(self, name: str, log: list[str]) -> None:
        self.name, self.log = name, log

    def on_train_start(self, trainer: Trainer) -> None:
        self.log.append(f"{self.name}:train_start")

    def on_step_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        self.log.append(f"{self.name}:step{int(metrics['step'])}")

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        self.log.append(f"{self.name}:epoch{int(metrics['epoch'])}")

    def on_train_end(self, trainer: Trainer) -> None:
        self.log.append(f"{self.name}:train_end")


def test_callbacks_called_in_order():
    seed_everything(0)
    log: list[str] = []
    loader, _ = make_dataloaders(regression_data(8), batch_size=4, seed=0)  # 2 steps/epoch
    cbs = [Recorder("a", log), Recorder("b", log)]
    Trainer(mlp(), TrainConfig(epochs=2, device=CPU), F.mse_loss, callbacks=cbs).fit(loader)

    expected = ["train_start", "step1", "step2", "epoch1", "step3", "step4", "epoch2", "train_end"]
    assert log == [f"{who}:{event}" for event in expected for who in "ab"]


def test_amp_uses_bfloat16_autocast_on_cpu():
    seed_everything(0)
    model = mlp()
    dtypes: list[torch.dtype] = []
    model[0].register_forward_hook(lambda m, i, out: dtypes.append(out.dtype))
    loader, _ = make_dataloaders(regression_data(16), batch_size=8, seed=0)
    trainer = Trainer(model, TrainConfig(epochs=1, use_amp=True, device=CPU), F.mse_loss)
    history = trainer.fit(loader)

    assert dtypes and all(d == torch.bfloat16 for d in dtypes)  # matmuls ran in bf16
    assert all(p.dtype == torch.float32 for p in model.parameters())  # master weights stay fp32
    assert math.isfinite(history[0]["train_loss"])
