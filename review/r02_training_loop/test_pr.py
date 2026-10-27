import weakref
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from review.r02_training_loop.pr import (
    EpochMetrics,
    TrainConfig,
    evaluate,
    fit,
    train_one_epoch,
)

# Everything runs on CPU in float64 so the loop under test and the hand-written reference
# produce bit-for-bit comparable parameters.
DTYPE = torch.float64
RTOL, ATOL = 1e-10, 1e-12
# Reported losses are metrics accumulated in a float32 running sum, so compare them loosely.
LOSS_RTOL = 1e-6


def _make_model(dropout: float = 0.0) -> nn.Sequential:
    torch.manual_seed(0)
    layers: list[nn.Module] = [nn.Linear(3, 8), nn.Tanh()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(8, 1))
    return nn.Sequential(*layers).to(DTYPE)


def _make_data(n: int = 10) -> TensorDataset:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(n, 3, generator=g, dtype=DTYPE)
    y = torch.randn(n, 1, generator=g, dtype=DTYPE)
    return TensorDataset(x, y)


def _batches(n: int = 10, batch_size: int = 2) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return list(DataLoader(_make_data(n), batch_size=batch_size, shuffle=False))


def _reference_epoch(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    accumulation_steps: int,
    max_grad_norm: float | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    """Textbook accumulation loop: one zero_grad/clip/step/scheduler-step per window."""
    model.train()
    total, count = 0.0, 0
    for start in range(0, len(batches), accumulation_steps):
        optimizer.zero_grad()
        for x, y in batches[start : start + accumulation_steps]:
            loss = F.mse_loss(model(x), y)
            (loss / accumulation_steps).backward()
            total += loss.item() * len(x)
            count += len(x)
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return total / count


def _assert_same_params(a: nn.Module, b: nn.Module) -> None:
    for (name, pa), pb in zip(a.named_parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb, rtol=RTOL, atol=ATOL, msg=f"parameter {name} differs")


class CountingSGD(torch.optim.SGD):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.num_steps = 0

    def step(self, closure: Any = None) -> Any:
        self.num_steps += 1
        return super().step(closure)


# --------------------------------------------------------------------------- train_one_epoch


@pytest.mark.parametrize("accumulation_steps", [1, 2, 3])
def test_train_epoch_matches_reference_loop(accumulation_steps: int) -> None:
    batches = _batches(10, 2)  # 5 micro-batches: k=2 and k=3 leave a partial last window
    model, ref = _make_model(), _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.1, momentum=0.9)

    for _ in range(3):
        loss = train_one_epoch(
            model, batches, opt, F.mse_loss, accumulation_steps=accumulation_steps
        )
        ref_loss = _reference_epoch(ref, batches, ref_opt, accumulation_steps)
        assert loss == pytest.approx(ref_loss, rel=LOSS_RTOL)
    _assert_same_params(model, ref)


def test_accumulated_step_equals_one_big_batch_step() -> None:
    model, big = _make_model(), _make_model()
    data = _make_data(8)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    big_opt = torch.optim.SGD(big.parameters(), lr=0.5)

    train_one_epoch(model, DataLoader(data, batch_size=2), opt, F.mse_loss, accumulation_steps=4)
    train_one_epoch(big, DataLoader(data, batch_size=8), big_opt, F.mse_loss)
    _assert_same_params(model, big)


@pytest.mark.parametrize(("num_batches", "k", "expected"), [(4, 2, 2), (5, 2, 3), (5, 3, 2)])
def test_every_micro_batch_contributes_to_an_optimizer_step(
    num_batches: int, k: int, expected: int
) -> None:
    model = _make_model()
    opt = CountingSGD(model.parameters(), lr=0.1)
    train_one_epoch(model, _batches(2 * num_batches, 2), opt, F.mse_loss, accumulation_steps=k)
    assert opt.num_steps == expected


def test_gradient_clipping_matches_reference_and_bounds_update() -> None:
    batches = _batches(10, 2)
    lr, max_norm = 0.1, 1e-2
    model, ref = _make_model(), _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=lr)

    before = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    train_one_epoch(model, batches[:1], opt, F.mse_loss, max_grad_norm=max_norm)
    after = torch.cat([p.detach().flatten() for p in model.parameters()])
    assert (after - before).norm() <= lr * max_norm * (1 + 1e-9)

    _reference_epoch(ref, batches[:1], ref_opt, 1, max_grad_norm=max_norm)
    train_one_epoch(model, batches[1:], opt, F.mse_loss, max_grad_norm=max_norm)
    _reference_epoch(ref, batches[1:], ref_opt, 1, max_grad_norm=max_norm)
    _assert_same_params(model, ref)


def test_scheduler_advances_once_per_optimizer_step() -> None:
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (1 + s))

    train_one_epoch(model, _batches(8, 2), opt, F.mse_loss, accumulation_steps=2, scheduler=sched)
    assert sched.last_epoch == 2  # 4 micro-batches / 2 = 2 optimizer steps
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0 / 3)


def test_train_with_scheduler_and_accumulation_matches_reference() -> None:
    batches = _batches(10, 2)
    model, ref = _make_model(), _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.2)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.2)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    ref_sched = torch.optim.lr_scheduler.StepLR(ref_opt, step_size=1, gamma=0.5)

    train_one_epoch(model, batches, opt, F.mse_loss, accumulation_steps=2, scheduler=sched)
    _reference_epoch(ref, batches, ref_opt, 2, scheduler=ref_sched)
    _assert_same_params(model, ref)
    assert opt.param_groups[0]["lr"] == pytest.approx(ref_opt.param_groups[0]["lr"])


def test_train_epoch_returns_python_float() -> None:
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = train_one_epoch(model, _batches(), opt, F.mse_loss)
    assert type(loss) is float


class _Tag(torch.autograd.Function):
    """Identity op whose backward node keeps ``token`` alive for as long as the node lives."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, token: object) -> torch.Tensor:
        ctx.token = token
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad, None


class _Token:
    pass


def test_epoch_does_not_keep_autograd_graphs_of_earlier_batches_alive() -> None:
    refs: list[weakref.ref[_Token]] = []
    max_alive = 0

    def tracking_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        nonlocal max_alive
        max_alive = max(max_alive, sum(r() is not None for r in refs))
        token = _Token()
        refs.append(weakref.ref(token))
        return F.mse_loss(_Tag.apply(pred, token), target)

    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    train_one_epoch(model, _batches(16, 2), opt, tracking_loss)
    # At most the immediately preceding batch's graph may still be referenced by a local.
    assert max_alive <= 1, f"{max_alive} earlier batch graphs still alive mid-epoch"


# --------------------------------------------------------------------------------- evaluate


def test_evaluate_is_sample_weighted_mean_with_ragged_last_batch() -> None:
    data = _make_data(10)
    model = _make_model()
    x, y = data.tensors
    with torch.no_grad():
        expected = F.mse_loss(model(x), y).item()
    got = evaluate(model, DataLoader(data, batch_size=4), F.mse_loss)  # batches of 4, 4, 2
    assert got == pytest.approx(expected, rel=LOSS_RTOL)


def test_evaluate_disables_dropout() -> None:
    data = _make_data(10)
    model = _make_model(dropout=0.5)
    x, y = data.tensors
    model.eval()
    with torch.no_grad():
        expected = F.mse_loss(model(x), y).item()
    model.train()

    loader = DataLoader(data, batch_size=10)
    first = evaluate(model, loader, F.mse_loss)
    second = evaluate(model, loader, F.mse_loss)
    assert first == pytest.approx(expected, rel=LOSS_RTOL)
    assert second == pytest.approx(expected, rel=LOSS_RTOL)


def test_evaluate_does_not_update_batchnorm_statistics() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 1)).to(DTYPE)
    bn = model[1]
    assert isinstance(bn, nn.BatchNorm1d)
    assert bn.running_mean is not None and bn.running_var is not None
    mean_before, var_before = bn.running_mean.clone(), bn.running_var.clone()

    evaluate(model, DataLoader(_make_data(10), batch_size=5), F.mse_loss)
    torch.testing.assert_close(bn.running_mean, mean_before, rtol=0, atol=0)
    torch.testing.assert_close(bn.running_var, var_before, rtol=0, atol=0)


def test_evaluate_does_not_track_gradients() -> None:
    model = _make_model()
    loss = evaluate(model, DataLoader(_make_data(10), batch_size=4), F.mse_loss)
    assert type(loss) is float
    assert all(p.grad is None for p in model.parameters())


# -------------------------------------------------------------------------------------- fit


def test_fit_returns_one_record_per_epoch() -> None:
    data = _make_data(10)
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    history = fit(
        model,
        DataLoader(data, batch_size=2),
        DataLoader(data, batch_size=4),
        opt,
        F.mse_loss,
        TrainConfig(epochs=3),
    )
    assert [m.epoch for m in history] == [0, 1, 2]
    assert all(isinstance(m, EpochMetrics) for m in history)


def test_fit_matches_reference_train_then_validate() -> None:
    # Plain lists, built before seeding: iterating a DataLoader consumes the global RNG, which
    # would desynchronise the dropout masks of the loop under test and the reference.
    train_batches = _batches(10, 2)
    val_batches = _batches(7, 3)
    val_data = _make_data(7)
    model, ref = _make_model(dropout=0.3), _make_model(dropout=0.3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    ref_sched = torch.optim.lr_scheduler.StepLR(ref_opt, step_size=2, gamma=0.5)
    config = TrainConfig(epochs=2, accumulation_steps=2, max_grad_norm=0.5)

    torch.manual_seed(123)
    history = fit(
        model,
        train_batches,
        val_batches,
        opt,
        F.mse_loss,
        config,
        scheduler=sched,
    )

    torch.manual_seed(123)
    x_val, y_val = val_data.tensors
    for epoch in range(config.epochs):
        ref_train = _reference_epoch(ref, train_batches, ref_opt, 2, 0.5, ref_sched)
        ref.eval()
        with torch.no_grad():
            ref_val = F.mse_loss(ref(x_val), y_val).item()
        ref.train()
        m = history[epoch]
        assert m.train_loss == pytest.approx(ref_train, rel=LOSS_RTOL)
        assert m.val_loss == pytest.approx(ref_val, rel=LOSS_RTOL)
        assert m.lr == pytest.approx(ref_opt.param_groups[0]["lr"])
    _assert_same_params(model, ref)
