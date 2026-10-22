import copy
import io
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

from common import get_device
from practice.p01_optimizers.optimizers import SGD, Adam, AdamW

# Equivalence tests run on CPU in float64 so tiny reordering differences (ulp level) between
# our math and torch's kernels cannot accumulate into a false failure over many steps.
DTYPE = torch.float64
RTOL, ATOL = 1e-9, 1e-11
STEPS = 60

OURS_TO_TORCH: dict[type, Callable[..., torch.optim.Optimizer]] = {
    SGD: torch.optim.SGD,
    Adam: torch.optim.Adam,
    AdamW: torch.optim.AdamW,
}

EQUIVALENCE_CASES = [
    pytest.param(SGD, {"lr": 0.1}, id="sgd-plain"),
    pytest.param(SGD, {"lr": 0.05, "momentum": 0.9}, id="sgd-momentum"),
    pytest.param(SGD, {"lr": 0.05, "momentum": 0.9, "dampening": 0.3}, id="sgd-dampening"),
    pytest.param(SGD, {"lr": 0.05, "momentum": 0.9, "nesterov": True}, id="sgd-nesterov"),
    pytest.param(SGD, {"lr": 0.05, "momentum": 0.9, "weight_decay": 0.01}, id="sgd-wd"),
    pytest.param(Adam, {}, id="adam-defaults"),
    pytest.param(Adam, {"lr": 0.01, "betas": (0.8, 0.99), "eps": 1e-6}, id="adam-custom"),
    pytest.param(Adam, {"lr": 0.01, "weight_decay": 0.1}, id="adam-l2"),
    pytest.param(AdamW, {}, id="adamw-defaults"),
    pytest.param(AdamW, {"lr": 0.01, "weight_decay": 0.1}, id="adamw-decay"),
]

ALL_OPTIMIZERS = [
    pytest.param(SGD, {"momentum": 0.9}, id="sgd"),
    pytest.param(Adam, {"weight_decay": 0.01}, id="adam"),
    pytest.param(AdamW, {}, id="adamw"),
]


def _make_model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1)).to(DTYPE)


def _make_data() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(32, 4, generator=g, dtype=DTYPE)
    y = torch.randn(32, 1, generator=g, dtype=DTYPE)
    return x, y


def _train_step(model: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor):
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()


def _assert_same_params(a: nn.Module, b: nn.Module, step: int) -> None:
    for (name, pa), pb in zip(a.named_parameters(), b.parameters(), strict=True):

        def msg(m: str, name: str = name) -> str:
            return f"{name} diverged at step {step}: {m}"

        torch.testing.assert_close(pa, pb, rtol=RTOL, atol=ATOL, msg=msg)


@pytest.mark.parametrize(("cls", "kwargs"), EQUIVALENCE_CASES)
def test_matches_torch_step_by_step(cls: type, kwargs: dict[str, Any]):
    ours_model = _make_model()
    ref_model = copy.deepcopy(ours_model)
    ours = cls(ours_model.parameters(), **kwargs)
    ref = OURS_TO_TORCH[cls](ref_model.parameters(), **kwargs)
    x, y = _make_data()

    for step in range(STEPS):
        _train_step(ours_model, ours, x, y)
        _train_step(ref_model, ref, x, y)
        _assert_same_params(ours_model, ref_model, step)


@pytest.mark.parametrize(("cls", "kwargs"), ALL_OPTIMIZERS)
def test_param_groups_with_different_hyperparams(cls: type, kwargs: dict[str, Any]):
    ours_model = _make_model()
    ref_model = copy.deepcopy(ours_model)

    def groups(model: nn.Sequential) -> list[dict[str, Any]]:
        # Group 0 overrides lr; group 1 inherits every default.
        return [
            {"params": model[0].parameters(), "lr": 0.1},
            {"params": model[2].parameters()},
        ]

    ours = cls(groups(ours_model), lr=0.01, **kwargs)
    ref = OURS_TO_TORCH[cls](groups(ref_model), lr=0.01, **kwargs)
    assert [g["lr"] for g in ours.param_groups] == [0.1, 0.01]
    x, y = _make_data()

    for step in range(STEPS):
        _train_step(ours_model, ours, x, y)
        _train_step(ref_model, ref, x, y)
        _assert_same_params(ours_model, ref_model, step)


@pytest.mark.parametrize(("cls", "kwargs"), ALL_OPTIMIZERS)
def test_lr_scheduler_changes_are_respected(cls: type, kwargs: dict[str, Any]):
    ours_model = _make_model()
    ref_model = copy.deepcopy(ours_model)
    ours = cls(ours_model.parameters(), lr=0.05, **kwargs)
    ref = OURS_TO_TORCH[cls](ref_model.parameters(), lr=0.05, **kwargs)
    ours_sched = torch.optim.lr_scheduler.StepLR(ours, step_size=5, gamma=0.3)
    ref_sched = torch.optim.lr_scheduler.StepLR(ref, step_size=5, gamma=0.3)
    x, y = _make_data()

    for step in range(30):
        _train_step(ours_model, ours, x, y)
        _train_step(ref_model, ref, x, y)
        ours_sched.step()
        ref_sched.step()
        _assert_same_params(ours_model, ref_model, step)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        pytest.param(SGD, {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.1}, id="sgd"),
        pytest.param(Adam, {"lr": 0.1, "weight_decay": 0.1}, id="adam"),
        pytest.param(AdamW, {"lr": 0.1, "weight_decay": 0.1}, id="adamw"),
    ],
)
def test_params_without_grad_are_skipped(cls: type, kwargs: dict[str, Any]):
    used = nn.Parameter(torch.ones(3, dtype=DTYPE))
    unused = nn.Parameter(torch.full((3,), 2.0, dtype=DTYPE))  # requires grad, never in the loss
    opt = cls([used, unused], **kwargs)

    for _ in range(5):
        opt.zero_grad()
        (used**2).sum().backward()
        assert unused.grad is None
        opt.step()

    # Weight decay must not touch it either, and it gets no optimizer state.
    torch.testing.assert_close(unused, torch.full((3,), 2.0, dtype=DTYPE), rtol=0, atol=0)
    assert not torch.equal(used, torch.ones(3, dtype=DTYPE))
    saved_state = opt.state_dict()["state"]
    assert 0 in saved_state
    assert 1 not in saved_state


@pytest.mark.parametrize(("cls", "kwargs"), ALL_OPTIMIZERS)
def test_intermittent_grads_match_torch(cls: type, kwargs: dict[str, Any]):
    """A branch that only joins the loss every 3rd step has grad None the rest of the time.

    Its momentum / Adam moments / step count must only advance on the steps it is used.
    """
    torch.manual_seed(0)
    ours_model = nn.ModuleList([nn.Linear(4, 1), nn.Linear(4, 1)]).to(DTYPE)
    ref_model = copy.deepcopy(ours_model)
    ours = cls(ours_model.parameters(), lr=0.05, **kwargs)
    ref = OURS_TO_TORCH[cls](ref_model.parameters(), lr=0.05, **kwargs)
    x, y = _make_data()

    for step in range(STEPS):
        for model, opt in ((ours_model, ours), (ref_model, ref)):
            opt.zero_grad()  # set_to_none=True by default, so the idle branch has grad None
            out = model[0](x)
            if step % 3 == 0:
                out = out + model[1](x)
            nn.functional.mse_loss(out, y).backward()
            opt.step()
        _assert_same_params(ours_model, ref_model, step)


@pytest.mark.parametrize(("cls", "kwargs"), ALL_OPTIMIZERS)
def test_state_dict_round_trip_continues_identically(cls: type, kwargs: dict[str, Any]):
    model = _make_model()
    opt = cls(model.parameters(), lr=0.05, **kwargs)
    x, y = _make_data()
    for _ in range(10):
        _train_step(model, opt, x, y)

    # Serialize like a real checkpoint, then restore into fresh objects.
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)

    restored_model = _make_model()
    restored_opt = cls(restored_model.parameters(), lr=0.05, **kwargs)
    restored_model.load_state_dict(checkpoint["model"])
    restored_opt.load_state_dict(checkpoint["opt"])

    for step in range(20):
        _train_step(model, opt, x, y)
        _train_step(restored_model, restored_opt, x, y)
        _assert_same_params(model, restored_model, step)


@pytest.mark.parametrize(("cls", "kwargs"), ALL_OPTIMIZERS)
def test_closure_support(cls: type, kwargs: dict[str, Any]):
    model = _make_model()
    opt = cls(model.parameters(), lr=0.05, **kwargs)
    x, y = _make_data()
    before = [p.detach().clone() for p in model.parameters()]

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()  # needs grad mode, even though step() runs under no_grad
        return loss

    loss = opt.step(closure)  # type: ignore[arg-type]
    assert isinstance(loss, torch.Tensor)
    expected = nn.functional.mse_loss(_make_model()(x), y)  # loss at the starting weights
    torch.testing.assert_close(loss.detach(), expected)
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))

    _train_step(model, opt, x, y)  # plain step() still works
    opt.zero_grad()
    nn.functional.mse_loss(model(x), y).backward()
    assert opt.step() is None


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        pytest.param(SGD, {"lr": -0.1}, id="sgd-lr"),
        pytest.param(SGD, {"lr": 0.1, "momentum": -0.5}, id="sgd-momentum"),
        pytest.param(SGD, {"lr": 0.1, "weight_decay": -1e-4}, id="sgd-wd"),
        pytest.param(SGD, {"lr": 0.1, "nesterov": True}, id="sgd-nesterov-no-momentum"),
        pytest.param(
            SGD,
            {"lr": 0.1, "momentum": 0.9, "dampening": 0.1, "nesterov": True},
            id="sgd-nesterov-dampening",
        ),
        pytest.param(Adam, {"lr": -1e-3}, id="adam-lr"),
        pytest.param(Adam, {"betas": (1.0, 0.999)}, id="adam-beta1"),
        pytest.param(Adam, {"betas": (0.9, -0.1)}, id="adam-beta2"),
        pytest.param(Adam, {"eps": -1e-8}, id="adam-eps"),
        pytest.param(Adam, {"weight_decay": -0.1}, id="adam-wd"),
        pytest.param(AdamW, {"lr": -1e-3}, id="adamw-lr"),
        pytest.param(AdamW, {"betas": (0.9, 1.0)}, id="adamw-beta2"),
        pytest.param(AdamW, {"eps": -1.0}, id="adamw-eps"),
    ],
)
def test_invalid_hyperparams_raise_value_error(cls: type, kwargs: dict[str, Any]):
    with pytest.raises(ValueError):
        cls([nn.Parameter(torch.zeros(2))], **kwargs)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        pytest.param(SGD, {"lr": 0.05, "momentum": 0.9}, id="sgd"),
        pytest.param(SGD, {"lr": 0.05, "momentum": 0.9, "nesterov": True}, id="sgd-nesterov"),
        pytest.param(Adam, {"lr": 0.05}, id="adam"),
        pytest.param(AdamW, {"lr": 0.05, "weight_decay": 1e-4}, id="adamw"),
    ],
)
def test_reduces_loss_on_linear_regression(cls: type, kwargs: dict[str, Any]):
    device = get_device()
    g = torch.Generator().manual_seed(2)
    x = torch.randn(256, 5, generator=g)
    w_true = torch.randn(5, 1, generator=g)
    y = (x @ w_true + 0.5).to(device)
    x = x.to(device)

    torch.manual_seed(0)
    model = nn.Linear(5, 1).to(device)
    opt = cls(model.parameters(), **kwargs)
    initial = nn.functional.mse_loss(model(x), y).item()
    for _ in range(300):
        _train_step(model, opt, x, y)
    final = nn.functional.mse_loss(model(x), y).item()

    assert final < 1e-3 * initial
