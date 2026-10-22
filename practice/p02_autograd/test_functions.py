import pytest
import torch
import torch.nn.functional as F
from torch import nn

from common import get_device
from practice.p02_autograd.functions import (
    LogSumExp,
    SiLU,
    StraightThroughRound,
    logsumexp,
    silu,
    ste_round,
)

DEVICE = get_device()
F64 = torch.float64  # gradcheck and exact comparisons: CPU float64 (MPS has no float64)


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_logsumexp_matches_torch_and_drops_dim(dim: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4, 5, device=DEVICE)
    out = LogSumExp.apply(x, dim)
    expected = torch.logsumexp(x, dim=dim)
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected)


def test_logsumexp_huge_values_are_finite() -> None:
    x = torch.tensor(
        [[1000.0, 1000.0, 999.0], [-1000.0, -1001.0, -1002.0], [5000.0, -5000.0, 0.0]],
        device=DEVICE,
    )
    out = logsumexp(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.logsumexp(x, dim=-1))


def test_logsumexp_all_neg_inf_slice_is_neg_inf() -> None:
    x = torch.tensor([[float("-inf")] * 3, [0.0, float("-inf"), 1.0]], dtype=F64)
    out = logsumexp(x, dim=1)
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out, torch.logsumexp(x, dim=1))


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_logsumexp_gradcheck(dim: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4, dtype=F64, requires_grad=True)
    # gradcheck compares the analytic backward against finite differences; dim (an int)
    # is passed through and must get a None gradient.
    assert torch.autograd.gradcheck(lambda t: LogSumExp.apply(t, dim), (x,))


def test_logsumexp_grad_is_softmax_times_upstream_for_huge_values() -> None:
    torch.manual_seed(0)
    base = torch.randn(4, 6, dtype=F64) * 3 + 1000.0
    upstream = torch.randn(4, dtype=F64)

    x = base.clone().requires_grad_(True)
    (logsumexp(x, dim=1) * upstream).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    expected = torch.softmax(base, dim=1) * upstream.unsqueeze(1)
    torch.testing.assert_close(x.grad, expected)


def test_ste_forward_matches_round() -> None:
    x = torch.tensor([-2.5, -1.6, -0.4, 0.0, 0.5, 1.5, 2.49, 3.7], device=DEVICE)
    out = StraightThroughRound.apply(x)
    torch.testing.assert_close(out, torch.round(x))
    assert out.dtype == x.dtype


def test_ste_gradient_equals_upstream_gradient() -> None:
    torch.manual_seed(0)
    x = torch.randn(5, 7, device=DEVICE, requires_grad=True)
    upstream = torch.randn(5, 7, device=DEVICE)
    ste_round(x * 4.0).backward(upstream)
    assert x.grad is not None
    # d(round(4x))/dx under STE is 4 * upstream (identity through round, chain rule through *4).
    torch.testing.assert_close(x.grad, 4.0 * upstream)


def test_ste_lets_quantized_weights_learn() -> None:
    torch.manual_seed(0)
    w = torch.full((3,), 0.2, device=DEVICE, requires_grad=True)
    target = torch.tensor([2.0, -1.0, 3.0], device=DEVICE)
    opt = torch.optim.SGD([w], lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        loss = ((ste_round(w) - target) ** 2).sum()
        loss.backward()
        opt.step()
    # With torch.round the gradient would be 0 and w would never move.
    torch.testing.assert_close(torch.round(w.detach()), target)


def test_silu_forward_matches_torch() -> None:
    torch.manual_seed(0)
    x = torch.cat(
        [torch.randn(100, device=DEVICE) * 5, torch.tensor([-100.0, 100.0], device=DEVICE)]
    )
    torch.testing.assert_close(SiLU.apply(x), F.silu(x))


def test_silu_gradcheck() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 5, dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(silu, (x,))


def test_silu_grad_matches_torch_at_extreme_inputs() -> None:
    base = torch.tensor([-800.0, -50.0, -3.0, 0.0, 2.0, 50.0, 800.0], dtype=F64)
    x = base.clone().requires_grad_(True)
    silu(x).sum().backward()
    ref = base.clone().requires_grad_(True)
    F.silu(ref).sum().backward()
    assert x.grad is not None and ref.grad is not None
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad, ref.grad)


def test_functions_inside_model_backward_match_builtins() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4)).to(DEVICE)
    x = torch.randn(32, 8, device=DEVICE)

    def loss_fn(act, lse) -> torch.Tensor:
        h = act(model[0](x))
        logits = model[1](h)
        # A cross-entropy-like loss: logsumexp(logits) - logits[target].
        return (lse(logits) - logits[:, 0]).mean()

    loss = loss_fn(silu, logsumexp)
    grads = torch.autograd.grad(loss, list(model.parameters()))

    ref_loss = loss_fn(F.silu, lambda t: torch.logsumexp(t, dim=-1))
    ref_grads = torch.autograd.grad(ref_loss, list(model.parameters()))

    torch.testing.assert_close(loss, ref_loss)
    for g, r in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(g, r, atol=1e-5, rtol=1e-4)
