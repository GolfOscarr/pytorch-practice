import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from review.r07_mixed_precision.pr import (
    StepStats,
    cosine_similarity,
    l2_normalize,
    masked_mean,
    multilabel_bce,
    sequence_log_probs,
    smoothed_cross_entropy,
    softmax_with_temperature,
    train_step,
)

# Everything runs on CPU: torch.autocast and torch.amp.GradScaler both support "cpu", which
# keeps the tests deterministic and independent of the machine's accelerator.
INIT_SCALE = 2.0**12
EXTREME = 1e4


def _make_setup(
    input_scale: float = 1.0,
) -> tuple[nn.Sequential, tuple[torch.Tensor, torch.Tensor], torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    g = torch.Generator().manual_seed(1)
    x = torch.randn(32, 8, generator=g) * input_scale
    y = torch.randint(0, 4, (32,), generator=g)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    return model, (x, y), opt


def _scaler() -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cpu", init_scale=INIT_SCALE)


def _flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def _reference_grad_norm(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
    """Gradient norm of the same loss computed in plain float32, no scaling."""
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(batch[0]), batch[1]).backward()
    norm = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).norm()
    model.zero_grad(set_to_none=True)
    return norm.item()


# --------------------------------------------------------------------------- train_step


def test_train_step_returns_plain_python_stats() -> None:
    model, batch, opt = _make_setup()
    stats = train_step(model, batch, opt, _scaler(), grad_clip=1.0, dtype=torch.bfloat16)
    assert isinstance(stats, StepStats)
    assert isinstance(stats.loss, float) and math.isfinite(stats.loss)
    assert isinstance(stats.grad_norm, float)
    assert isinstance(stats.skipped, bool)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_train_step_forward_runs_in_requested_low_precision(dtype: torch.dtype) -> None:
    model, batch, opt = _make_setup()
    seen: list[torch.dtype] = []
    model[0].register_forward_hook(lambda _m, _i, out: seen.append(out.dtype))
    train_step(model, batch, opt, _scaler(), grad_clip=1.0, dtype=dtype)
    assert seen == [dtype]


def test_train_step_reports_unscaled_gradient_norm() -> None:
    model, batch, opt = _make_setup()
    expected = _reference_grad_norm(model, batch)
    stats = train_step(model, batch, opt, _scaler(), grad_clip=1e6, dtype=torch.bfloat16)
    assert stats.grad_norm == pytest.approx(expected, rel=0.05)


def test_train_step_clipped_update_has_norm_grad_clip() -> None:
    # Large inputs -> true gradient norm far above the threshold, so clipping is active.
    # With SGD(lr=1) the parameter change is exactly the clipped gradient.
    model, batch, opt = _make_setup(input_scale=50.0)
    assert _reference_grad_norm(model, batch) > 5.0
    before = _flat_params(model)
    stats = train_step(model, batch, opt, _scaler(), grad_clip=0.5, dtype=torch.bfloat16)
    assert not stats.skipped
    update_norm = (_flat_params(model) - before).norm().item()
    assert update_norm == pytest.approx(0.5, rel=1e-2)


def test_train_step_unclipped_update_matches_fp32_sgd() -> None:
    model, batch, opt = _make_setup()
    ref_model, _, ref_opt = _make_setup()
    ref_opt.zero_grad()
    F.cross_entropy(ref_model(batch[0]), batch[1]).backward()
    ref_opt.step()
    train_step(model, batch, opt, _scaler(), grad_clip=1e6, dtype=torch.bfloat16)
    torch.testing.assert_close(_flat_params(model), _flat_params(ref_model), rtol=0, atol=2e-2)


def test_train_step_skips_update_and_lowers_scale_on_nonfinite_gradients() -> None:
    model, (x, y), opt = _make_setup()
    x = x.clone()
    x[0, 0] = float("inf")
    before = _flat_params(model)
    stats = train_step(model, (x, y), opt, _scaler(), grad_clip=1.0, dtype=torch.bfloat16)
    assert stats.skipped
    assert stats.scale < INIT_SCALE
    assert torch.equal(_flat_params(model), before)


# --------------------------------------------------------------------------- losses


def _ref_smoothed_ce(logits: torch.Tensor, targets: torch.Tensor, s: float) -> torch.Tensor:
    logp = torch.log_softmax(logits.double(), dim=-1)
    nll = -logp.gather(-1, targets[:, None]).squeeze(-1)
    uniform = -logp.mean(dim=-1)
    return ((1 - s) * nll + s * uniform).mean()


@pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.3])
def test_smoothed_cross_entropy_matches_reference(smoothing: float) -> None:
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(16, 10, generator=g) * 3
    targets = torch.randint(0, 10, (16,), generator=g)
    out = smoothed_cross_entropy(logits, targets, smoothing)
    torch.testing.assert_close(
        out.double(), _ref_smoothed_ce(logits, targets, smoothing), rtol=1e-5, atol=1e-6
    )


def test_smoothed_cross_entropy_is_finite_and_correct_for_extreme_logits() -> None:
    logits = torch.tensor([[EXTREME, -EXTREME, 0.0], [-EXTREME, 0.0, EXTREME]])
    targets = torch.tensor([1, 0])  # confidently wrong -> loss ~ 1e4
    out = smoothed_cross_entropy(logits, targets, 0.1)
    assert torch.isfinite(out)
    torch.testing.assert_close(
        out.double(), _ref_smoothed_ce(logits, targets, 0.1), rtol=1e-5, atol=1e-3
    )


def _ref_bce(logits: torch.Tensor, targets: torch.Tensor, pos_weight=None) -> torch.Tensor:
    x, y = logits.double(), targets.double()
    pw = torch.ones_like(x) if pos_weight is None else pos_weight.double().expand_as(x)
    log_sig = F.logsigmoid(x)  # log p
    log_one_minus = F.logsigmoid(-x)  # log (1 - p)
    return -(pw * y * log_sig + (1 - y) * log_one_minus).mean()


@pytest.mark.parametrize("use_pos_weight", [False, True])
def test_multilabel_bce_matches_reference(use_pos_weight: bool) -> None:
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(8, 5, generator=g) * 2
    targets = (torch.rand(8, 5, generator=g) > 0.5).float()
    pos_weight = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.0]) if use_pos_weight else None
    out = multilabel_bce(logits, targets, pos_weight)
    torch.testing.assert_close(
        out.double(), _ref_bce(logits, targets, pos_weight), rtol=1e-5, atol=1e-6
    )


def test_multilabel_bce_confident_correct_predictions_have_near_zero_loss() -> None:
    logits = torch.tensor([[EXTREME, -EXTREME], [-EXTREME, EXTREME]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert multilabel_bce(logits, targets).item() < 1e-6


def _ref_seq_log_probs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.double(), dim=-1).gather(-1, tokens[..., None])[..., 0]


def test_sequence_log_probs_shape_and_values() -> None:
    g = torch.Generator().manual_seed(4)
    logits = torch.randn(2, 5, 7, generator=g)
    tokens = torch.randint(0, 7, (2, 5), generator=g)
    out = sequence_log_probs(logits, tokens)
    assert out.shape == (2, 5)
    torch.testing.assert_close(
        out.double(), _ref_seq_log_probs(logits, tokens), rtol=1e-5, atol=1e-6
    )


def test_sequence_log_probs_finite_for_unlikely_tokens_under_extreme_logits() -> None:
    logits = torch.zeros(1, 2, 4)
    logits[0, 0, 0] = EXTREME
    logits[0, 1, 3] = EXTREME
    tokens = torch.tensor([[1, 0]])  # tokens the model considers ~impossible
    out = sequence_log_probs(logits, tokens)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.double(), _ref_seq_log_probs(logits, tokens), rtol=1e-5, atol=1e-3
    )


# --------------------------------------------------------------------------- reductions


def test_masked_mean_ignores_masked_positions() -> None:
    values = torch.tensor([[1.0, 2.0, 1e3], [3.0, 1e3, 1e3]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    assert masked_mean(values, mask).item() == pytest.approx(2.0)


def test_masked_mean_of_many_fp16_values_does_not_overflow() -> None:
    # 8192 tokens with loss 10.0: the true sum (81920) exceeds the float16 max (65504).
    values = torch.full((4, 2048), 10.0, dtype=torch.float16)
    mask = torch.ones(4, 2048, dtype=torch.bool)
    out = masked_mean(values, mask)
    assert torch.isfinite(out)
    assert out.item() == pytest.approx(10.0, rel=1e-3)


# --------------------------------------------------------------------------- normalisation


def test_l2_normalize_gives_unit_rows() -> None:
    g = torch.Generator().manual_seed(5)
    x = torch.randn(6, 5, generator=g) * 4
    out = l2_normalize(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(6), rtol=1e-5, atol=1e-6)


def test_l2_normalize_zero_vector_stays_zero_in_float16() -> None:
    x = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 4.0]], dtype=torch.float16)
    out = l2_normalize(x)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.float(), torch.tensor([[0.0, 0.0, 0.0], [0.6, 0.0, 0.8]]), rtol=1e-3, atol=1e-3
    )


def test_cosine_similarity_matches_torch() -> None:
    g = torch.Generator().manual_seed(6)
    a, b = torch.randn(4, 8, generator=g), torch.randn(4, 8, generator=g)
    torch.testing.assert_close(cosine_similarity(a, b), F.cosine_similarity(a, b, dim=-1))


def test_cosine_similarity_with_zero_vector_is_zero_in_float16() -> None:
    a = torch.zeros(2, 4, dtype=torch.float16)
    b = torch.ones(2, 4, dtype=torch.float16)
    out = cosine_similarity(a, b)
    assert torch.isfinite(out).all()
    assert torch.equal(out.float(), torch.zeros(2))


# --------------------------------------------------------------------------- softmax


def test_softmax_with_temperature_matches_reference_for_extreme_logits() -> None:
    logits = torch.tensor([[EXTREME, 0.0, -EXTREME], [1.0, 2.0, 3.0]], dtype=torch.float16)
    for t in (0.5, 1.0, 4.0):
        out = softmax_with_temperature(logits, t)
        ref = torch.softmax(logits.double() / t, dim=-1)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=1e-6)


def test_softmax_higher_temperature_is_flatter() -> None:
    logits = torch.tensor([1.0, 2.0, 3.0])
    sharp = softmax_with_temperature(logits, 0.5)
    flat = softmax_with_temperature(logits, 5.0)
    assert sharp.max() > flat.max()
