import math
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from review.r11_contrastive.pr import ClipLoss, ProjectionHead, retrieval_accuracy

# Loss math runs on CPU in float64. logit_scale is created in float32 (then cast), so its
# init value carries ~1e-7 relative error; tolerances allow for that and nothing more.
DTYPE = torch.float64
RTOL, ATOL = 1e-6, 1e-8


def _pairs(n: int = 8, dim: int = 16, seed: int = 0, noise: float = 0.5):
    g = torch.Generator().manual_seed(seed)
    image = torch.randn(n, dim, generator=g, dtype=DTYPE)
    text = image + noise * torch.randn(n, dim, generator=g, dtype=DTYPE)
    return image, text


def _reference_infonce(image: torch.Tensor, text: torch.Tensor, scale: float) -> torch.Tensor:
    image = image / image.norm(dim=1, keepdim=True)
    text = text / text.norm(dim=1, keepdim=True)
    logits = scale * image @ text.T
    labels = torch.arange(image.shape[0])
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def _detached_gather(x: torch.Tensor) -> list[torch.Tensor]:
    """World size 1 gather that, like dist.all_gather, returns tensors without autograd history."""
    return [x.detach().clone()]


def _fake_two_rank_gather(rank: int) -> Callable[[torch.Tensor], list[torch.Tensor]]:
    """Simulate 2 ranks. The other rank's shard is a feature permutation of ours, applied
    identically to image and text, so its pairs stay aligned with each other."""

    def gather(x: torch.Tensor) -> list[torch.Tensor]:
        other = torch.roll(x.detach(), shifts=3, dims=-1).flip(0).clone()
        local = x.detach().clone()
        return [local, other] if rank == 0 else [other, local]

    return gather


# --------------------------------------------------------------------------- projection head


def test_projection_head_maps_to_projection_dim():
    torch.manual_seed(0)
    head = ProjectionHead(embedding_dim=48, projection_dim=24)
    out = head(torch.randn(5, 48))
    assert out.shape == (5, 24)
    assert torch.isfinite(out).all()


def test_projection_head_is_deterministic_in_eval_mode():
    torch.manual_seed(0)
    head = ProjectionHead(embedding_dim=32, projection_dim=32, dropout=0.5).eval()
    x = torch.randn(16, 32)
    torch.testing.assert_close(head(x), head(x))


def test_projection_head_applies_dropout_in_train_mode():
    torch.manual_seed(0)
    head = ProjectionHead(embedding_dim=32, projection_dim=32, dropout=0.5).train()
    x = torch.randn(16, 32)
    assert not torch.allclose(head(x), head(x))


# --------------------------------------------------------------------------- loss values


def test_loss_is_finite_scalar_with_gradients():
    image, text = _pairs()
    image.requires_grad_(True)
    loss = ClipLoss().to(DTYPE)(image, text)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert image.grad is not None


def test_initial_logit_scale_is_inverse_temperature():
    image, _ = _pairs()
    logits_per_image, _, _ = ClipLoss(init_temperature=0.07).to(DTYPE).compute_logits(image, image)
    torch.testing.assert_close(
        logits_per_image.diagonal(), torch.full((8,), 1 / 0.07, dtype=DTYPE), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("temperature", [0.07, 0.5])
def test_loss_matches_reference_infonce(temperature: float):
    image, text = _pairs(n=8, dim=16, seed=1)
    loss = ClipLoss(init_temperature=temperature).to(DTYPE)(image, text)
    expected = _reference_infonce(image, text, scale=1 / temperature)
    torch.testing.assert_close(loss, expected, rtol=RTOL, atol=ATOL)


def test_loss_is_symmetric_in_image_and_text():
    image, text = _pairs(n=8, dim=16, seed=2, noise=1.0)
    loss_fn = ClipLoss(init_temperature=0.2).to(DTYPE)
    torch.testing.assert_close(loss_fn(image, text), loss_fn(text, image), rtol=RTOL, atol=ATOL)


def test_perfectly_aligned_pairs_have_near_zero_loss():
    image = torch.eye(6, 16, dtype=DTYPE)
    loss = ClipLoss(init_temperature=0.01).to(DTYPE)(image, image.clone())
    assert loss.item() < 1e-6


def test_loss_is_invariant_to_per_sample_embedding_scale():
    image, text = _pairs(n=8, dim=16, seed=3)
    factors = torch.linspace(0.1, 10.0, 8, dtype=DTYPE).unsqueeze(1)
    loss_fn = ClipLoss().to(DTYPE)
    torch.testing.assert_close(
        loss_fn(image * factors, text * factors.flip(0)), loss_fn(image, text), rtol=1e-9, atol=1e-9
    )


# --------------------------------------------------------------------------- logit scale


def test_logit_scale_is_capped_at_max():
    image, _ = _pairs()
    loss_fn = ClipLoss(max_logit_scale=100.0).to(DTYPE)
    with torch.no_grad():
        loss_fn.logit_scale.fill_(math.log(1000.0))
    logits_per_image, _, _ = loss_fn.compute_logits(image, image)
    torch.testing.assert_close(
        logits_per_image.diagonal(), torch.full((8,), 100.0, dtype=DTYPE), rtol=1e-6, atol=1e-6
    )


def test_logit_scale_receives_gradient_below_cap():
    image, text = _pairs()
    loss_fn = ClipLoss().to(DTYPE)
    loss_fn(image, text).backward()
    assert loss_fn.logit_scale.grad is not None
    assert loss_fn.logit_scale.grad.abs().item() > 0


# --------------------------------------------------------------------------- distributed


def test_gradient_matches_reference_when_gather_returns_detached_copies():
    image, text = _pairs(n=8, dim=16, seed=4)
    ours_img, ours_txt = image.clone().requires_grad_(), text.clone().requires_grad_()
    ref_img, ref_txt = image.clone().requires_grad_(), text.clone().requires_grad_()

    ClipLoss(rank=0, gather_fn=_detached_gather).to(DTYPE)(ours_img, ours_txt).backward()
    _reference_infonce(ref_img, ref_txt, scale=1 / 0.07).backward()

    assert ours_img.grad is not None and ours_txt.grad is not None
    torch.testing.assert_close(ours_img.grad, ref_img.grad, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(ours_txt.grad, ref_txt.grad, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("rank", [0, 1])
def test_labels_point_at_local_rows_of_global_batch(rank: int):
    image, text = _pairs(n=4, dim=16, seed=5, noise=0.0)
    loss_fn = ClipLoss(rank=rank, gather_fn=_fake_two_rank_gather(rank)).to(DTYPE)
    logits_per_image, logits_per_text, labels = loss_fn.compute_logits(image, text)
    assert logits_per_image.shape == logits_per_text.shape == (4, 8)
    torch.testing.assert_close(labels, torch.arange(4) + 4 * rank)


@pytest.mark.parametrize("rank", [0, 1])
def test_aligned_pairs_on_any_rank_have_low_loss_and_perfect_retrieval(rank: int):
    image, text = _pairs(n=4, dim=16, seed=6, noise=0.0)
    loss_fn = ClipLoss(init_temperature=0.01, rank=rank, gather_fn=_fake_two_rank_gather(rank))
    loss_fn = loss_fn.to(DTYPE)
    assert loss_fn(image, text).item() < 1e-3
    metrics = loss_fn.retrieval_metrics(image, text)
    assert metrics == {"image_to_text_acc": 1.0, "text_to_image_acc": 1.0}


# --------------------------------------------------------------------------- metrics


def test_retrieval_accuracy_scores_each_query_row():
    # Row-wise argmax is [0, 1] -> both image queries retrieve their caption.
    logits_per_image = torch.tensor([[5.0, 4.0], [0.0, 1.0]])
    # Row-wise argmax is [1, 1] -> only the second text query is correct.
    logits_per_text = torch.tensor([[0.0, 3.0], [1.0, 2.0]])
    labels = torch.arange(2)
    metrics = retrieval_accuracy(logits_per_image, logits_per_text, labels)
    assert metrics == {"image_to_text_acc": 1.0, "text_to_image_acc": 0.5}


def test_retrieval_metrics_perfect_for_identical_embeddings():
    image, _ = _pairs()
    metrics = ClipLoss().to(DTYPE).retrieval_metrics(image, image.clone())
    assert metrics == {"image_to_text_acc": 1.0, "text_to_image_acc": 1.0}
