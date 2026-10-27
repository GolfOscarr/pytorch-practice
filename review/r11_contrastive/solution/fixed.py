"""Reference fix for r11_contrastive: CLIP-style contrastive (InfoNCE) objective.

Same public API as pr.py. Lines carrying `# FIX(Bn)` map to solution/REVIEW.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

GatherFn = Callable[[torch.Tensor], Sequence[torch.Tensor]]


class ProjectionHead(nn.Module):
    """Project encoder features into the shared contrastive embedding space."""

    def __init__(self, embedding_dim: int, projection_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)  # FIX(B1): normalize the projected width
        self.dropout = nn.Dropout(dropout)  # FIX(B8): module dropout respects .eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        x = self.fc(F.gelu(projected))
        x = self.dropout(x)  # FIX(B8): was F.dropout(x, p) -> training=True even in eval
        return self.layer_norm(x + projected)


def default_gather(features: torch.Tensor) -> list[torch.Tensor]:
    """All-gather `features` across the default process group (identity when not distributed).

    Like `dist.all_gather`, the returned shards carry no autograd history; `gather_features`
    re-inserts the local shard so this rank's loss sends gradients to this rank's embeddings.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [features]
    shards = [torch.empty_like(features) for _ in range(dist.get_world_size())]
    dist.all_gather(shards, features.contiguous())
    return shards


def gather_features(features: torch.Tensor, rank: int, gather_fn: GatherFn) -> torch.Tensor:
    """Concatenate every rank's `features` (shape [local_batch, dim]) into the global batch.

    Shards are ordered by rank, so this rank's rows live at
    `[rank * local_batch, (rank + 1) * local_batch)`. Remote shards are constants; the local
    shard is the live tensor, so the gradient is exact for this rank's loss terms. (Other ranks'
    losses treat our features as constants; for the exact full-batch gradient use global logits
    or `torch.distributed.nn.functional.all_gather`.)
    """
    shards = list(gather_fn(features))
    if not 0 <= rank < len(shards):
        raise ValueError(f"rank {rank} out of range for world size {len(shards)}")
    if any(s.shape != features.shape for s in shards):
        raise ValueError("all ranks must contribute the same local batch shape")
    shards[rank] = features  # FIX(B6): keep the autograd graph for the local shard
    return torch.cat(shards, dim=0)


@torch.no_grad()
def retrieval_accuracy(
    logits_per_image: torch.Tensor, logits_per_text: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    """Top-1 retrieval accuracy in both directions.

    Each logits matrix has one row per local query and one column per candidate in the global
    batch; `labels[i]` is the column index of query i's true match.
    """
    i2t = (logits_per_image.argmax(dim=-1) == labels).float().mean()  # FIX(B7): over candidates
    t2i = (logits_per_text.argmax(dim=-1) == labels).float().mean()
    return {"image_to_text_acc": i2t.item(), "text_to_image_acc": t2i.item()}


class ClipLoss(nn.Module):
    """Symmetric InfoNCE loss over L2-normalized image/text embeddings (CLIP)."""

    def __init__(
        self,
        init_temperature: float = 0.07,
        max_logit_scale: float = 100.0,
        rank: int | None = None,
        gather_fn: GatherFn | None = None,
    ) -> None:
        super().__init__()
        if init_temperature <= 0 or max_logit_scale <= 0:
            raise ValueError("init_temperature and max_logit_scale must be positive")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))
        self.max_logit_scale = max_logit_scale
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.rank = rank
        self.gather_fn: GatherFn = gather_fn if gather_fn is not None else default_gather

    def compute_logits(
        self, image_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(logits_per_image, logits_per_text, labels)` for this rank's local batch.

        Both logits matrices have shape [local_batch, global_batch].
        """
        if image_emb.ndim != 2 or image_emb.shape != text_emb.shape:
            raise ValueError(
                f"expected matching [batch, dim] embeddings, got {tuple(image_emb.shape)} "
                f"and {tuple(text_emb.shape)}"
            )
        image = F.normalize(image_emb, dim=-1)  # FIX(B2): per-sample (feature dim), not per-batch
        text = F.normalize(text_emb, dim=-1)
        all_image = gather_features(image, self.rank, self.gather_fn)
        all_text = gather_features(text, self.rank, self.gather_fn)

        # FIX(B4): clamp the *scale* (exp of the parameter), not the log value.
        scale = self.logit_scale.exp().clamp(max=self.max_logit_scale)
        logits_per_image = scale * image @ all_text.T
        logits_per_text = scale * text @ all_image.T

        local_batch = image.shape[0]
        # FIX(B5): positives live at this rank's offset inside the global batch.
        labels = torch.arange(local_batch, device=image.device) + self.rank * local_batch
        return logits_per_image, logits_per_text, labels

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        logits_per_image, logits_per_text, labels = self.compute_logits(image_emb, text_emb)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)  # FIX(B3): text->image direction
        return (loss_i2t + loss_t2i) / 2

    @torch.no_grad()
    def retrieval_metrics(
        self, image_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> dict[str, float]:
        """Top-1 i2t / t2i accuracy for logging (no gradients)."""
        return retrieval_accuracy(*self.compute_logits(image_emb, text_emb))
