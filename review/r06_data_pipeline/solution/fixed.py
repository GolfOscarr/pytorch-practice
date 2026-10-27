"""Reference fix for review/r06_data_pipeline/pr.py (same public API).

Bug fixes are tagged `FIX(Bn)`; see REVIEW.md for the answer key. Production improvements
(input validation, logging, per-loader generators, persistent workers) are tagged `IMPROVE`.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, get_worker_info

logger = logging.getLogger(__name__)

PAD_ID = 0  # token ids are in [0, vocab_size); padding is identified by `lengths`, not by id
STD_EPS = 1e-8


@dataclass(frozen=True)
class Record:
    """One exported example: tabular features plus an event-token sequence."""

    id: int
    user_id: str
    features: tuple[float, ...]
    tokens: tuple[int, ...]
    label: int


@dataclass
class Normalizer:
    """Per-feature z-score normalizer."""

    mean: np.ndarray
    std: np.ndarray

    def apply(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.std


@dataclass
class PreparedData:
    train: list[Record]
    val: list[Record]
    test: list[Record]
    normalizer: Normalizer


class Batch(NamedTuple):
    ids: torch.Tensor  # [B] long
    features: torch.Tensor  # [B, F] float32
    tokens: torch.Tensor  # [B, T] long, padded with PAD_ID
    lengths: torch.Tensor  # [B] long
    mask: torch.Tensor  # [B, T] bool, True on real tokens
    labels: torch.Tensor  # [B] long


def split_dataset(
    records: list[Record],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    group_key: str | None = None,
) -> tuple[list[Record], list[Record], list[Record]]:
    """Shuffle and split records into (train, val, test) according to `ratios`.

    If `group_key` is given (e.g. "user_id"), every group lands in exactly one split.
    """
    if not np.isclose(sum(ratios), 1.0) or min(ratios) < 0:  # IMPROVE: validate inputs
        raise ValueError(f"ratios must be non-negative and sum to 1, got {ratios}")
    rng = random.Random(seed)
    # FIX(B3): split over shuffled *groups*, never over records, so a group cannot straddle a
    # cut point (sorting by key kept groups contiguous but still cut through them, and also
    # assigned groups to splits in key order instead of at random).
    groups: dict[object, list[Record]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[getattr(r, group_key) if group_key is not None else i].append(r)
    keys = list(groups)
    rng.shuffle(keys)
    n = len(records)
    targets = (ratios[0] * n, (ratios[0] + ratios[1]) * n)
    splits: tuple[list[Record], list[Record], list[Record]] = ([], [], [])
    seen = 0
    for key in keys:
        part = 0 if seen < targets[0] else 1 if seen < targets[1] else 2
        splits[part].extend(groups[key])
        seen += len(groups[key])
    return splits


def oversample_minority(records: list[Record], seed: int = 0) -> list[Record]:
    """Duplicate records of the smaller classes until every class matches the largest one."""
    by_label: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        by_label[r.label].append(r)
    target = max(len(group) for group in by_label.values())
    rng = random.Random(seed)
    out = list(records)
    for group in by_label.values():
        out.extend(rng.choices(group, k=target - len(group)))
    return out


def fit_normalizer(records: list[Record]) -> Normalizer:
    """Fit per-feature mean/std on `records` (pass the training split only)."""
    if not records:
        raise ValueError("cannot fit a normalizer on zero records")
    x = np.asarray([r.features for r in records], dtype=np.float64)
    std = x.std(axis=0)
    # FIX(B4): constant features have std 0 -> division gives nan/inf. Leave them unscaled.
    constant = std < STD_EPS
    if constant.any():
        logger.warning("features %s are constant; not scaling them", np.flatnonzero(constant))
    return Normalizer(mean=x.mean(axis=0), std=np.where(constant, 1.0, std))


def prepare_datasets(
    records: list[Record],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    group_key: str | None = None,
    oversample: bool = True,
) -> PreparedData:
    """Split, fit the feature normalizer on train, then balance the train split only."""
    # FIX(B2): split BEFORE oversampling; duplicating first puts copies of the same record in
    # train and val/test, and val/test must keep the real class distribution anyway.
    train, val, test = split_dataset(records, ratios, seed=seed, group_key=group_key)
    # FIX(B1): fit normalization statistics on the training split only (no val/test leakage).
    normalizer = fit_normalizer(train)
    if oversample:
        train = oversample_minority(train, seed=seed)
    logger.info("split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))
    return PreparedData(train=train, val=val, test=test, normalizer=normalizer)


class SequenceDataset(Dataset):
    """Normalizes features and (optionally) adds Gaussian noise as augmentation."""

    def __init__(
        self,
        records: list[Record],
        normalizer: Normalizer,
        augment: bool = False,
        noise_std: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.records = records
        self.normalizer = normalizer
        self.augment = augment
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, torch.Tensor, int]:
        r = self.records[idx]
        x = self.normalizer.apply(np.asarray(r.features, dtype=np.float64))
        if self.augment:
            x = x + self.rng.normal(0.0, self.noise_std, size=x.shape)
        features = torch.from_numpy(x).float()
        tokens = torch.tensor(r.tokens, dtype=torch.long)
        return r.id, features, tokens, r.label


def collate_batch(batch: list[tuple[int, torch.Tensor, torch.Tensor, int]]) -> Batch:
    """Stack features and pad token sequences to the longest one in the batch."""
    ids, features, tokens, labels = zip(*batch, strict=True)
    padded = pad_sequence(list(tokens), batch_first=True, padding_value=PAD_ID)
    lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
    return Batch(
        ids=torch.tensor(ids, dtype=torch.long),
        features=torch.stack(features),
        tokens=padded,
        lengths=lengths,
        # FIX(B5): build the mask from lengths. PAD_ID == 0 is also a real token id, so
        # `padded != PAD_ID` silently masked out genuine token-0 positions.
        mask=torch.arange(padded.shape[1])[None, :] < lengths[:, None],
        labels=torch.tensor(labels, dtype=torch.long),
    )


def _seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker its own RNG seed."""
    seed = torch.initial_seed() % 2**32  # base_seed + worker_id, differs per worker and epoch
    np.random.seed(seed)
    random.seed(seed)
    # FIX(B7): the dataset owns its own Generator, which is copied verbatim into every worker
    # (fork or pickle). Seeding the global NumPy RNG does not touch it, so all workers drew the
    # same "random" noise. Re-seed the worker's dataset copy explicitly.
    info = get_worker_info()
    if info is not None and isinstance(info.dataset, SequenceDataset):
        info.dataset.rng = np.random.default_rng(seed)


def make_loaders(
    data: PreparedData,
    batch_size: int = 32,
    num_workers: int = 0,
    augment: bool = True,
    noise_std: float = 0.1,
    seed: int = 0,
    multiprocessing_context: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build the (train, val) DataLoaders."""
    train_ds = SequenceDataset(data.train, data.normalizer, augment, noise_std, seed)
    # FIX(B8): validation must be deterministic: no augmentation and no shuffling.
    val_ds = SequenceDataset(data.val, data.normalizer, augment=False)
    worker_kwargs: dict[str, object] = {}
    if num_workers > 0:
        worker_kwargs = {
            "multiprocessing_context": multiprocessing_context,
            "persistent_workers": True,  # IMPROVE: don't respawn workers every epoch
        }
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),  # IMPROVE: one generator per loader
        **worker_kwargs,  # type: ignore[arg-type]
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,  # FIX(B8)
        num_workers=num_workers,
        collate_fn=collate_batch,
        **worker_kwargs,  # type: ignore[arg-type]
    )
    return train_loader, val_loader


def masked_sequence_loss(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy at every real timestep against the sequence label.

    logits: [B, T, C], labels: [B], mask: [B, T] (True on real tokens).
    """
    _, steps, _ = logits.shape
    targets = labels[:, None].expand(-1, steps)
    per_step = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    # FIX(B6): average over real positions only. `.mean()` divided by B*T, so padding diluted
    # the loss and a sample's weight depended on how long its batch-mates were.
    mask = mask.to(per_step.dtype)
    return (per_step * mask).sum() / mask.sum().clamp_min(1.0)
