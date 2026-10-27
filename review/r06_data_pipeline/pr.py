"""Add dataset prep + DataLoaders for the tabular/sequence classifier.

## Summary
This PR adds the input pipeline for the hybrid classifier (per-user tabular features + an event
token sequence). It covers everything between the raw `Record` list we export from the warehouse
and the batches the training loop consumes, so the model PR can land on top of it.

## Changes
- `split_dataset`: seeded train/val/test split. Pass `group_key="user_id"` to keep each user's
  records together (users appear many times in the export).
- `oversample_minority`: duplicates minority-class records so every class has equal weight.
- `Normalizer` / `fit_normalizer`: per-feature z-scoring for the tabular part.
- `prepare_datasets`: one call that runs oversampling, the split and the normalizer fit and
  returns a `PreparedData` bundle.
- `SequenceDataset`: applies normalization plus Gaussian feature-noise augmentation; it owns a
  seeded NumPy `Generator`, so augmentation is reproducible.
- `collate_batch`: pads variable-length token sequences and builds the padding mask. Returns a
  `Batch` named tuple (ids, features, tokens, lengths, mask, labels).
- `make_loaders`: builds the train/val `DataLoader`s with workers, a seeded shuffle generator
  and a `worker_init_fn` that seeds every worker.
- `masked_sequence_loss`: per-timestep cross-entropy (each step predicts the sequence label, so
  the model learns to classify early) averaged over the real, non-pad positions.

## Testing
Ran `prepare_datasets` + `make_loaders(num_workers=4)` over the full export; batch shapes and
dtypes look right and a smoke-training run converges. Loss goes down on both splits.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD_ID = 0  # token ids are in [0, vocab_size); padded positions use this id


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

    If `group_key` is given (e.g. "user_id"), records sharing that attribute are kept together.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    order = list(records)
    random.Random(seed).shuffle(order)
    if group_key is not None:
        order.sort(key=lambda r: getattr(r, group_key))
    n_train = int(ratios[0] * len(order))
    n_val = int(ratios[1] * len(order))
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


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
    """Fit per-feature mean/std on `records`."""
    x = np.asarray([r.features for r in records], dtype=np.float64)
    return Normalizer(mean=x.mean(axis=0), std=x.std(axis=0))


def prepare_datasets(
    records: list[Record],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    group_key: str | None = None,
    oversample: bool = True,
) -> PreparedData:
    """Balance classes, split, and fit the feature normalizer."""
    if oversample:
        records = oversample_minority(records, seed=seed)
    train, val, test = split_dataset(records, ratios, seed=seed, group_key=group_key)
    normalizer = fit_normalizer(records)
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
    return Batch(
        ids=torch.tensor(ids, dtype=torch.long),
        features=torch.stack(features),
        tokens=padded,
        lengths=torch.tensor([len(t) for t in tokens], dtype=torch.long),
        mask=padded != PAD_ID,
        labels=torch.tensor(labels, dtype=torch.long),
    )


def _seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker its own RNG seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


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
    val_ds = SequenceDataset(data.val, data.normalizer, augment, noise_std, seed)
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": collate_batch,
        "worker_init_fn": _seed_worker,
        "generator": torch.Generator().manual_seed(seed),
        "multiprocessing_context": multiprocessing_context if num_workers > 0 else None,
    }
    return DataLoader(train_ds, **common), DataLoader(val_ds, **common)


def masked_sequence_loss(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy at every real timestep against the sequence label.

    logits: [B, T, C], labels: [B], mask: [B, T] (True on real tokens).
    """
    _, steps, _ = logits.shape
    targets = labels[:, None].expand(-1, steps)
    per_step = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return (per_step * mask).mean()
