import random
from collections import Counter

import numpy as np
import pytest
import torch

from review.r06_data_pipeline.pr import (
    PAD_ID,
    Batch,
    Record,
    collate_batch,
    fit_normalizer,
    make_loaders,
    masked_sequence_loss,
    prepare_datasets,
    split_dataset,
)

N_FEATURES = 3
VOCAB = 5


def _make_records(n: int = 200, n_users: int = 40, minority_every: int = 5) -> list[Record]:
    """Synthetic export: users repeat, labels are imbalanced, tokens include id 0."""
    rng = random.Random(123)
    records = []
    for i in range(n):
        user = f"u{rng.randrange(n_users):03d}"
        features = tuple(rng.gauss(10.0 * (j + 1), 1.0 + j) for j in range(N_FEATURES))
        tokens = tuple(rng.randrange(VOCAB) for _ in range(rng.randint(2, 9)))
        label = 1 if i % minority_every == 0 else 0
        records.append(Record(i, user, features, tokens, label))
    return records


def _ids(records: list[Record]) -> set[int]:
    return {r.id for r in records}


# ---------------------------------------------------------------- splitting / leakage


def test_split_sizes_follow_ratios_and_cover_everything():
    records = _make_records()
    train, val, test = split_dataset(records, (0.8, 0.1, 0.1), seed=0)
    assert (len(train), len(val), len(test)) == (160, 20, 20)
    assert sorted(_ids(train) | _ids(val) | _ids(test)) == list(range(len(records)))


def test_split_is_reproducible_for_a_seed():
    records = _make_records()
    a = split_dataset(records, seed=7, group_key="user_id")
    b = split_dataset(records, seed=7, group_key="user_id")
    assert [[r.id for r in s] for s in a] == [[r.id for r in s] for s in b]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_group_split_puts_each_user_in_exactly_one_split(seed):
    records = _make_records()
    train, val, test = split_dataset(records, (0.6, 0.2, 0.2), seed=seed, group_key="user_id")
    users = [{r.user_id for r in s} for s in (train, val, test)]
    assert not users[0] & users[1]
    assert not users[0] & users[2]
    assert not users[1] & users[2]
    assert len(train) + len(val) + len(test) == len(records)
    assert val and test


def test_group_split_assigns_users_to_splits_at_random():
    # Every split should contain a mix of users, not just a contiguous range of user ids.
    records = _make_records()
    train, _, test = split_dataset(records, (0.5, 0.25, 0.25), seed=0, group_key="user_id")
    assert min(r.user_id for r in test) < max(r.user_id for r in train)


def test_no_record_appears_in_both_train_and_eval_splits():
    records = _make_records()
    data = prepare_datasets(records, seed=0, oversample=True)
    train_ids = _ids(data.train)
    assert not train_ids & _ids(data.val)
    assert not train_ids & _ids(data.test)
    for split in (data.val, data.test):
        assert len(split) == len(_ids(split)), "eval splits must not contain duplicates"


def test_oversampling_balances_the_training_split():
    data = prepare_datasets(_make_records(), seed=0, oversample=True)
    counts = Counter(r.label for r in data.train)
    assert counts[0] == counts[1]


def test_normalizer_statistics_come_from_training_records_only():
    data = prepare_datasets(_make_records(), seed=0, group_key="user_id", oversample=False)
    x_train = np.asarray([r.features for r in data.train])
    np.testing.assert_allclose(data.normalizer.mean, x_train.mean(axis=0))
    np.testing.assert_allclose(data.normalizer.std, x_train.std(axis=0))


# ---------------------------------------------------------------- normalizer


def test_normalized_training_features_are_standardized():
    records = _make_records()
    norm = fit_normalizer(records)
    z = norm.apply(np.asarray([r.features for r in records]))
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-9)


def test_constant_feature_normalizes_to_finite_values():
    records = [Record(i, "u", (float(i), 5.0), (1,), 0) for i in range(10)]
    norm = fit_normalizer(records)
    z = norm.apply(np.asarray([r.features for r in records]))
    assert np.isfinite(z).all()
    np.testing.assert_allclose(z[:, 1], 0.0)


# ---------------------------------------------------------------- collate / loss


def _item(i: int, tokens: tuple[int, ...]) -> tuple[int, torch.Tensor, torch.Tensor, int]:
    return i, torch.zeros(N_FEATURES), torch.tensor(tokens, dtype=torch.long), i % 2


def test_collate_pads_to_longest_sequence():
    batch = collate_batch([_item(0, (3, 1, 4)), _item(1, (2,)), _item(2, (4, 4))])
    assert isinstance(batch, Batch)
    assert batch.tokens.shape == (3, 3)
    assert batch.features.shape == (3, N_FEATURES)
    assert batch.tokens.dtype == torch.long and batch.mask.dtype == torch.bool
    assert batch.lengths.tolist() == [3, 1, 2]
    assert batch.tokens[1, 1:].eq(PAD_ID).all()
    assert batch.labels.tolist() == [0, 1, 0]


def test_mask_marks_exactly_the_real_tokens_even_for_token_id_zero():
    batch = collate_batch([_item(0, (0, 2, 0, 1)), _item(1, (0,)), _item(2, (3, 0))])
    expected = torch.tensor(
        [[True, True, True, True], [True, False, False, False], [True, True, False, False]]
    )
    assert torch.equal(batch.mask, expected)
    assert batch.mask.sum(dim=1).tolist() == batch.lengths.tolist()


def test_loss_of_a_sample_does_not_depend_on_padding():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 3)
    labels = torch.tensor([2])
    alone = masked_sequence_loss(logits, labels, torch.ones(1, 4, dtype=torch.bool))

    padded_logits = torch.cat([logits, torch.randn(1, 6, 3) * 50], dim=1)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, :4] = True
    padded = masked_sequence_loss(padded_logits, labels, mask)
    torch.testing.assert_close(padded, alone)


def test_loss_matches_mean_cross_entropy_over_real_positions():
    torch.manual_seed(1)
    logits = torch.randn(2, 5, 4)
    labels = torch.tensor([1, 3])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    rows = [logits[0, :3], logits[1]]
    targets = torch.tensor([1, 1, 1, 3, 3, 3, 3, 3])
    expected = torch.nn.functional.cross_entropy(torch.cat(rows), targets)
    torch.testing.assert_close(masked_sequence_loss(logits, labels, mask), expected)


# ---------------------------------------------------------------- loaders


def _clean_features(data) -> dict[int, torch.Tensor]:
    return {
        r.id: torch.from_numpy(data.normalizer.apply(np.asarray(r.features))).float()
        for r in data.train + data.val
    }


def test_validation_loader_is_deterministic_and_unaugmented():
    data = prepare_datasets(_make_records(), seed=0, oversample=False)
    _, val_loader = make_loaders(data, batch_size=8, augment=True, noise_std=0.5)
    clean = _clean_features(data)

    first = list(val_loader)
    second = list(val_loader)
    ids = torch.cat([b.ids for b in first]).tolist()
    assert ids == [r.id for r in data.val], "val order must be stable and unshuffled"
    assert ids == torch.cat([b.ids for b in second]).tolist()
    for b in first:
        expected = torch.stack([clean[i] for i in b.ids.tolist()])
        torch.testing.assert_close(b.features, expected)


def test_training_loader_augments_features():
    data = prepare_datasets(_make_records(), seed=0, oversample=False)
    train_loader, _ = make_loaders(data, batch_size=16, augment=True, noise_std=0.5)
    clean = _clean_features(data)
    b = next(iter(train_loader))
    expected = torch.stack([clean[i] for i in b.ids.tolist()])
    assert not torch.allclose(b.features, expected)


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_each_worker_draws_different_augmentation_noise():
    # fork keeps the (possibly --solution redirected) module in the workers; spawn would
    # re-import pr.py from disk in each worker.
    data = prepare_datasets(_make_records(n=64), seed=0, oversample=False)
    train_loader, _ = make_loaders(
        data,
        batch_size=4,
        num_workers=2,
        augment=True,
        noise_std=1.0,
        seed=0,
        multiprocessing_context="fork",
    )
    clean = _clean_features(data)
    noise = []
    for b in train_loader:
        expected = torch.stack([clean[i] for i in b.ids.tolist()])
        noise.append(b.features - expected)
    rows = torch.cat(noise)
    assert rows.shape[0] == len(data.train)
    distinct = {tuple(round(v, 4) for v in row.tolist()) for row in rows}
    assert len(distinct) == rows.shape[0], "augmentation noise repeated across samples"
