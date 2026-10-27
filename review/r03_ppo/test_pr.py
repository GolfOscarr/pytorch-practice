import dataclasses

import pytest
import torch

from review.r03_ppo.pr import PPOLossOutput, compute_gae, normalize_advantages, ppo_loss

DTYPE = torch.float64


def _t(*xs: float) -> torch.Tensor:
    return torch.tensor(xs, dtype=DTYPE)


# ---------------------------------------------------------------------------- compute_gae


def test_gae_matches_hand_computed_example_with_episode_boundary():
    # gamma=0.9, lam=0.8; the episode ends at t=1, a new one starts at t=2.
    # t=2: delta = 3 + 0.9*2.0 - 1.5 = 3.3                      A = 3.3
    # t=1: terminal -> delta = 2 - 1.0 = 1.0, no carry          A = 1.0
    # t=0: delta = 1 + 0.9*1.0 - 0.5 = 1.4; A = 1.4 + 0.72*1.0  A = 2.12
    adv, ret = compute_gae(
        rewards=_t(1.0, 2.0, 3.0),
        values=_t(0.5, 1.0, 1.5),
        dones=_t(0.0, 1.0, 0.0),
        last_value=torch.tensor(2.0, dtype=DTYPE),
        gamma=0.9,
        lam=0.8,
    )
    torch.testing.assert_close(adv, _t(2.12, 1.0, 3.3))
    torch.testing.assert_close(ret, _t(2.62, 2.0, 4.8))


def test_gae_lambda_zero_is_one_step_td_error():
    rewards, values = _t(1.0, -1.0, 0.5, 2.0), _t(0.2, 0.4, -0.3, 1.0)
    last = torch.tensor(0.7, dtype=DTYPE)
    adv, _ = compute_gae(rewards, values, torch.zeros(4, dtype=DTYPE), last, gamma=0.95, lam=0.0)
    next_values = torch.cat([values[1:], last.view(1)])
    torch.testing.assert_close(adv, rewards + 0.95 * next_values - values)


def test_gae_lambda_one_returns_are_bootstrapped_discounted_sums():
    rewards, values = _t(1.0, 2.0, 3.0, 4.0), _t(9.0, -3.0, 5.0, 0.1)
    last, gamma = torch.tensor(10.0, dtype=DTYPE), 0.5
    _, ret = compute_gae(rewards, values, torch.zeros(4, dtype=DTYPE), last, gamma=gamma, lam=1.0)
    expected, g = [], last.item()
    for r in reversed(rewards.tolist()):
        g = r + gamma * g
        expected.append(g)
    torch.testing.assert_close(ret, torch.tensor(expected[::-1], dtype=DTYPE))


def test_gae_terminal_step_ignores_value_of_next_episode():
    # Only the value of the first state of the *next* episode changes; advantages for
    # the finished episode (t <= 1) must not move.
    rewards, dones = _t(1.0, 1.0, 1.0), _t(0.0, 1.0, 0.0)
    last = torch.tensor(0.0, dtype=DTYPE)
    adv_a, _ = compute_gae(rewards, _t(0.3, 0.6, 0.0), dones, last)
    adv_b, _ = compute_gae(rewards, _t(0.3, 0.6, 100.0), dones, last)
    torch.testing.assert_close(adv_a[:2], adv_b[:2])


def test_gae_batched_envs_match_per_env_computation():
    g = torch.Generator().manual_seed(0)
    T, N = 6, 3
    rewards = torch.randn(T, N, generator=g, dtype=DTYPE)
    values = torch.randn(T, N, generator=g, dtype=DTYPE)
    dones = (torch.rand(T, N, generator=g) < 0.3).to(DTYPE)
    last = torch.randn(N, generator=g, dtype=DTYPE)
    adv, ret = compute_gae(rewards, values, dones, last, gamma=0.97, lam=0.9)
    assert adv.shape == ret.shape == (T, N)
    for n in range(N):
        a_n, r_n = compute_gae(rewards[:, n], values[:, n], dones[:, n], last[n], 0.97, 0.9)
        torch.testing.assert_close(adv[:, n], a_n)
        torch.testing.assert_close(ret[:, n], r_n)


def test_gae_outputs_are_constants_for_the_optimizer():
    values = torch.tensor([0.5, 1.0, 1.5], dtype=DTYPE, requires_grad=True)
    adv, ret = compute_gae(
        _t(1.0, 2.0, 3.0), values, _t(0.0, 0.0, 0.0), torch.tensor(0.0, dtype=DTYPE)
    )
    assert not adv.requires_grad
    assert not ret.requires_grad


# ---------------------------------------------------------------------------- ppo_loss


def _reference_ppo(new_logp, old_logp, adv, ret, values, entropy, clip_eps, vf_coef, ent_coef):
    ratio = torch.exp(new_logp - old_logp)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    pg = -torch.minimum(ratio * adv, clipped * adv).mean()
    vf = 0.5 * ((values - ret) ** 2).mean()
    ent = entropy.mean()
    return pg, vf, ent, pg + vf_coef * vf - ent_coef * ent


def _batch(seed: int = 0, b: int = 64) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    old_logp = -torch.rand(b, generator=g, dtype=DTYPE) * 3.0 - 0.1
    return {
        "new_logp": old_logp + 0.4 * torch.randn(b, generator=g, dtype=DTYPE),
        "old_logp": old_logp,
        "advantages": torch.randn(b, generator=g, dtype=DTYPE),
        "returns": torch.randn(b, generator=g, dtype=DTYPE),
        "values": torch.randn(b, generator=g, dtype=DTYPE),
        "entropy": torch.rand(b, generator=g, dtype=DTYPE) * 2.0,
    }


def test_returns_frozen_dataclass_with_scalar_components():
    out = ppo_loss(**_batch())
    assert isinstance(out, PPOLossOutput)
    for f in dataclasses.fields(out):
        assert getattr(out, f.name).shape == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.total = torch.tensor(0.0)  # type: ignore[misc]


def test_identical_policies_give_negative_mean_advantage():
    b = _batch()
    b["new_logp"] = b["old_logp"].clone()
    out = ppo_loss(**b)
    torch.testing.assert_close(out.policy_loss, -b["advantages"].mean())
    assert out.clip_fraction.item() == 0.0
    assert abs(out.approx_kl.item()) < 1e-12


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_formula(seed: int):
    b = _batch(seed)
    clip_eps, vf_coef, ent_coef = 0.2, 0.5, 0.05
    out = ppo_loss(**b, clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef)
    pg, vf, ent, total = _reference_ppo(
        b["new_logp"], b["old_logp"], b["advantages"], b["returns"], b["values"], b["entropy"],
        clip_eps, vf_coef, ent_coef,
    )  # fmt: skip
    torch.testing.assert_close(out.policy_loss, pg)
    torch.testing.assert_close(out.value_loss, vf)
    torch.testing.assert_close(out.entropy, ent)
    torch.testing.assert_close(out.total, total)


def _single(ratio: float, advantage: float, clip_eps: float = 0.2):
    old_logp = torch.tensor([-1.0], dtype=DTYPE)
    new_logp = (old_logp + torch.log(torch.tensor(ratio, dtype=DTYPE))).requires_grad_()
    zeros = torch.zeros(1, dtype=DTYPE)
    out = ppo_loss(
        new_logp, old_logp, torch.tensor([advantage], dtype=DTYPE), zeros, zeros, zeros,
        clip_eps=clip_eps, vf_coef=0.0, ent_coef=0.0,
    )  # fmt: skip
    out.policy_loss.backward()
    assert new_logp.grad is not None
    return out, new_logp.grad.item()


@pytest.mark.parametrize(
    ("ratio", "advantage", "expected_loss"),
    [
        (1.5, 1.0, -1.2),  # good action already made much likelier: objective capped at 1+eps
        (0.5, -1.0, 0.8),  # bad action already made much rarer: objective capped at 1-eps
    ],
)
def test_clipped_region_caps_objective_and_has_zero_gradient(ratio, advantage, expected_loss):
    out, grad = _single(ratio, advantage)
    assert out.policy_loss.item() == pytest.approx(expected_loss)
    assert grad == 0.0
    assert out.clip_fraction.item() == 1.0


@pytest.mark.parametrize(
    ("ratio", "advantage", "expected_loss"),
    [
        (0.5, 1.0, -0.5),  # good action made rarer: unclipped term is the pessimistic one
        (1.5, -1.0, 1.5),  # bad action made likelier: unclipped term is the pessimistic one
    ],
)
def test_moves_away_from_improvement_are_not_clipped(ratio, advantage, expected_loss):
    out, grad = _single(ratio, advantage)
    assert out.policy_loss.item() == pytest.approx(expected_loss)
    assert grad != 0.0


@pytest.mark.parametrize(("advantage", "sign"), [(1.0, -1.0), (-1.0, 1.0)])
def test_gradient_descent_raises_logp_of_positive_advantage_actions(advantage, sign):
    # At ratio=1 (first epoch), d(loss)/d(new_logp) = -A: descent increases logp iff A > 0.
    _, grad = _single(1.0, advantage)
    assert grad == pytest.approx(sign * abs(advantage))


def test_gradient_inside_trust_region_matches_ratio_times_advantage():
    _, grad = _single(1.1, 2.0)
    assert grad == pytest.approx(-1.1 * 2.0)


def test_loss_gradients_do_not_flow_into_rollout_tensors():
    b = _batch()
    for k in b:
        b[k] = b[k].clone().requires_grad_()
    ppo_loss(**b, old_values=b["values"].detach().clone().requires_grad_()).total.backward()
    assert b["new_logp"].grad is not None
    assert b["values"].grad is not None
    assert b["old_logp"].grad is None
    assert b["advantages"].grad is None
    assert b["returns"].grad is None


def test_entropy_bonus_lowers_the_loss():
    b = _batch()
    b["entropy"] = b["entropy"].clone().requires_grad_()
    ent_coef = 0.05
    ppo_loss(**b, ent_coef=ent_coef).total.backward()
    assert b["entropy"].grad is not None
    torch.testing.assert_close(
        b["entropy"].grad, torch.full_like(b["entropy"], -ent_coef / len(b["entropy"]))
    )


def test_value_clipping_uses_pessimistic_value_error():
    zeros = torch.zeros(1, dtype=DTYPE)
    # old V = 0, new V = 1, target = 1: unclipped error 0, clipped V = 0.2 -> error 0.64.
    out = ppo_loss(
        zeros, zeros, zeros, returns=torch.ones(1, dtype=DTYPE), values=torch.ones(1, dtype=DTYPE),
        entropy=zeros, old_values=zeros, clip_eps=0.2,
    )  # fmt: skip
    assert out.value_loss.item() == pytest.approx(0.5 * 0.64)
    out_no_clip = ppo_loss(
        zeros, zeros, zeros, returns=torch.ones(1, dtype=DTYPE), values=torch.ones(1, dtype=DTYPE),
        entropy=zeros,
    )  # fmt: skip
    assert out_no_clip.value_loss.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------- normalize


def test_normalized_minibatch_has_zero_mean_unit_std():
    adv = torch.randn(256, generator=torch.Generator().manual_seed(0), dtype=DTYPE) * 5 + 3
    out = normalize_advantages(adv)
    assert out.mean().item() == pytest.approx(0.0, abs=1e-10)
    assert out.std().item() == pytest.approx(1.0, rel=1e-6)


def test_normalizes_rollout_over_all_steps_and_envs():
    g = torch.Generator().manual_seed(1)
    adv = torch.randn(16, 4, generator=g, dtype=DTYPE) + torch.arange(16, dtype=DTYPE)[:, None]
    out = normalize_advantages(adv)
    assert out.shape == adv.shape
    torch.testing.assert_close(out, (adv - adv.mean()) / (adv.std() + 1e-8))


def test_constant_advantages_stay_finite():
    out = normalize_advantages(torch.full((32,), 0.7, dtype=DTYPE))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros(32, dtype=DTYPE))


def test_does_not_modify_the_rollout_buffer():
    adv = torch.randn(4, 8, generator=torch.Generator().manual_seed(2), dtype=DTYPE)
    before = adv.clone()
    normalize_advantages(adv)
    torch.testing.assert_close(adv, before)
