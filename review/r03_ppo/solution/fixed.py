"""PPO loss + GAE for the RL fine-tuning pipeline (reviewed and fixed).

Same public API as `pr.py`. Lines marked `# FIX(Bn)` map to the answer key in REVIEW.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PPOLossOutput:
    """All PPO loss components. `total` is what you call `.backward()` on."""

    total: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def _check_same_shape(ref_name: str, ref: torch.Tensor, **others: torch.Tensor | None) -> None:
    # A [B, 1] value head against [B] returns silently broadcasts to [B, B]; fail loudly instead.
    for name, t in others.items():
        if t is not None and t.shape != ref.shape:
            raise ValueError(
                f"{name} has shape {tuple(t.shape)}, expected {tuple(ref.shape)} like {ref_name}"
            )


@torch.no_grad()  # FIX(B6): advantages/returns are fixed targets; never build a graph here.
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation over a time-major rollout.

    Args:
        rewards: `[T]` or `[T, N]` rewards r_t.
        values: same shape, critic estimates V(s_t).
        dones: same shape, 1 where the episode terminated at step t (no bootstrap past it).
        last_value: `[]` or `[N]`, V(s_T) for the state after the final step.
        gamma: discount factor.
        lam: GAE lambda (0 -> one-step TD, 1 -> Monte Carlo).

    Returns:
        (advantages, returns), both shaped like `rewards`.
    """
    _check_same_shape("rewards", rewards, values=values, dones=dones)
    if last_value.shape != rewards.shape[1:]:
        raise ValueError(f"last_value must have shape {tuple(rewards.shape[1:])}")

    values = values.detach()  # FIX(B6)
    T = rewards.shape[0]
    not_dones = 1.0 - dones.to(rewards.dtype)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])

    for t in reversed(range(T)):  # FIX(B1): t runs T-1 .. 0 (was T .. 1 -> IndexError)
        next_value = last_value if t == T - 1 else values[t + 1]
        # FIX(B5): a terminal step must not bootstrap from the next (new-episode) state.
        delta = rewards[t] + gamma * next_value * not_dones[t] - values[t]
        last_gae = delta + gamma * lam * not_dones[t] * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    entropy: torch.Tensor,
    old_values: torch.Tensor | None = None,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> PPOLossOutput:
    """Clipped PPO objective (Schulman et al. 2017), as a loss to minimize.

    Args:
        new_logp: `[B]` log pi_theta(a|s) under the current policy (requires grad).
        old_logp: `[B]` log pi_old(a|s) recorded at rollout time.
        advantages: `[B]` advantage estimates (usually normalized).
        returns: `[B]` value targets from `compute_gae`.
        values: `[B]` current critic predictions (requires grad).
        entropy: `[B]` per-sample policy entropy.
        old_values: `[B]` critic predictions at rollout time; enables value clipping.
        clip_eps: PPO clip range epsilon.
        vf_coef: weight of the value loss.
        ent_coef: weight of the entropy bonus.
    """
    if clip_eps <= 0:
        raise ValueError(f"clip_eps must be > 0, got {clip_eps}")
    _check_same_shape(
        "new_logp",
        new_logp,
        old_logp=old_logp,
        advantages=advantages,
        returns=returns,
        values=values,
        entropy=entropy,
        old_values=old_values,
    )

    # FIX(B6): rollout-time quantities are constants of the objective.
    old_logp = old_logp.detach()
    advantages = advantages.detach()
    returns = returns.detach()
    if old_values is not None:
        old_values = old_values.detach()

    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()  # FIX(B3): pi/pi_old = exp(logp - logp_old), not logp / logp_old
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()  # FIX(B4): pessimistic bound is the min

    value_err = (values - returns).pow(2)
    if old_values is not None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        value_err = torch.max(value_err, (values_clipped - returns).pow(2))
    value_loss = 0.5 * value_err.mean()

    entropy_mean = entropy.mean()
    # FIX(B7): entropy is a bonus -> subtract it from the loss.
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean

    with torch.no_grad():
        # k3 estimator of KL(old || new): unbiased and always >= 0.
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_eps).to(ratio.dtype).mean()

    return PPOLossOutput(
        total=total,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy_mean,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
    )


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Whiten advantages to zero mean / unit std over the whole batch (all dims)."""
    if advantages.numel() < 2:
        return advantages - advantages.mean()  # std of a single element is undefined (NaN)
    mean = advantages.mean()  # FIX(B2): global stats; dim=1 crashes on [B], per-step on [T, N]
    std = advantages.std()
    return (advantages - mean) / (std + eps)  # FIX(B8): out-of-place; don't mutate the buffer
