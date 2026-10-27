"""PPO loss + GAE for the RL fine-tuning pipeline.

## Summary
Adds the core PPO math we need for RL fine-tuning, as pure tensor functions with no gym /
environment dependency, so the trainer can call them on whatever rollout buffer it builds.

## Changes
- `compute_gae(rewards, values, dones, last_value, gamma, lam)`: Generalized Advantage
  Estimation (Schulman et al. 2016). Works on `[T]` or `[T, N]` rollouts (time-major, N parallel
  envs / sequences). `dones[t] = 1` means the episode ended at step `t`. Returns
  `(advantages, returns)` with `returns = advantages + values` as the value-function target.
- `ppo_loss(...)`: clipped-surrogate policy loss, optional clipped value loss (pass
  `old_values`), and entropy bonus. Returns a frozen `PPOLossOutput` dataclass so the trainer can
  log every component (plus approx KL and clip fraction) without recomputing anything.
- `normalize_advantages(advantages)`: standard per-batch advantage whitening.

## Testing
Ran it inside the fine-tuning loop on a small run; losses are finite and the policy loss starts
at `-mean(advantage)` on the first epoch as expected. Unit tests in `test_pr.py`.
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
    T = rewards.shape[0]
    not_dones = 1.0 - dones.to(rewards.dtype)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])

    for t in range(T, 0, -1):
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
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
    ratio = new_logp / old_logp
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.max(surr1, surr2).mean()

    value_err = (values - returns).pow(2)
    if old_values is not None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        value_err = torch.max(value_err, (values_clipped - returns).pow(2))
    value_loss = 0.5 * value_err.mean()

    entropy_mean = entropy.mean()
    total = policy_loss + vf_coef * value_loss + ent_coef * entropy_mean

    with torch.no_grad():
        approx_kl = (old_logp - new_logp).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()

    return PPOLossOutput(
        total=total,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy_mean,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
    )


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Whiten advantages to zero mean / unit std over the batch."""
    mean = advantages.mean(dim=1, keepdim=True)
    std = advantages.std(dim=1, keepdim=True)
    return advantages.sub_(mean).div_(std + eps)
