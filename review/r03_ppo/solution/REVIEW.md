# r03_ppo: answer key

PR: "Add PPO loss + GAE for our RL fine-tuning pipeline"

## Bugs

| ID | Severity | Location | What's wrong | Impact |
|----|----------|----------|--------------|--------|
| B1 | High | pr.py:67 | GAE loop is `range(T, 0, -1)`: it starts at `t = T` (out of range) and never visits `t = 0` | crash (IndexError on every call) |
| B2 | High | pr.py:133-134 | Normalization stats taken over `dim=1`: crashes on a flat `[B]` minibatch, and on `[T, N]` it normalizes each timestep across envs | crash / silently wrong |
| B3 | Critical | pr.py:103 | Probability ratio computed as `new_logp / old_logp` instead of `exp(new_logp - old_logp)` | silently wrong |
| B4 | Critical | pr.py:106 | Clipped surrogate uses `torch.max` (optimistic bound) instead of `torch.min` (pessimistic bound) | silently wrong |
| B5 | High | pr.py:69 | TD error bootstraps `gamma * V(s_{t+1})` even when step `t` is terminal: value leaks across episode boundaries | silently wrong |
| B6 | High | pr.py:69-73, 103-115 | Rollout tensors are not treated as constants: `compute_gae` builds a graph through `values`, and `ppo_loss` never detaches `old_logp` / `advantages` / `returns` | silently wrong (gradients reach the targets) |
| B7 | High | pr.py:115 | Entropy bonus is *added* to the loss, so minimizing the loss lowers entropy | silently wrong |
| B8 | Medium | pr.py:135 | `sub_` / `div_` normalize the caller's tensor in place | silently wrong (corrupts the rollout buffer; production aliasing bug) |

### B1: GAE loop starts out of range

GAE is a backward recursion `A_t = delta_t + gamma*lam*(1-d_t)*A_{t+1}`, so `t` must run
`T-1, T-2, ..., 0`. `range(T, 0, -1)` yields `T, ..., 1`. The first iteration takes the
`values[t + 1]` branch with `t = T` and raises IndexError. If the indexing had happened to work,
step 0 would still be skipped, leaving `A_0 = 0`.

```python
for t in range(T, 0, -1):            # buggy
for t in reversed(range(T)):         # fixed: T-1 .. 0
```

Caught by: every `test_gae_*` test (e.g. `test_gae_matches_hand_computed_example_with_episode_boundary`).

### B2: normalization over the wrong dimension

Advantage whitening is a per-batch scale fix. The mean and std should come from the whole batch
being optimized, which means every element. With `dim=1`:
- a flat `[B]` minibatch (the common case after flattening `[T, N]`) raises IndexError;
- a `[T, N]` rollout gets normalized *per timestep across envs*. This removes real differences
  between timesteps, and with `N = 1` the std is NaN.

The tests expect the default `.std()` (sample, n−1). `std(unbiased=False)` is equally valid
whitening but differs slightly, so it fails the tight tolerances.

```python
mean = advantages.mean(dim=1, keepdim=True)  # buggy
std = advantages.std(dim=1, keepdim=True)
mean, std = advantages.mean(), advantages.std()  # fixed: global stats
```

Caught by: `test_normalized_minibatch_has_zero_mean_unit_std`,
`test_normalizes_rollout_over_all_steps_and_envs`, `test_constant_advantages_stay_finite`.

### B3: ratio of log-probs instead of ratio of probs

PPO uses `r = pi(a|s) / pi_old(a|s) = exp(log pi - log pi_old)`. Dividing log-probs gives a
meaningless number. Its gradient `1/old_logp` is **negative**, because log-probs are <= 0, so the
update moves in the wrong direction. It is especially hard to spot because when
`new_logp == old_logp`, which is the first minibatch of every epoch, `new/old = 1 = exp(0)`. So the
"policy loss starts at -mean(adv)" sanity check in the PR description passes.

```python
ratio = new_logp / old_logp  # buggy
ratio = torch.exp(new_logp - old_logp)  # fixed
```

Caught by: `test_matches_reference_formula`, `test_gradient_descent_raises_logp_of_positive_advantage_actions`,
`test_clipped_region_caps_objective_and_has_zero_gradient`, `test_gradient_inside_trust_region_matches_ratio_times_advantage`.

### B4: `max` instead of `min` in the clipped surrogate

`L^CLIP = E[min(r*A, clip(r, 1-eps, 1+eps)*A)]` is a pessimistic lower bound. It ignores
changes that would make the objective look *better* than the trust region allows, and keeps the
full penalty for changes that make it worse. With `max`, the optimistic term always wins. For
A > 0 the loss keeps rewarding a ratio far above 1+eps, so clipping does nothing to stop the
policy from running away. For a ratio moving the wrong way, the update stalls on the flat clipped
term.

```python
policy_loss = -torch.max(surr1, surr2).mean()  # buggy
policy_loss = -torch.min(surr1, surr2).mean()  # fixed
```

Caught by: `test_clipped_region_caps_objective_and_has_zero_gradient`,
`test_moves_away_from_improvement_are_not_clipped`, `test_matches_reference_formula`.

### B5: bootstrapping across episode boundaries

When step `t` ends an episode, `s_{t+1}` in the buffer is the first state of a *new* episode.
The correct TD error is `delta_t = r_t - V(s_t)`. The PR masks the GAE carry with `not_dones[t]`
but not the bootstrap term in `delta`, so the next episode's value leaks into the current one.
Everything still runs, and on dense-reward tasks it looks like it trains. It is a classic bug that
interviewers ask about.

```python
delta = rewards[t] + gamma * next_value - values[t]  # buggy
delta = rewards[t] + gamma * next_value * not_dones[t] - values[t]  # fixed
```

Caught by: `test_gae_matches_hand_computed_example_with_episode_boundary`,
`test_gae_terminal_step_ignores_value_of_next_episode`.

### B6: gradients flow into rollout-time quantities

In PPO, `old_logp`, `advantages`, `returns` (and `old_values`) are *constants* of the surrogate
objective. If the caller computes them with grad enabled, which is easy in an RL fine-tuning loop
where the value head shares the LM trunk, then:
- the policy loss `-A * r` backpropagates through `A = f(values)` into the critic, pushing it to
  change advantages instead of fitting returns;
- the value loss `(V - (A + V))^2` moves its own target;
- gradients reach `old_logp`, pulling the "old" policy along.

Nothing crashes. The losses just optimize a different objective.

The key (and the tests) require the functions to enforce this themselves, defensively. Some
codebases (e.g. CleanRL) leave it to the caller and document that rollout tensors must be
detached; that's a weaker contract, and it fails these tests.

```python
# buggy: no detach anywhere
@torch.no_grad()                      # fixed: compute_gae builds no graph
def compute_gae(...): ...
old_logp = old_logp.detach(); advantages = advantages.detach(); returns = returns.detach()
```

Caught by: `test_loss_gradients_do_not_flow_into_rollout_tensors`,
`test_gae_outputs_are_constants_for_the_optimizer`.

### B7: entropy bonus has the wrong sign

The objective is `L = L_clip - c1*L_vf + c2*H`. Turned into a loss to minimize, that becomes
`-L_clip + c1*L_vf - c2*H`. Adding `+ent_coef*H` *penalizes* exploration and speeds up entropy
collapse / mode collapse, which is the exact failure the term is there to prevent. Note that
`entropy` in the returned dataclass is still correct, so logs look fine.

```python
total = policy_loss + vf_coef * value_loss + ent_coef * entropy_mean  # buggy
total = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean  # fixed
```

Caught by: `test_entropy_bonus_lowers_the_loss`, `test_matches_reference_formula`.

### B8: in-place normalization mutates the caller's buffer

`advantages.sub_(mean).div_(std + eps)` writes into the tensor the caller passed in, usually a
slice of the rollout buffer. Later consumers see whitened values instead of raw advantages:
logging, `returns = advantages + values` recomputed downstream, or a second PPO epoch that
re-normalizes a view. If the tensor is a leaf that requires grad, the call raises an error.
A helper that looks pure must not alias its input.

```python
return advantages.sub_(mean).div_(std + eps)  # buggy
return (advantages - mean) / (std + eps)  # fixed
```

Caught by: `test_does_not_modify_the_rollout_buffer`.

## Production improvements

1. **Shape validation** *(applied in fixed.py)*. A `[B, 1]` value head against `[B]` returns
   broadcasts `(values - returns)` to `[B, B]` and silently produces a wrong value loss.
   Validate shapes up front:
   ```python
   _check_same_shape("new_logp", new_logp, values=values, returns=returns, ...)  # raises ValueError
   ```
2. **Unbiased, non-negative KL estimate** *(applied)*. `mean(old - new)` (k1) is noisy and can be
   negative, which makes it a poor early-stopping signal. Use k3 instead:
   ```python
   approx_kl = ((ratio - 1) - log_ratio).mean()
   ```
3. **`@torch.no_grad()` on GAE + argument validation** *(applied)*. Saves memory (no graph over T
   steps), makes the "targets are constants" contract explicit, and rejects `clip_eps <= 0` and a
   mismatched `last_value` shape. Handle the single-element std case (NaN) in
   `normalize_advantages`.
4. **Distinguish `terminated` from `truncated`** *(not applied)*. Time-limit truncation should still
   bootstrap from `V(s_{t+1})`. Only true termination should cut it. Accept an optional
   `truncated` mask and pass the final observation's value:
   ```python
   delta = r[t] + gamma * next_v * (1 - terminated[t]) - v[t]
   carry = gamma * lam * (1 - (terminated[t] | truncated[t])) * last_gae
   ```
5. **Hyperparameters as a frozen dataclass config** *(not applied)*. This keeps trainer call sites
   readable and lets you log the config with the run:
   ```python
   @dataclass(frozen=True)
   class PPOConfig:
       clip_eps: float = 0.2
       vf_coef: float = 0.5
       ent_coef: float = 0.01
       vf_clip_eps: float | None = None  # separate value clip range (often != policy clip)
   ```
6. **Separate value-clip range** *(not applied)*. Reusing `clip_eps` for value clipping ties a
   probability-ratio scale to a reward scale. Make it its own knob (see `vf_clip_eps`) or drop
   value clipping, which several ablations found neutral or harmful.
7. **Avoid host syncs in the training loop** *(not applied)*. Keep the diagnostics as tensors, as
   the dataclass already does, and call `.item()` only at the logging cadence. On MPS/CUDA, a
   `.item()` per minibatch serializes the device. For long rollouts, a vectorized/JIT'd GAE (or
   running it on CPU) beats a Python loop of tiny device kernels.
8. **Masks for RL fine-tuning of LMs** *(not applied)*. Token-level PPO needs a `mask` (padding /
   prompt tokens) in every mean: `(x * mask).sum() / mask.sum().clamp_min(1)`. A plain `.mean()`
   weights padding.

## Interview talking points

- **The ratio lives in log space for numerical reasons, but the ratio itself is a probability
  ratio**: `exp(logp - logp_old)`. Then point out the trap: at the start of each epoch the ratio
  is exactly 1 under *both* formulas, so "loss looks right on step 1" proves nothing. Test with
  `ratio != 1` and check the gradient sign.
- **Clipping = pessimism**: `min(r*A, clip(r)*A)` takes the lower bound, so the policy gets no
  credit for moving past `1 +/- eps` in the direction that helps, but still pays fully for moving
  the wrong way. In the clipped region the gradient is exactly zero, and that is the test to write.
- **GAE and `dones`**: a terminal step must zero *both* the bootstrap in `delta_t` and the carry
  of `A_{t+1}`. Mention terminated vs truncated as the follow-up an interviewer is looking for.
- **What is a constant in the objective**: old log-probs, advantages and returns come from the
  rollout. Detach them (or compute them under `no_grad`), or the critic learns to game its own
  targets. Test it with `requires_grad=True` inputs and assert `.grad is None`.
- **Loss-sign hygiene**: write the *objective* first (`L_clip - c1 L_vf + c2 H`), then negate it
  once. Sign errors on entropy or the surrogate train without crashing, so they need explicit
  gradient-direction tests.
