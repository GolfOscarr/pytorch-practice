# r04 — 3D-parallelism planner: answer key

## Bugs

| ID | Severity | Location | What's wrong | Impact |
|----|----------|----------|--------------|--------|
| B1 | Critical | pr.py:145 | World-size check is `>` instead of `!=`: a layout that uses only part of the job's GPUs passes | silently wrong (idle GPUs, or ranks with no coords → crash at group setup) |
| B2 | High | pr.py:145-157 (missing check), pr.py:216 | No `tp <= gpus_per_node` check, and `plan()` enumerates TP over divisors of the *world* size | perf (TP all-reduces over InfiniBand instead of NVLink) |
| B3 | High | pr.py:245-247 | Rank mapping makes **dp** the fastest-varying axis, so TP groups are strided across nodes, contradicting the docstring | perf (silently slow, often 2-5x) |
| B4 | High | pr.py:152-153 (missing check) | Only `n_heads % tp` is checked; with GQA `n_kv_heads % tp` is not | crash at model build (or silently wrong KV sharding) |
| B5 | Medium | pr.py:145-157 (missing check) | `n_layers % pp` never checked; `layers_per_stage` becomes fractional | crash at model build / unbalanced pipeline |
| B6 | Critical | pr.py:115, pr.py:154 | `grad_accum = global_batch // micro_batch` ignores dp; divisibility checked against `micro_batch` instead of `micro_batch * dp` | silently wrong (effective batch dp× too large, LR schedule off; samples dropped by `//`; bubble warning suppressed) |
| B7 | Medium | pr.py:176 | Optimizer state always divided by dp, even for `zero_stage=0` | silently wrong (memory under-estimated → OOM) |
| B8 | Medium | pr.py:39 | Adam costs 8 B/param in the estimate; mixed precision also keeps an fp32 master copy → 12 B (16 B total with bf16 params+grads) | silently wrong (memory under-estimated by ~25% of model states → OOM; planner picks layouts that don't fit) |

### B1: world size must match exactly
A launcher starts `num_nodes * gpus_per_node` processes and every one of them needs a
(dp, pp, tp) coordinate. `tp*pp*dp < world_size` is not "fine, we have spare GPUs": the extra
ranks either fail inside `rank_to_coords` (out of range) or are silently left out of every
process group while the job burns their allocation.

```python
if cfg.world_size > cluster.world_size:          # buggy
if cfg.world_size != cluster.world_size:         # fixed
```
Caught by `test_layout_using_fewer_gpus_than_launched_is_rejected`, `test_all_violations_are_reported_together`.

### B2: tensor parallelism must stay inside a node
TP does two all-reduces of the full activation per layer in forward and two in backward. It
only works on NVLink (~900 GB/s intra-node), not on inter-node IB (~50 GB/s per GPU).
Rule: `tp | gpus_per_node`; spread across nodes with pp/dp. The planner's tie-break
"fewer pipeline stages" then happily picks tp=16, pp=1 over tp=8, pp=2.

```python
# buggy: no check; plan() does
for tp in _divisors(world):
# fixed
if cfg.tp > cluster.gpus_per_node or cluster.gpus_per_node % cfg.tp:
    errors.append(...)
for tp in _divisors(cluster.gpus_per_node):
```
Caught by `test_tensor_parallel_degree_larger_than_a_node_is_rejected`, `test_plan_keeps_tensor_parallelism_inside_a_node[mha]`.

### B3: TP must be the fastest-varying rank axis
Ranks are numbered node-major (ranks 0-7 = node 0). Megatron's order is tp (fastest) → dp → pp.
With dp fastest, TP group for (dp=0, pp=0) on tp=8, pp=2, dp=2 is `[0, 4, 8, ..., 28]` — all
four nodes. The groups still *partition* the ranks, so a "covers every rank once" test passes;
the bug only shows as terrible step time.

```python
# buggy
dp = rank % cfg.dp
pp = (rank // cfg.dp) % cfg.pp
tp = rank // (cfg.dp * cfg.pp)
# fixed
tp = rank % cfg.tp
dp = (rank // cfg.tp) % cfg.dp
pp = rank // (cfg.tp * cfg.dp)
```
Caught by `test_every_tensor_parallel_group_lives_on_one_node[*]`, `test_consecutive_ranks_are_tensor_parallel_peers`, `test_plan_keeps_tensor_parallelism_inside_a_node[*]`.

### B4: KV heads must divide by tp (GQA)
Column-parallel attention gives each TP rank `n_heads/tp` query heads **and** `n_kv_heads/tp`
KV heads. With 64 Q heads / 4 KV heads, tp=8 splits Q fine but cannot split K/V (Megatron:
`num_query_groups % tp == 0`). Replicating KV heads is possible but must be an explicit design choice.

```python
# fixed (added)
if model.kv_heads % cfg.tp:
    errors.append(f"n_kv_heads={model.kv_heads} is not divisible by tp={cfg.tp}")
```
Caught by `test_kv_heads_must_split_evenly_across_tp`.

### B5: layers must divide by pp
80 layers on pp=3 means stages of 26/27/27 at best (the estimate even uses 26.67); most stacks
assert equal stages, and unequal ones make the slowest stage set the pace for the whole pipeline.

```python
# fixed (added)
if model.n_layers % cfg.pp:
    errors.append(f"n_layers={model.n_layers} is not divisible by pp={cfg.pp}")
```
Caught by `test_layers_must_split_evenly_across_pipeline_stages`, `test_all_violations_are_reported_together`.

### B6: grad_accum must divide by dp too
`global_batch = micro_batch * dp * grad_accum`. Each of the dp replicas runs its own
micro-batches, so the PR makes every step consume `dp × global_batch` samples (changes the
optimization, the LR-warmup-in-samples math and the epoch length). The divisibility check uses the
wrong modulus, so `//` can also drop samples silently (100 / (2*4) → 12 steps = 96 samples). As
a knock-on effect, the "pipeline bubble" warning compares an inflated grad_accum against pp, so
it is suppressed in cases where it should fire.

```python
return self.global_batch // self.micro_batch                     # buggy
if cfg.global_batch % cfg.micro_batch:                           # buggy
return self.global_batch // (self.micro_batch * self.dp)         # fixed
if cfg.global_batch % (cfg.micro_batch * cfg.dp):                # fixed
```
Caught by `test_one_optimizer_step_consumes_exactly_the_global_batch[*]`, `test_grad_accum_for_known_layout`,
`test_global_batch_not_divisible_by_micro_batch_times_dp_is_rejected`, `test_warns_when_pipeline_cannot_be_filled`.

### B7: ZeRO-0 does not shard optimizer state
ZeRO stage k shards: 1 → optimizer state, 2 → + gradients, 3 → + parameters, each over dp.
Plain DDP (stage 0) replicates everything. The PR handles params/grads by stage but always
divides the optimizer term.

```python
optimizer_shard = cfg.dp  # buggy
optimizer_shard = cfg.dp if cfg.zero_stage >= 1 else 1  # fixed
```
Caught by `test_model_state_memory_matches_hand_computation[0]`, `test_each_zero_stage_strictly_reduces_memory_when_dp_above_one`.

### B8: mixed-precision Adam is 16 bytes/param, not 12
bf16 weights (2) + bf16 grads (2) + fp32 master weights (4) + fp32 m (4) + fp32 v (4) = 16 B
(ZeRO paper's "K = 12" for the optimizer part). Forgetting the master copy under-counts by 4 B/param:
for 70B on tp*pp=16 that is ~17.6 GB/GPU at ZeRO-0. It also changes `plan()`'s answer (picks tp=8,
pp=1, which really needs ~73 GB > 72 GB usable).

```python
OPTIMIZER_BYTES = 8  # buggy: fp32 m, v only
OPTIMIZER_BYTES = 12  # fixed: fp32 master + m + v
```
Caught by `test_model_state_memory_matches_hand_computation[*]`, `test_unsharded_adam_costs_16_bytes_per_param`,
`test_plan_for_70b_on_4x8_picks_tp8_pp2_dp2`.

Bonus finding (same class as B4/B5, not planted): `ffn_hidden % tp` is not checked either. A
column-parallel MLP splits `ffn_hidden` across TP ranks; fixed.py checks it. Vocab size has the
same constraint unless the embedding is padded to a multiple of tp.

## Production improvements

1. **Model-spec self-validation** (applied in fixed.py): `ModelSpec.__post_init__` rejects
   `hidden % n_heads` and `n_heads % n_kv_heads` so invalid shapes fail at construction, not deep
   inside `validate`.
2. **Observable planner** (applied in fixed.py): log *why* each candidate is rejected; without it
   "no layout fits" is undebuggable.
   ```python
   logger.debug("skip tp=%d pp=%d: needs %.1f GB", tp, pp, mem.total_gb)
   ```
3. **Return a ranked plan, not one answer**: return `list[tuple[ParallelConfig, MemoryEstimate]]`
   sorted by an explicit cost model (bubble fraction `(pp-1)/(grad_accum+pp-1)`, TP comm volume,
   DP all-reduce bytes) so the launcher can print the top 3 and the user can override.
   ```python
   def bubble_fraction(c):
       return (c.pp - 1) / (c.grad_accum + c.pp - 1)
   ```
4. **Make the topology explicit**: the rank → node mapping is assumed to be `rank // gpus_per_node`.
   Take `local_world_size` / hostfile from the launcher (`torchrun` sets `LOCAL_WORLD_SIZE`) and
   assert the assumption, or build groups with `torch.distributed.device_mesh.init_device_mesh(
   "cuda", (pp, dp, tp), mesh_dim_names=("pp", "dp", "tp"))`, which uses the same tp-innermost order.
5. **Memory model fidelity**: add a safety margin for CUDA context / NCCL buffers / fragmentation
   (~3-5 GB), the embedding/LM-head imbalance on the first/last pipeline stage, the logits buffer
   (`b*s*vocab*4/tp` bytes), and a `recompute: Literal["none","selective","full"]` knob instead of
   the hard-coded 34.
6. **Typed errors**: raise a `ConfigError` carrying `errors: list[str]` (not only a joined string)
   so callers and tests can inspect individual violations; use `warnings.warn(..., stacklevel=2)`
   or a structured result instead of a bare `list[str]`.
7. **Property-based tests**: hypothesis over (nodes, gpus, tp, pp, dp) asserting bijection of
   `rank_to_coords`, partition of groups, `micro*dp*accum == global` for every accepted config.

Assumptions worth stating in a review:
- The activation constant (34 bytes per token per hidden per layer, divided by tp) is the
  Korthikanti et al. figure for TP **with sequence parallelism and selective recompute**. Without
  sequence parallelism it is `s·b·h·(10 + 24/t)`, so only part of it shrinks with tp.
- In DeepSpeed, ZeRO stage 2/3 doesn't combine with its pipeline engine (stage 1 only), so a
  real launcher should reject `pp > 1` with `zero_stage >= 2` for that backend.

## Interview talking points

- **Axis order is a performance contract**: innermost = highest-bandwidth comm. TP (per-layer
  all-reduce) on NVLink, then DP (one gradient all-reduce per step, overlappable), PP outermost
  (point-to-point activations, cheapest). A mapping can be a perfect partition and still be wrong.
- **Batch identity**: `global = micro × dp × accum`. Always validate divisibility with the full
  product; integer division silently drops samples and changes the effective LR schedule.
- **Memory napkin math**: 16 B/param for mixed-precision Adam (2+2+4+4+4); ZeRO-1/2/3 shard
  optimizer / +grads / +params over dp; model parallelism (tp×pp) divides everything. 70B ×
  16 B = 1.13 TB of model state — that's why it needs ≥16-way sharding on 80 GB cards.
- **Shape constraints of TP/PP**: heads *and* KV heads divisible by tp, layers by pp, tp ≤ GPUs per
  node; and keep `grad_accum ≥ pp` (ideally ≫ pp) or the 1F1B bubble dominates.
- **Validation design**: collect all violations and report them at once; planners should be
  explainable (log rejections) because a wrong "valid" answer costs a multi-node job.
