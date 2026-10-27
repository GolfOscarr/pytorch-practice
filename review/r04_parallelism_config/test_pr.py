import pytest

from review.r04_parallelism_config.pr import (
    ClusterSpec,
    ConfigError,
    ModelSpec,
    ParallelConfig,
    build_groups,
    estimate_memory,
    plan,
    rank_to_coords,
    validate,
)

GB = 1e9

# 4 nodes x 8 x 80 GB, and a Llama-3-70B-shaped model (GQA with 8 KV heads).
CLUSTER = ClusterSpec(num_nodes=4, gpus_per_node=8, gpu_mem_gb=80.0)
LLAMA_70B = ModelSpec(
    n_layers=80, hidden=8192, n_heads=64, ffn_hidden=28672, vocab=128256, seq_len=4096, n_kv_heads=8
)


def cfg(tp: int, pp: int, dp: int, micro_batch: int = 1, global_batch: int = 512, zero: int = 1):
    return ParallelConfig(
        tp=tp, pp=pp, dp=dp, micro_batch=micro_batch, global_batch=global_batch, zero_stage=zero
    )


# --- model / config basics ----------------------------------------------------------------------


def test_70b_param_count_is_about_70_billion():
    assert LLAMA_70B.num_params == pytest.approx(70.55e9, rel=1e-3)


def test_non_positive_degrees_are_rejected():
    with pytest.raises(ConfigError):
        cfg(tp=0, pp=1, dp=32)


# --- validate: scenarios that must pass ---------------------------------------------------------


@pytest.mark.parametrize(
    ("tp", "pp", "dp"), [(8, 1, 4), (8, 2, 2), (4, 4, 2), (8, 4, 1), (2, 2, 8)]
)
def test_valid_layouts_on_4x8_cluster_are_accepted(tp, pp, dp):
    validate(CLUSTER, LLAMA_70B, cfg(tp, pp, dp))


# --- validate: scenarios that must fail ---------------------------------------------------------


def test_layout_using_fewer_gpus_than_launched_is_rejected():
    # 8 * 2 * 1 = 16 ranks on a 32-GPU job: half the GPUs would have no role.
    with pytest.raises(ConfigError, match=r"tp\*pp\*dp"):
        validate(CLUSTER, LLAMA_70B, cfg(tp=8, pp=2, dp=1))


def test_layout_using_more_gpus_than_cluster_is_rejected():
    with pytest.raises(ConfigError, match=r"tp\*pp\*dp"):
        validate(CLUSTER, LLAMA_70B, cfg(tp=8, pp=2, dp=4))


def test_tensor_parallel_degree_larger_than_a_node_is_rejected():
    with pytest.raises(ConfigError, match=r"(?i)gpus.per.node|node"):
        validate(CLUSTER, LLAMA_70B, cfg(tp=16, pp=1, dp=2))


def test_query_heads_must_split_evenly_across_tp():
    # hidden=1536 divides by 8, but 12 attention heads cannot be split between 8 ranks.
    model = ModelSpec(n_layers=16, hidden=1536, n_heads=12, ffn_hidden=4096, vocab=32000, seq_len=1)
    one_node = ClusterSpec(num_nodes=1, gpus_per_node=8, gpu_mem_gb=80.0)
    validate(one_node, model, cfg(tp=4, pp=1, dp=2, global_batch=8))
    with pytest.raises(ConfigError, match="n_heads"):
        validate(one_node, model, cfg(tp=8, pp=1, dp=1, global_batch=8))


def test_kv_heads_must_split_evenly_across_tp():
    # 64 query heads split fine over tp=8, but 4 KV heads cannot be divided between 8 ranks.
    gqa4 = ModelSpec(
        n_layers=80,
        hidden=8192,
        n_heads=64,
        ffn_hidden=28672,
        vocab=128256,
        seq_len=4096,
        n_kv_heads=4,
    )
    with pytest.raises(ConfigError, match=r"(?i)kv"):
        validate(CLUSTER, gqa4, cfg(tp=8, pp=2, dp=2))
    validate(CLUSTER, gqa4, cfg(tp=4, pp=4, dp=2))


def test_layers_must_split_evenly_across_pipeline_stages():
    three_nodes = ClusterSpec(num_nodes=3, gpus_per_node=8, gpu_mem_gb=80.0)
    with pytest.raises(ConfigError, match="n_layers"):
        validate(three_nodes, LLAMA_70B, cfg(tp=8, pp=3, dp=1, global_batch=96))
    validate(three_nodes, LLAMA_70B, cfg(tp=8, pp=1, dp=3, global_batch=96))


def test_all_violations_are_reported_together():
    with pytest.raises(ConfigError) as info:
        validate(CLUSTER, LLAMA_70B, cfg(tp=8, pp=3, dp=1))
    assert "n_layers" in str(info.value) and "tp*pp*dp" in str(info.value)


# --- batch math ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tp", "pp", "dp", "micro", "global_batch"),
    [(8, 2, 2, 1, 512), (8, 1, 4, 2, 512), (4, 4, 2, 4, 64), (2, 2, 8, 1, 1024)],
)
def test_one_optimizer_step_consumes_exactly_the_global_batch(tp, pp, dp, micro, global_batch):
    c = cfg(tp, pp, dp, micro_batch=micro, global_batch=global_batch)
    validate(CLUSTER, LLAMA_70B, c)
    assert c.micro_batch * c.dp * c.grad_accum == c.global_batch


def test_grad_accum_for_known_layout():
    assert cfg(tp=8, pp=2, dp=2, micro_batch=2, global_batch=64).grad_accum == 16


def test_global_batch_not_divisible_by_micro_batch_times_dp_is_rejected():
    # 100 % 2 == 0, but 100 samples cannot be split over 4 replicas of 2-sample micro-batches.
    with pytest.raises(ConfigError, match="global_batch"):
        validate(CLUSTER, LLAMA_70B, cfg(tp=4, pp=2, dp=4, micro_batch=2, global_batch=100))


def test_warns_when_pipeline_cannot_be_filled():
    # 2 replicas * 4-sample micro-batches * 2 steps = 16: only 2 micro-batches for 4 stages.
    warnings = validate(CLUSTER, LLAMA_70B, cfg(tp=4, pp=4, dp=2, micro_batch=4, global_batch=16))
    assert any("bubble" in w for w in warnings)
    assert validate(CLUSTER, LLAMA_70B, cfg(tp=4, pp=4, dp=2, global_batch=512)) == []


# --- rank layout --------------------------------------------------------------------------------

LAYOUTS = [(8, 2, 2), (4, 4, 2), (2, 2, 8), (8, 1, 4), (4, 2, 4)]


@pytest.mark.parametrize(("tp", "pp", "dp"), LAYOUTS)
def test_groups_partition_all_ranks(tp, pp, dp):
    c = cfg(tp, pp, dp)
    groups = build_groups(c)
    for axis, size, count in [("tp", tp, pp * dp), ("pp", pp, tp * dp), ("dp", dp, tp * pp)]:
        assert len(groups[axis]) == count
        assert all(len(g) == size for g in groups[axis])
        assert sorted(r for g in groups[axis] for r in g) == list(range(32))


@pytest.mark.parametrize(("tp", "pp", "dp"), LAYOUTS)
def test_every_tensor_parallel_group_lives_on_one_node(tp, pp, dp):
    for group in build_groups(cfg(tp, pp, dp))["tp"]:
        nodes = {rank // CLUSTER.gpus_per_node for rank in group}
        assert len(nodes) == 1, f"TP group {group} spans nodes {nodes}"


def test_consecutive_ranks_are_tensor_parallel_peers():
    c = cfg(tp=8, pp=2, dp=2)
    coords = [rank_to_coords(r, c) for r in range(8)]
    assert [x.tp for x in coords] == list(range(8))
    assert len({(x.dp, x.pp) for x in coords}) == 1


@pytest.mark.parametrize(("tp", "pp", "dp"), LAYOUTS)
def test_coords_are_a_bijection(tp, pp, dp):
    c = cfg(tp, pp, dp)
    coords = {rank_to_coords(r, c) for r in range(32)}
    assert len(coords) == 32
    assert all(0 <= x.tp < tp and 0 <= x.pp < pp and 0 <= x.dp < dp for x in coords)


def test_rank_outside_world_is_rejected():
    with pytest.raises(ValueError):
        rank_to_coords(32, cfg(tp=8, pp=2, dp=2))


# --- memory estimate ----------------------------------------------------------------------------


def _hand_estimate(model: ModelSpec, tp: int, pp: int, dp: int, zero: int) -> dict[str, float]:
    """bf16 params + bf16 grads + fp32 (master, m, v) = 2 + 2 + 12 = 16 bytes/param, sharded by
    tp*pp; ZeRO-1 shards optimizer state, ZeRO-2 also grads, ZeRO-3 also params, over dp."""
    p = model.num_params / (tp * pp)
    return {
        "params": p * 2 / (dp if zero >= 3 else 1) / GB,
        "grads": p * 2 / (dp if zero >= 2 else 1) / GB,
        "optimizer": p * 12 / (dp if zero >= 1 else 1) / GB,
    }


@pytest.mark.parametrize("zero", [0, 1, 2, 3])
def test_model_state_memory_matches_hand_computation(zero):
    est = estimate_memory(LLAMA_70B, cfg(tp=8, pp=2, dp=2, zero=zero))
    hand = _hand_estimate(LLAMA_70B, tp=8, pp=2, dp=2, zero=zero)
    assert est.params_gb == pytest.approx(hand["params"])
    assert est.grads_gb == pytest.approx(hand["grads"])
    assert est.optimizer_gb == pytest.approx(hand["optimizer"])


def test_unsharded_adam_costs_16_bytes_per_param():
    est = estimate_memory(LLAMA_70B, cfg(tp=8, pp=4, dp=1, zero=0))
    model_states = est.params_gb + est.grads_gb + est.optimizer_gb
    assert model_states * GB == pytest.approx(16 * LLAMA_70B.num_params / 32)


def test_each_zero_stage_strictly_reduces_memory_when_dp_above_one():
    totals = [estimate_memory(LLAMA_70B, cfg(8, 2, 2, zero=z)).total_gb for z in range(4)]
    assert totals == sorted(totals, reverse=True)
    assert len(set(totals)) == 4


def test_activation_memory_matches_hand_computation():
    # 34 bytes * s * b * h per layer / tp; 40 layers per stage; 2 micro-batches in flight (1F1B).
    est = estimate_memory(LLAMA_70B, cfg(tp=8, pp=2, dp=2))
    expected = 34 * 4096 * 1 * 8192 * 40 * 2 / 8 / GB
    assert est.activations_gb == pytest.approx(expected)


# --- planner ------------------------------------------------------------------------------------


def test_plan_for_70b_on_4x8_returns_a_valid_layout_that_fits():
    chosen = plan(CLUSTER, LLAMA_70B, global_batch=512)
    validate(CLUSTER, LLAMA_70B, chosen)
    assert estimate_memory(LLAMA_70B, chosen).total_gb <= CLUSTER.usable_mem_gb
    assert chosen.world_size == CLUSTER.world_size


@pytest.mark.parametrize("n_kv_heads", [8, None], ids=["gqa", "mha"])
def test_plan_keeps_tensor_parallelism_inside_a_node(n_kv_heads):
    model = ModelSpec(
        n_layers=80,
        hidden=8192,
        n_heads=64,
        ffn_hidden=28672,
        vocab=128256,
        seq_len=4096,
        n_kv_heads=n_kv_heads,
    )
    chosen = plan(CLUSTER, model, global_batch=512)
    assert chosen.tp <= CLUSTER.gpus_per_node
    for group in build_groups(chosen)["tp"]:
        assert len({rank // CLUSTER.gpus_per_node for rank in group}) == 1


def test_plan_for_70b_on_4x8_picks_tp8_pp2_dp2():
    # tp=8/pp=1 needs ~73 GB > 72 GB usable; next smallest model-parallel degree within a node.
    chosen = plan(CLUSTER, LLAMA_70B, global_batch=512)
    assert (chosen.tp, chosen.pp, chosen.dp) == (8, 2, 2)


def test_plan_raises_when_nothing_fits():
    small = ClusterSpec(num_nodes=1, gpus_per_node=8, gpu_mem_gb=24.0)
    with pytest.raises(ConfigError):
        plan(small, LLAMA_70B, global_batch=64)
