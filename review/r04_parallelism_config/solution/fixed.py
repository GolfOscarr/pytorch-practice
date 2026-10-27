"""3D-parallelism planner (reference fix of pr.py; see REVIEW.md for the bug IDs)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

GB = 1e9

PARAM_BYTES = 2  # bf16 weights
GRAD_BYTES = 2  # bf16 gradients
# FIX(B8): mixed-precision Adam keeps an fp32 master copy (4) plus fp32 m (4) and v (4).
OPTIMIZER_BYTES = 12  # fp32 master weights + Adam first and second moments
ACTIVATION_BYTES_PER_TOKEN_PER_HIDDEN = 34  # per layer; TP + seq-parallel + selective recompute


class ConfigError(ValueError):
    """Raised when a parallel layout cannot run on the given cluster/model."""


@dataclass(frozen=True)
class ClusterSpec:
    num_nodes: int
    gpus_per_node: int
    gpu_mem_gb: float
    usable_mem_fraction: float = 0.9

    @property
    def world_size(self) -> int:
        return self.num_nodes * self.gpus_per_node

    @property
    def usable_mem_gb(self) -> float:
        return self.gpu_mem_gb * self.usable_mem_fraction


@dataclass(frozen=True)
class ModelSpec:
    n_layers: int
    hidden: int
    n_heads: int
    ffn_hidden: int
    vocab: int
    seq_len: int
    n_kv_heads: int | None = None

    def __post_init__(self) -> None:
        if self.hidden % self.n_heads:
            raise ConfigError(f"hidden={self.hidden} is not divisible by n_heads={self.n_heads}")
        if self.n_heads % self.kv_heads:
            raise ConfigError(f"n_heads={self.n_heads} is not divisible by n_kv_heads")

    @property
    def kv_heads(self) -> int:
        return self.n_heads if self.n_kv_heads is None else self.n_kv_heads

    @property
    def head_dim(self) -> int:
        return self.hidden // self.n_heads

    @property
    def num_params(self) -> int:
        """Llama-style decoder: GQA attention, SwiGLU MLP, untied input/output embeddings."""
        h = self.hidden
        attn = 2 * h * h + 2 * h * self.kv_heads * self.head_dim  # q, o + k, v
        mlp = 3 * h * self.ffn_hidden  # gate, up, down
        norms = 2 * h
        embeddings = 2 * self.vocab * h
        return self.n_layers * (attn + mlp + norms) + embeddings + h


@dataclass(frozen=True)
class ParallelConfig:
    tp: int
    pp: int
    dp: int
    micro_batch: int
    global_batch: int
    zero_stage: int = 1

    def __post_init__(self) -> None:
        for name in ("tp", "pp", "dp", "micro_batch", "global_batch"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.zero_stage not in (0, 1, 2, 3):
            raise ConfigError(f"zero_stage must be 0-3, got {self.zero_stage}")

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.dp

    @property
    def grad_accum(self) -> int:
        """Micro-batches each pipeline processes per optimizer step."""
        # FIX(B6): each of the dp replicas runs its own micro-batches.
        return self.global_batch // (self.micro_batch * self.dp)


@dataclass(frozen=True)
class RankCoords:
    dp: int
    pp: int
    tp: int


@dataclass(frozen=True)
class MemoryEstimate:
    params_gb: float
    grads_gb: float
    optimizer_gb: float
    activations_gb: float

    @property
    def total_gb(self) -> float:
        return self.params_gb + self.grads_gb + self.optimizer_gb + self.activations_gb


def validate(cluster: ClusterSpec, model: ModelSpec, cfg: ParallelConfig) -> list[str]:
    """Check `cfg` against the cluster and model.

    Raises `ConfigError` listing every violation; returns non-fatal warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # FIX(B1): the layout must use exactly the GPUs launched; fewer leaves ranks without a role.
    if cfg.world_size != cluster.world_size:
        errors.append(
            f"tp*pp*dp = {cfg.world_size} does not match the cluster world size "
            f"({cluster.world_size})"
        )
    # FIX(B2): TP all-reduces every layer; it must stay inside one node's NVLink domain.
    if cfg.tp > cluster.gpus_per_node or cluster.gpus_per_node % cfg.tp:
        errors.append(
            f"tp={cfg.tp} must divide gpus_per_node={cluster.gpus_per_node} "
            "(tensor parallelism across nodes)"
        )
    if model.hidden % cfg.tp:
        errors.append(f"hidden={model.hidden} is not divisible by tp={cfg.tp}")
    if model.n_heads % cfg.tp:
        errors.append(f"n_heads={model.n_heads} is not divisible by tp={cfg.tp}")
    # FIX(B4): with GQA each TP rank must also own whole key/value heads.
    if model.kv_heads % cfg.tp:
        errors.append(f"n_kv_heads={model.kv_heads} is not divisible by tp={cfg.tp}")
    # Column-parallel MLP: each TP rank owns ffn_hidden / tp columns.
    if model.ffn_hidden % cfg.tp:
        errors.append(f"ffn_hidden={model.ffn_hidden} is not divisible by tp={cfg.tp}")
    # FIX(B5): every pipeline stage must get the same whole number of layers.
    if model.n_layers % cfg.pp:
        errors.append(f"n_layers={model.n_layers} is not divisible by pp={cfg.pp}")
    # FIX(B6): one optimizer step consumes micro_batch * dp samples per accumulation step.
    samples_per_micro_step = cfg.micro_batch * cfg.dp
    if cfg.global_batch % samples_per_micro_step:
        errors.append(
            f"global_batch={cfg.global_batch} is not divisible by "
            f"micro_batch*dp={samples_per_micro_step}"
        )

    if errors:
        raise ConfigError("invalid parallel config:\n  " + "\n  ".join(errors))

    if cfg.pp > 1 and cfg.grad_accum < cfg.pp:
        warnings.append(
            f"only {cfg.grad_accum} micro-batches for {cfg.pp} pipeline stages; "
            "most of each step will be pipeline bubble"
        )
    return warnings


def estimate_memory(model: ModelSpec, cfg: ParallelConfig) -> MemoryEstimate:
    """Rough per-GPU memory for bf16 mixed-precision Adam with ZeRO sharding over dp."""
    params_per_gpu = model.num_params / (cfg.tp * cfg.pp)

    param_shard = cfg.dp if cfg.zero_stage >= 3 else 1
    grad_shard = cfg.dp if cfg.zero_stage >= 2 else 1
    optimizer_shard = cfg.dp if cfg.zero_stage >= 1 else 1  # FIX(B7): ZeRO-0 replicates it

    layers_per_stage = model.n_layers / cfg.pp
    # 1F1B: the first stage holds activations for up to `pp` in-flight micro-batches.
    in_flight = min(cfg.pp, cfg.grad_accum)
    activation_bytes = (
        ACTIVATION_BYTES_PER_TOKEN_PER_HIDDEN
        * model.seq_len
        * cfg.micro_batch
        * model.hidden
        * layers_per_stage
        * in_flight
        / cfg.tp
    )

    return MemoryEstimate(
        params_gb=params_per_gpu * PARAM_BYTES / param_shard / GB,
        grads_gb=params_per_gpu * GRAD_BYTES / grad_shard / GB,
        optimizer_gb=params_per_gpu * OPTIMIZER_BYTES / optimizer_shard / GB,
        activations_gb=activation_bytes / GB,
    )


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def plan(
    cluster: ClusterSpec,
    model: ModelSpec,
    global_batch: int,
    micro_batch: int = 1,
    zero_stage: int = 1,
) -> ParallelConfig:
    """Pick the valid layout with the smallest tp*pp that fits in GPU memory.

    Ties are broken towards fewer pipeline stages. Raises `ConfigError` if nothing fits.
    """
    world = cluster.world_size
    candidates: list[ParallelConfig] = []
    # FIX(B2): only consider TP degrees that fit inside one node.
    for tp in _divisors(cluster.gpus_per_node):
        for pp in _divisors(world // tp):
            cfg = ParallelConfig(
                tp=tp,
                pp=pp,
                dp=world // (tp * pp),
                micro_batch=micro_batch,
                global_batch=global_batch,
                zero_stage=zero_stage,
            )
            try:
                validate(cluster, model, cfg)
            except ConfigError as exc:
                logger.debug("skip tp=%d pp=%d: %s", tp, pp, exc)
                continue
            mem = estimate_memory(model, cfg)
            if mem.total_gb <= cluster.usable_mem_gb:
                candidates.append(cfg)
            else:
                logger.debug("skip tp=%d pp=%d: needs %.1f GB", tp, pp, mem.total_gb)

    if not candidates:
        raise ConfigError(f"no (tp, pp, dp) layout fits {model} on {cluster}")
    return min(candidates, key=lambda c: (c.tp * c.pp, c.pp))


def rank_to_coords(rank: int, cfg: ParallelConfig) -> RankCoords:
    """Map a global rank to its (dp, pp, tp) coordinates.

    TP ranks are contiguous so each tensor-parallel group sits on one node's NVLink domain.
    """
    if not 0 <= rank < cfg.world_size:
        raise ValueError(f"rank {rank} out of range for world size {cfg.world_size}")
    # FIX(B3): tp must be the fastest-varying axis (then dp, then pp), or TP groups are strided
    # across nodes and every per-layer all-reduce goes over the inter-node fabric.
    tp = rank % cfg.tp
    dp = (rank // cfg.tp) % cfg.dp
    pp = rank // (cfg.tp * cfg.dp)
    return RankCoords(dp=dp, pp=pp, tp=tp)


def build_groups(cfg: ParallelConfig) -> dict[str, list[list[int]]]:
    """Process groups for each axis: ranks that differ only along that axis."""
    groups: dict[str, dict[tuple[int, int], list[int]]] = {"tp": {}, "pp": {}, "dp": {}}
    for rank in range(cfg.world_size):
        c = rank_to_coords(rank, cfg)
        groups["tp"].setdefault((c.dp, c.pp), []).append(rank)
        groups["pp"].setdefault((c.dp, c.tp), []).append(rank)
        groups["dp"].setdefault((c.pp, c.tp), []).append(rank)
    return {axis: sorted(by_key.values()) for axis, by_key in groups.items()}
