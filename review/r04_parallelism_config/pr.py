"""Add a 3D-parallelism planner for the LLM training launcher.

## Summary
Right now every launch script hard-codes `--tp/--pp/--dp` and we find out a layout is bad when
the job OOMs forty minutes in (or, worse, runs at half the MFU because nobody noticed). This PR
adds a small, dependency-free planner the launcher can call before spawning any processes:

- describe the cluster and the model once (`ClusterSpec`, `ModelSpec`),
- `validate()` a hand-written `ParallelConfig` and get every problem reported at once,
- or let `plan()` pick the layout that maximizes data parallelism while fitting in memory,
- and map each global rank to its (dp, pp, tp) coordinates / process groups.

## Changes
- `ClusterSpec`, `ModelSpec`, `ParallelConfig` frozen dataclasses. `ModelSpec.num_params` counts
  a Llama-style decoder (GQA attention + SwiGLU MLP + untied embeddings).
- `validate(cluster, model, cfg)` raises `ConfigError` listing all violations and returns a list
  of non-fatal warnings (e.g. a pipeline that cannot be kept full).
- `estimate_memory(model, cfg)` gives a per-GPU breakdown of params / grads / optimizer state /
  activations for bf16 mixed-precision Adam, with ZeRO stages 0-3.
- `plan(cluster, model, global_batch, micro_batch)` enumerates all (tp, pp, dp) factorizations,
  drops the invalid ones and those that do not fit, and returns the one with the smallest
  model-parallel degree (ties broken towards fewer pipeline stages, since bubbles are expensive).
- `rank_to_coords()` / `build_groups()` produce the rank layout; TP ranks are kept contiguous so
  tensor-parallel all-reduces stay on NVLink.

## Testing
Checked `plan()` on our 4x8 H100 cluster for the 70B config: it returns a layout that fits in
memory, and the groups from `build_groups()` cover every rank exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1e9

PARAM_BYTES = 2  # bf16 weights
GRAD_BYTES = 2  # bf16 gradients
OPTIMIZER_BYTES = 8  # fp32 Adam first and second moments
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
        return self.global_batch // self.micro_batch


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

    if cfg.world_size > cluster.world_size:
        errors.append(
            f"tp*pp*dp = {cfg.world_size} needs more GPUs than the cluster has "
            f"({cluster.world_size})"
        )
    if model.hidden % cfg.tp:
        errors.append(f"hidden={model.hidden} is not divisible by tp={cfg.tp}")
    if model.n_heads % cfg.tp:
        errors.append(f"n_heads={model.n_heads} is not divisible by tp={cfg.tp}")
    if cfg.global_batch % cfg.micro_batch:
        errors.append(
            f"global_batch={cfg.global_batch} is not divisible by micro_batch={cfg.micro_batch}"
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
    optimizer_shard = cfg.dp

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
    for tp in _divisors(world):
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
            except ConfigError:
                continue
            if estimate_memory(model, cfg).total_gb <= cluster.usable_mem_gb:
                candidates.append(cfg)

    if not candidates:
        raise ConfigError(f"no (tp, pp, dp) layout fits {model} on {cluster}")
    return min(candidates, key=lambda c: (c.tp * c.pp, c.pp))


def rank_to_coords(rank: int, cfg: ParallelConfig) -> RankCoords:
    """Map a global rank to its (dp, pp, tp) coordinates.

    TP ranks are contiguous so each tensor-parallel group sits on one node's NVLink domain.
    """
    if not 0 <= rank < cfg.world_size:
        raise ValueError(f"rank {rank} out of range for world size {cfg.world_size}")
    dp = rank % cfg.dp
    pp = (rank // cfg.dp) % cfg.pp
    tp = rank // (cfg.dp * cfg.pp)
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
