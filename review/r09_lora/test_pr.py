import copy

import pytest
import torch
from torch import nn

from review.r09_lora.pr import (
    LoRALinear,
    build_optimizer,
    count_trainable_params,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    mark_only_lora_trainable,
)

# CPU + float64 so equivalence checks can use tight tolerances.
DTYPE = torch.float64
D, VOCAB, R, ALPHA = 16, 50, 4, 8.0


class SelfAttention(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        att = torch.softmax(q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5, dim=-1)
        return self.o_proj(att @ v)


class CrossAttention(nn.Module):
    """Cross-attention with a fused key/value projection, as in many encoder-decoder models."""

    def __init__(self, d: int):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.kv_proj = nn.Linear(d, 2 * d)
        self.o_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k, v = self.kv_proj(memory).chunk(2, dim=-1)
        att = torch.softmax(q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5, dim=-1)
        return self.o_proj(att @ v)


class Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.self_attn = SelfAttention(d)
        self.norm2 = nn.LayerNorm(d)
        self.cross_attn = CrossAttention(d)
        self.fc1 = nn.Linear(d, 2 * d)
        self.fc2 = nn.Linear(2 * d, d)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), memory)
        return x + self.fc2(torch.relu(self.fc1(x)))


class TinyModel(nn.Module):
    def __init__(self, d: int = D, vocab: int = VOCAB, n_blocks: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n_blocks))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        memory = x
        for block in self.blocks:
            x = block(x, memory)
        return self.head(self.norm(x))


EXPECTED_TARGETS = [
    "blocks.0.self_attn.q_proj",
    "blocks.0.self_attn.v_proj",
    "blocks.0.cross_attn.q_proj",
    "blocks.1.self_attn.q_proj",
    "blocks.1.self_attn.v_proj",
    "blocks.1.cross_attn.q_proj",
]


def _make_model() -> TinyModel:
    torch.manual_seed(0)
    return TinyModel().to(DTYPE)


def _ids() -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, VOCAB, (3, 7), generator=g)


def _lora_model(dropout: float = 0.0) -> TinyModel:
    model = _make_model()
    inject_lora(model, r=R, alpha=ALPHA, dropout=dropout)
    mark_only_lora_trainable(model)
    return model


def _randomize_adapters(model: nn.Module, seed: int = 2) -> None:
    """Give B non-zero values so the adapter actually changes the output."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.copy_(torch.randn(m.lora_A.shape, generator=g, dtype=DTYPE))
                m.lora_B.copy_(torch.randn(m.lora_B.shape, generator=g, dtype=DTYPE))


def _lora_layer(in_f: int = 12, out_f: int = 20, dropout: float = 0.0) -> LoRALinear:
    torch.manual_seed(3)
    layer = LoRALinear(nn.Linear(in_f, out_f).to(DTYPE), r=R, alpha=ALPHA, dropout=dropout)
    _randomize_adapters(layer)
    return layer


def _x(in_f: int = 12) -> torch.Tensor:
    return torch.randn(5, in_f, generator=torch.Generator().manual_seed(4), dtype=DTYPE)


# --- LoRALinear -------------------------------------------------------------------------------


def test_lora_linear_output_shape_and_param_shapes():
    layer = _lora_layer()
    assert layer.lora_A.shape == (R, 12)
    assert layer.lora_B.shape == (20, R)
    assert layer(_x()).shape == (5, 20)


def test_fresh_adapter_does_not_change_layer_output():
    torch.manual_seed(5)
    base = nn.Linear(12, 20).to(DTYPE)
    layer = LoRALinear(copy.deepcopy(base), r=R, alpha=ALPHA)
    torch.testing.assert_close(layer(_x()), base(_x()), rtol=0, atol=0)


def test_forward_matches_low_rank_formula_with_alpha_over_r_scaling():
    layer = _lora_layer()
    x = _x()
    expected = layer.base(x) + (ALPHA / R) * (x @ (layer.lora_B @ layer.lora_A).T)
    torch.testing.assert_close(layer(x), expected)


def test_merged_output_matches_unmerged_output():
    layer = _lora_layer()
    x = _x()
    reference = layer(x)
    layer.merge()
    torch.testing.assert_close(layer(x), reference)


def test_merged_output_matches_for_square_layer():
    layer = _lora_layer(in_f=12, out_f=12)
    x = _x()
    reference = layer(x)
    layer.merge()
    torch.testing.assert_close(layer(x), reference)


def test_merging_twice_does_not_change_output():
    layer = _lora_layer(in_f=12, out_f=12)
    x = _x()
    reference = layer(x)
    layer.merge()
    layer.merge()
    torch.testing.assert_close(layer(x), reference)


def test_unmerge_restores_base_weight():
    layer = _lora_layer(in_f=12, out_f=12)
    original = layer.base.weight.detach().clone()
    layer.merge()
    assert not torch.allclose(layer.base.weight, original)
    layer.unmerge()
    torch.testing.assert_close(layer.base.weight, original, rtol=0, atol=1e-12)


def test_adapter_dropout_does_not_touch_base_path_in_training():
    torch.manual_seed(6)
    base = nn.Linear(12, 20).to(DTYPE)
    layer = LoRALinear(copy.deepcopy(base), r=R, alpha=ALPHA, dropout=0.5).train()
    # A fresh adapter contributes nothing, so even in train mode the output is exactly base(x).
    torch.testing.assert_close(layer(_x()), base(_x()), rtol=0, atol=0)


def test_dropout_is_disabled_in_eval_mode():
    layer = _lora_layer(dropout=0.5).eval()
    x = _x()
    torch.testing.assert_close(layer(x), layer(x), rtol=0, atol=0)
    expected = layer.base(x) + (ALPHA / R) * (x @ (layer.lora_B @ layer.lora_A).T)
    torch.testing.assert_close(layer(x), expected)


# --- model-level helpers ----------------------------------------------------------------------


def test_inject_wraps_exactly_the_target_modules():
    model = _make_model()
    replaced = inject_lora(model, r=R, alpha=ALPHA)
    assert sorted(replaced) == sorted(EXPECTED_TARGETS)
    wrapped = sorted(n for n, m in model.named_modules() if isinstance(m, LoRALinear))
    assert wrapped == sorted(EXPECTED_TARGETS)
    for block in model.blocks:
        assert type(block.cross_attn.kv_proj) is nn.Linear
        assert type(block.self_attn.k_proj) is nn.Linear


def test_injected_model_matches_base_model_at_init():
    base = _make_model()
    model = copy.deepcopy(base)
    inject_lora(model, r=R, alpha=ALPHA, dropout=0.1)
    model.eval()
    base.eval()
    torch.testing.assert_close(model(_ids()), base(_ids()), rtol=0, atol=0)


def test_trainable_param_count_is_r_times_in_plus_out_per_target():
    model = _lora_layer()  # standalone layer first
    mark_only_lora_trainable(model)
    assert count_trainable_params(model) == R * (12 + 20)

    model = _lora_model()
    assert count_trainable_params(model) == len(EXPECTED_TARGETS) * R * (D + D)


def test_only_lora_params_require_grad():
    model = _lora_model()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    expected = {f"{t}.lora_{ab}" for t in EXPECTED_TARGETS for ab in "AB"}
    assert trainable == expected


def test_gradients_flow_only_to_adapters():
    model = _lora_model()
    _randomize_adapters(model)
    model(_ids()).sum().backward()
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.grad is not None and param.grad.abs().sum() > 0, name
        else:
            assert param.grad is None, name


def test_model_merge_matches_unmerged_output():
    model = _lora_model()
    _randomize_adapters(model)
    model.eval()
    reference = model(_ids())
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()
    torch.testing.assert_close(model(_ids()), reference)


def test_optimizer_param_groups_hold_only_trainable_params_with_decay_policy():
    model = _lora_model()
    model.norm.weight.requires_grad_(True)  # e.g. also tuning the final norm
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.1)

    def ids(params) -> set[int]:
        return {id(p) for p in params}

    by_decay = {g["weight_decay"]: ids(g["params"]) for g in opt.param_groups}
    lora = ids(p for n, p in model.named_parameters() if "lora_" in n)
    assert by_decay == {0.1: lora, 0.0: {id(model.norm.weight)}}
    assert all(g["lr"] == 1e-3 for g in opt.param_groups)


def test_training_step_changes_only_adapters():
    model = _lora_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = build_optimizer(model, lr=1e-2, weight_decay=0.0)
    model(_ids()).logsumexp(-1).mean().backward()
    opt.step()
    for name, param in model.named_parameters():
        changed = not torch.equal(param, before[name])
        # A only gets gradient once B != 0, so after one step only B is guaranteed to move.
        if name.endswith("lora_B"):
            assert changed, name
        elif "lora_" not in name:
            assert not changed, name


def test_lora_state_dict_contains_only_adapter_tensors():
    model = _lora_model()
    state = lora_state_dict(model)
    assert set(state) == {f"{t}.lora_{ab}" for t in EXPECTED_TARGETS for ab in "AB"}


def test_adapter_snapshot_can_roll_back_training():
    model = _lora_model()
    snapshot = lora_state_dict(model)
    reference = model(_ids()).detach()

    opt = build_optimizer(model, lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        model(_ids()).logsumexp(-1).mean().backward()
        opt.step()
    assert not torch.allclose(model(_ids()), reference)

    load_lora_state_dict(model, snapshot)
    torch.testing.assert_close(model(_ids()), reference)


def test_adapter_checkpoint_transfers_to_fresh_model():
    trained = _lora_model()
    _randomize_adapters(trained, seed=7)
    fresh = _lora_model()
    load_lora_state_dict(fresh, lora_state_dict(trained))
    torch.testing.assert_close(fresh(_ids()), trained(_ids()))


@pytest.mark.parametrize("r", [1, 4, 8])
def test_trainable_count_scales_linearly_with_rank(r: int):
    model = _make_model()
    inject_lora(model, r=r, alpha=2.0 * r)
    mark_only_lora_trainable(model)
    assert count_trainable_params(model) == len(EXPECTED_TARGETS) * r * 2 * D
