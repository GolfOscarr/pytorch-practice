import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from common import get_device
from practice.p05_transformer.transformer import (
    GPT,
    CausalSelfAttention,
    FeedForward,
    GPTConfig,
    LayerNorm,
    TransformerBlock,
)

DEVICE = get_device()


def small_config(**overrides: float) -> GPTConfig:
    fields = dict(vocab_size=50, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    fields.update(overrides)
    return GPTConfig(**fields)  # type: ignore[arg-type]


def test_layernorm_matches_nn_layernorm_forward_and_backward(monkeypatch: pytest.MonkeyPatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("implement LayerNorm yourself: F.layer_norm is not allowed")

    torch.manual_seed(0)
    ref = nn.LayerNorm(24, dtype=torch.float64)
    with torch.no_grad():
        ref.weight.normal_()
        ref.bias.normal_()
    ln = LayerNorm(24).double()
    ln.load_state_dict(ref.state_dict())  # same parameter names: weight, bias

    x = torch.randn(3, 5, 24, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    grad_out = torch.randn(3, 5, 24, dtype=torch.float64)

    expected = ref(x_ref)
    monkeypatch.setattr(F, "layer_norm", forbidden)
    out = ln(x)
    torch.testing.assert_close(out, expected)

    out.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(ln.weight.grad, ref.weight.grad)
    torch.testing.assert_close(ln.bias.grad, ref.bias.grad)


def test_layernorm_parameters_and_eps():
    ln = LayerNorm(8, eps=1e-3)
    assert {name for name, _ in ln.named_parameters()} == {"weight", "bias"}
    torch.testing.assert_close(ln.weight.data, torch.ones(8))
    torch.testing.assert_close(ln.bias.data, torch.zeros(8))
    # A constant input has zero variance: the output must be finite (eps) and equal to bias.
    out = ln(torch.full((2, 8), 3.0))
    torch.testing.assert_close(out, torch.zeros(2, 8))
    # eps must actually be used: compare against nn.LayerNorm with the same eps.
    x = torch.randn(4, 8) * 0.01
    torch.testing.assert_close(ln(x), nn.LayerNorm(8, eps=1e-3)(x))


def test_feedforward_shape_params_and_dropout_modes():
    torch.manual_seed(0)
    C = 16
    ffn = FeedForward(C, dropout=0.5).to(DEVICE)
    # Linear(C, 4C) + Linear(4C, C), both with bias.
    assert sum(p.numel() for p in ffn.parameters()) == 8 * C * C + 5 * C
    x = torch.randn(2, 7, C, device=DEVICE)

    ffn.eval()
    out = ffn(x)
    assert out.shape == (2, 7, C)
    torch.testing.assert_close(ffn(x), out)  # eval: deterministic

    ffn.train()
    assert not torch.allclose(ffn(x), ffn(x))  # train: dropout is active


def test_attention_shape_params_and_bad_heads():
    C = 32
    attn = CausalSelfAttention(C, n_head=4).to(DEVICE)
    # fused qkv Linear(C, 3C) + output Linear(C, C), both with bias.
    assert sum(p.numel() for p in attn.parameters()) == 4 * C * C + 4 * C
    assert attn(torch.randn(3, 10, C, device=DEVICE)).shape == (3, 10, C)
    with pytest.raises(ValueError):
        CausalSelfAttention(30, n_head=4)


def test_attention_is_causal():
    torch.manual_seed(0)
    attn = CausalSelfAttention(16, n_head=2).to(DEVICE).eval()
    x = torch.randn(2, 8, 16, device=DEVICE)
    x2 = x.clone()
    x2[:, 5] += 1.0  # perturb position 5
    y, y2 = attn(x), attn(x2)
    torch.testing.assert_close(y[:, :5], y2[:, :5])  # the past cannot see the future
    assert not torch.allclose(y[:, 5:], y2[:, 5:])  # ...but positions >= 5 must change


def test_block_shape_and_pre_norm_residual():
    torch.manual_seed(0)
    block = TransformerBlock(32, n_head=4).to(DEVICE).eval()
    x = torch.randn(2, 9, 32, device=DEVICE)
    assert block(x).shape == (2, 9, 32)
    # With every parameter zeroed, each sublayer outputs 0. A pre-norm residual block is then the
    # identity; a post-norm block (LayerNorm on the residual stream) would output 0 instead.
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
    torch.testing.assert_close(block(x), x)


def test_config_validation():
    small_config()  # valid
    bad = [
        dict(n_embd=30, n_head=4),  # not divisible
        dict(vocab_size=0),
        dict(block_size=-1),
        dict(n_layer=0),
        dict(n_head=0),
        dict(dropout=1.0),
        dict(dropout=-0.1),
    ]
    for overrides in bad:
        with pytest.raises(ValueError):
            small_config(**overrides)


def test_gpt_structure_weight_tying_and_param_count():
    cfg = small_config()
    model = GPT(cfg)
    assert model.lm_head.weight is model.tok_emb.weight  # the SAME Parameter, not a copy
    assert model.lm_head.bias is None
    assert isinstance(model.blocks, nn.ModuleList) and len(model.blocks) == cfg.n_layer
    assert not any(isinstance(m, nn.LayerNorm) for m in model.modules())
    # pos_ids is a buffer (moves with .to), not a parameter.
    assert "pos_ids" in dict(model.named_buffers())
    assert "pos_ids" not in dict(model.named_parameters())

    V, T, L, C = cfg.vocab_size, cfg.block_size, cfg.n_layer, cfg.n_embd
    per_block = 12 * C * C + 13 * C  # 2 LayerNorms + attention + feed-forward
    expected = V * C + T * C + L * per_block + 2 * C  # tied lm_head adds nothing
    assert model.num_parameters() == expected
    assert model.num_parameters() == sum(p.numel() for p in model.parameters())


def test_forward_shapes_and_initial_loss(monkeypatch: pytest.MonkeyPatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("use your own LayerNorm, not F.layer_norm")

    monkeypatch.setattr(F, "layer_norm", forbidden)
    torch.manual_seed(0)
    cfg = small_config()
    model = GPT(cfg).to(DEVICE)
    idx = torch.randint(cfg.vocab_size, (4, 12), device=DEVICE)

    logits, loss = model(idx)
    assert logits.shape == (4, 12, cfg.vocab_size)
    assert loss is None

    logits, loss = model(idx, targets=torch.randint(cfg.vocab_size, (4, 12), device=DEVICE))
    assert loss is not None and loss.ndim == 0
    # Near-uniform predictions at init: loss ~ ln(vocab_size) ~ 3.91.
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.3
    loss.backward()
    assert model.tok_emb.weight.grad is not None
    assert model.pos_emb.weight.grad is not None


def test_gpt_is_causal():
    torch.manual_seed(0)
    cfg = small_config()
    model = GPT(cfg).to(DEVICE).eval()
    idx = torch.randint(cfg.vocab_size, (2, 10), device=DEVICE)
    idx2 = idx.clone()
    idx2[:, 6] = (idx2[:, 6] + 1) % cfg.vocab_size
    logits, _ = model(idx)
    logits2, _ = model(idx2)
    torch.testing.assert_close(logits[:, :6], logits2[:, :6])
    assert not torch.allclose(logits[:, 6:], logits2[:, 6:])


def test_too_long_input_raises_but_block_size_is_ok():
    cfg = small_config()
    model = GPT(cfg).to(DEVICE)
    logits, _ = model(torch.zeros(1, cfg.block_size, dtype=torch.long, device=DEVICE))
    assert logits.shape == (1, cfg.block_size, cfg.vocab_size)
    with pytest.raises(ValueError):
        model(torch.zeros(1, cfg.block_size + 1, dtype=torch.long, device=DEVICE))


def test_generate_shape_crops_context_and_greedy_is_deterministic():
    torch.manual_seed(0)
    cfg = small_config()
    model = GPT(cfg).to(DEVICE).eval()
    # Prompt longer than block_size: generate must crop the context instead of raising.
    prompt = torch.randint(cfg.vocab_size, (2, cfg.block_size + 4), device=DEVICE)

    out = model.generate(prompt, max_new_tokens=6)
    assert out.shape == (2, cfg.block_size + 10)
    assert out.dtype == torch.long
    torch.testing.assert_close(out[:, : prompt.shape[1]], prompt)  # prompt is kept

    torch.manual_seed(1)
    greedy_a = model.generate(prompt, max_new_tokens=6, top_k=1)
    torch.manual_seed(2)
    greedy_b = model.generate(prompt, max_new_tokens=6, top_k=1, temperature=0.7)
    torch.testing.assert_close(greedy_a, greedy_b)

    # top_k=1 must equal an explicit argmax rollout.
    expected = prompt
    for _ in range(6):
        logits, _ = model(expected[:, -cfg.block_size :])
        expected = torch.cat([expected, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    torch.testing.assert_close(greedy_a, expected)

    with pytest.raises(ValueError):
        model.generate(prompt, max_new_tokens=1, temperature=0.0)


def test_state_dict_round_trip():
    torch.manual_seed(0)
    cfg = small_config()
    model = GPT(cfg).eval()
    clone = GPT(cfg).eval()
    clone.load_state_dict(model.state_dict())  # strict: keys must match exactly
    idx = torch.randint(cfg.vocab_size, (2, 8))
    torch.testing.assert_close(clone(idx)[0], model(idx)[0])
    assert clone.lm_head.weight is clone.tok_emb.weight  # tying survives loading


def test_tiny_gpt_overfits_repeated_sequence():
    torch.manual_seed(0)
    cfg = small_config(vocab_size=16, block_size=24, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg).to(DEVICE)
    pattern = torch.randperm(cfg.vocab_size)[:6]
    seq = pattern.repeat(5)[: cfg.block_size + 1].to(DEVICE)
    x, y = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = model(x, y)
        assert loss is not None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05

    model.eval()
    out = model.generate(seq[:6].unsqueeze(0), max_new_tokens=12, top_k=1)
    torch.testing.assert_close(out[0], seq[:18])  # greedy decoding continues the pattern
