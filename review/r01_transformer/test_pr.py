from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from review.r01_transformer.pr import GPT, Block, CausalSelfAttention, GPTConfig

# Reference comparisons run on CPU in float64 so only real math differences can fail them.
DTYPE = torch.float64
ATOL = 1e-10

CFG = GPTConfig(vocab_size=37, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


def _model(cfg: GPTConfig = CFG, seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(cfg).to(DTYPE).eval()


def _tokens(batch: int, length: int, vocab: int = CFG.vocab_size, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, length), generator=g)


def _hidden(batch: int, length: int, seed: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, length, CFG.n_embd, generator=g, dtype=DTYPE)


# ---------------------------------------------------------------- references from torch primitives


def _ref_attention(attn: CausalSelfAttention, x: torch.Tensor, n_head: int) -> torch.Tensor:
    B, T, C = x.shape
    q, k, v = F.linear(x, attn.qkv.weight, attn.qkv.bias).split(C, dim=2)
    q, k, v = (t.view(B, T, n_head, C // n_head).transpose(1, 2) for t in (q, k, v))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = y.transpose(1, 2).reshape(B, T, C)
    return F.linear(y, attn.proj.weight, attn.proj.bias)


def _ref_layer_norm(ln: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, ln.normalized_shape, ln.weight, ln.bias, ln.eps)


def _ref_block(block: Block, x: torch.Tensor, n_head: int) -> torch.Tensor:
    fc, proj = cast(nn.Linear, block.mlp[0]), cast(nn.Linear, block.mlp[2])
    x = x + _ref_attention(block.attn, _ref_layer_norm(block.ln_1, x), n_head)
    h = F.gelu(F.linear(_ref_layer_norm(block.ln_2, x), fc.weight, fc.bias))
    return x + F.linear(h, proj.weight, proj.bias)


def _ref_logits(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    T = idx.shape[1]
    x = model.tok_emb.weight[idx] + model.pos_emb.weight[:T]
    for block in model.blocks:
        x = _ref_block(cast(Block, block), x, model.config.n_head)
    x = _ref_layer_norm(model.ln_f, x)
    return x @ model.tok_emb.weight.T  # tied head: logits use the embedding matrix


# --------------------------------------------------------------------------------- forward / shapes


def test_forward_returns_logits_of_expected_shape_and_no_loss_without_targets():
    model = _model()
    logits, loss = model(_tokens(3, 10))
    assert logits.shape == (3, 10, CFG.vocab_size)
    assert loss is None


def test_forward_rejects_sequences_longer_than_block_size():
    model = _model()
    with pytest.raises(ValueError):
        model(_tokens(1, CFG.block_size + 1))


def test_attention_matches_scaled_dot_product_attention_reference():
    torch.manual_seed(0)
    attn = CausalSelfAttention(CFG).to(DTYPE).eval()
    x = _hidden(2, 11)
    torch.testing.assert_close(attn(x), _ref_attention(attn, x, CFG.n_head), rtol=0, atol=ATOL)


def test_block_matches_pre_norm_reference():
    torch.manual_seed(0)
    block = Block(CFG).to(DTYPE).eval()
    x = _hidden(2, 11)
    torch.testing.assert_close(block(x), _ref_block(block, x, CFG.n_head), rtol=0, atol=ATOL)


def test_model_logits_match_reference_built_from_torch_primitives():
    model = _model()
    idx = _tokens(2, 12)
    logits, _ = model(idx)
    torch.testing.assert_close(logits, _ref_logits(model, idx), rtol=0, atol=ATOL)


def test_future_tokens_do_not_affect_past_logits():
    model = _model()
    idx = _tokens(2, 12)
    changed = idx.clone()
    changed[:, 7:] = (changed[:, 7:] + 1) % CFG.vocab_size
    before, _ = model(idx)
    after, _ = model(changed)
    torch.testing.assert_close(before[:, :7], after[:, :7], rtol=0, atol=ATOL)
    assert not torch.allclose(before[:, 7], after[:, 7])


def test_each_position_depends_on_its_own_token():
    model = _model()
    idx = _tokens(1, 12)
    changed = idx.clone()
    changed[:, 0] = (changed[:, 0] + 1) % CFG.vocab_size
    before, _ = model(idx)
    after, _ = model(changed)
    assert not torch.allclose(before[:, 0], after[:, 0])


# ------------------------------------------------------------------------------------------- loss


def test_loss_is_mean_token_cross_entropy():
    model = _model()
    idx, targets = _tokens(3, 10, seed=1), _tokens(3, 10, seed=2)
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 10, CFG.vocab_size)
    assert loss is not None and loss.ndim == 0
    expected = F.cross_entropy(logits.reshape(-1, CFG.vocab_size), targets.reshape(-1))
    torch.testing.assert_close(loss, expected)


def test_loss_backpropagates_to_all_parameters():
    model = _model().train()
    idx, targets = _tokens(2, 8, seed=1), _tokens(2, 8, seed=2)
    _, loss = model(idx, targets)
    assert loss is not None
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []


# ------------------------------------------------------------------------------------ weight tying


def test_lm_head_shares_the_token_embedding_parameter():
    model = _model()
    assert model.lm_head.weight is model.tok_emb.weight


def test_tied_weights_stay_identical_after_an_optimizer_step():
    model = _model().train()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    _, loss = model(_tokens(2, 8, seed=1), _tokens(2, 8, seed=2))
    assert loss is not None
    loss.backward()
    opt.step()
    torch.testing.assert_close(model.lm_head.weight, model.tok_emb.weight, rtol=0, atol=0)


def test_parameter_count_counts_the_tied_matrix_once():
    V, C, L, S = CFG.vocab_size, CFG.n_embd, CFG.n_layer, CFG.block_size
    per_block = (2 * C) + (3 * C * C + 3 * C) + (C * C + C) + (2 * C) + (8 * C * C + 5 * C)
    expected = V * C + S * C + L * per_block + 2 * C
    assert sum(p.numel() for p in _model().parameters()) == expected


# -------------------------------------------------------------------------------------- generation


def _eval_greedy(model: GPT, idx: torch.Tensor, steps: int) -> torch.Tensor:
    """Greedy decoding reference: eval mode, no grad, cropped to the context window."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            logits = _ref_logits(model, idx[:, -model.config.block_size :])
            idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    model.train(was_training)
    return idx


def test_generate_appends_new_tokens_after_the_prompt():
    model = _model()
    prompt = _tokens(2, 4)
    out = model.generate(prompt, max_new_tokens=5, generator=torch.Generator().manual_seed(0))
    assert out.shape == (2, 9)
    assert torch.equal(out[:, :4], prompt)
    assert int(out.min()) >= 0 and int(out.max()) < CFG.vocab_size


def test_generate_can_continue_past_the_context_window():
    model = _model()
    prompt = _tokens(1, 4)
    steps = 2 * CFG.block_size
    out = model.generate(prompt, max_new_tokens=steps, greedy=True)
    assert out.shape == (1, 4 + steps)
    assert torch.equal(out, _eval_greedy(model, prompt, steps))


def test_greedy_generation_ignores_dropout_and_keeps_training_mode():
    cfg = GPTConfig(vocab_size=37, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.5)
    model = _model(cfg).train()
    prompt = _tokens(2, 3)
    expected = _eval_greedy(model, prompt, steps=10)
    torch.manual_seed(123)
    out = model.generate(prompt, max_new_tokens=10, greedy=True)
    assert torch.equal(out, expected)
    assert model.training


def test_generate_does_not_record_autograd_history():
    model = _model()
    seen: list[bool] = []
    model.lm_head.register_forward_hook(lambda _m, _i, out: seen.append(out.requires_grad))
    model.generate(_tokens(1, 3), max_new_tokens=3, greedy=True)
    assert seen == [False, False, False]


def test_sampling_is_reproducible_with_a_seeded_generator():
    model = _model()
    prompt = _tokens(2, 3)
    a = model.generate(prompt, 8, top_k=5, generator=torch.Generator().manual_seed(7))
    b = model.generate(prompt, 8, top_k=5, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
