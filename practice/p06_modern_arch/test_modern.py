import math

import pytest
import torch
import torch.nn.functional as F

from common import get_device
from practice.p06_modern_arch.modern import (
    GroupedQueryAttention,
    KVCache,
    RMSNorm,
    SwiGLU,
    apply_rope,
    precompute_rope,
)

DEVICE = get_device()
TOL = {"atol": 1e-5, "rtol": 1e-4}  # float32 on MPS/CPU; different op orders differ ~1e-6


# ---------------------------------------------------------------- RMSNorm


def test_rmsnorm_matches_torch():
    torch.manual_seed(0)
    ours = RMSNorm(16, eps=1e-5).to(DEVICE)
    ref = torch.nn.RMSNorm(16, eps=1e-5).to(DEVICE)
    with torch.no_grad():
        ref.weight.copy_(torch.randn(16))
    ours.load_state_dict(ref.state_dict())
    x = torch.randn(3, 5, 16, device=DEVICE) * 4
    torch.testing.assert_close(ours(x), ref(x), **TOL)


def test_rmsnorm_gradcheck():
    torch.manual_seed(0)
    norm = RMSNorm(6, eps=1e-6).double()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(6, dtype=torch.float64))
    x = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(norm, (x,))


# ---------------------------------------------------------------- RoPE


def test_precompute_rope_tables():
    cos, sin = precompute_rope(8, 32, theta=10000.0)
    assert cos.shape == sin.shape == (32, 8)
    # Position 0 is the identity rotation.
    torch.testing.assert_close(cos[0], torch.ones(8))
    torch.testing.assert_close(sin[0], torch.zeros(8))
    # Rotate-half layout: both halves share the same angles; first frequency is 1 rad/position.
    torch.testing.assert_close(cos[:, :4], cos[:, 4:])
    torch.testing.assert_close(sin[5, 0], torch.tensor(math.sin(5.0)))
    expected_last = 5.0 * 10000.0 ** (-6 / 8)
    torch.testing.assert_close(sin[5, 3], torch.tensor(math.sin(expected_last)))


def test_rope_rotate_half_convention():
    # e_0 at position 1 rotates within the pair (dim 0, dim D/2), not (dim 0, dim 1).
    cos, sin = precompute_rope(4, 4)
    x = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 1, 4)
    out = apply_rope(x, cos, sin, offset=1).flatten()
    expected = torch.tensor([math.cos(1.0), 0.0, math.sin(1.0), 0.0])
    torch.testing.assert_close(out, expected)


def test_rope_preserves_norm():
    torch.manual_seed(0)
    cos, sin = precompute_rope(32, 64)
    x = torch.randn(2, 3, 64, 32)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1), **TOL)


def test_rope_relative_position_property():
    # <rope(q, m), rope(k, n)> depends only on m - n.
    torch.manual_seed(0)
    d, t = 16, 20
    cos, sin = precompute_rope(d, t)
    q, k = torch.randn(d), torch.randn(d)
    rq = apply_rope(q.expand(1, 1, t, d), cos, sin)[0, 0]  # rq[m] = rope(q, m)
    rk = apply_rope(k.expand(1, 1, t, d), cos, sin)[0, 0]
    scores = rq @ rk.T  # scores[m, n]
    for delta in (-3, 0, 2, 7):
        diag = torch.diagonal(scores, offset=-delta)  # all (m, n) with m - n == delta
        torch.testing.assert_close(diag, diag[0].expand_as(diag), **TOL)
    # ...and it genuinely changes with distance (not a constant).
    assert not torch.allclose(scores[5, 5], scores[5, 1], atol=1e-3)


def test_rope_offset_equals_slicing():
    torch.manual_seed(0)
    cos, sin = precompute_rope(8, 16)
    x = torch.randn(2, 2, 16, 8)
    full = apply_rope(x, cos, sin)
    torch.testing.assert_close(apply_rope(x[:, :, 5:9], cos, sin, offset=5), full[:, :, 5:9])
    torch.testing.assert_close(apply_rope(x[:, :, 15:], cos, sin, offset=15), full[:, :, 15:])


def test_rope_invalid_inputs_raise():
    with pytest.raises(ValueError):
        precompute_rope(7, 16)
    cos, sin = precompute_rope(8, 16)
    with pytest.raises(ValueError):
        apply_rope(torch.randn(1, 1, 4, 8), cos, sin, offset=13)  # needs positions 13..16


# ---------------------------------------------------------------- SwiGLU


def test_swiglu_formula_shape_and_grad():
    torch.manual_seed(0)
    ffn = SwiGLU(12, 40).to(DEVICE)
    assert all(lin.bias is None for lin in (ffn.w1, ffn.w2, ffn.w3))
    assert ffn.w1.weight.shape == ffn.w3.weight.shape == (40, 12)
    assert ffn.w2.weight.shape == (12, 40)
    x = torch.randn(2, 7, 12, device=DEVICE)
    out = ffn(x)
    assert out.shape == x.shape
    w1, w2, w3 = ffn.w1.weight, ffn.w2.weight, ffn.w3.weight
    torch.testing.assert_close(out, (F.silu(x @ w1.T) * (x @ w3.T)) @ w2.T, **TOL)
    out.square().sum().backward()
    for w in (w1, w2, w3):
        assert w.grad is not None and w.grad.abs().sum() > 0


# ---------------------------------------------------------------- GQA + KV cache


def _reference_attention(attn: GroupedQueryAttention, x: torch.Tensor) -> torch.Tensor:
    """Slow, explicit causal attention from the module's weights: loop over query heads."""
    bsz, t, _ = x.shape
    d = attn.head_dim
    n_heads, n_kv = attn.n_heads, attn.n_kv_heads
    q = (x @ attn.wq.weight.T).view(bsz, t, n_heads, d).transpose(1, 2)
    k = (x @ attn.wk.weight.T).view(bsz, t, n_kv, d).transpose(1, 2)
    v = (x @ attn.wv.weight.T).view(bsz, t, n_kv, d).transpose(1, 2)

    # Independent rotate-half RoPE at positions 0..t-1.
    inv_freq = 10000.0 ** (-torch.arange(0, d, 2, device=x.device).float() / d)
    ang = torch.arange(t, device=x.device).float()[:, None] * inv_freq
    cos, sin = torch.cat([ang, ang], -1).cos(), torch.cat([ang, ang], -1).sin()

    def rope(z: torch.Tensor) -> torch.Tensor:
        z1, z2 = z.chunk(2, dim=-1)
        return z * cos + torch.cat([-z2, z1], dim=-1) * sin

    q, k = rope(q), rope(k)
    causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
    heads = []
    for h in range(n_heads):
        g = h // (n_heads // n_kv)  # query head h reads kv head g
        scores = q[:, h] @ k[:, g].transpose(-1, -2) / math.sqrt(d)
        scores = scores.masked_fill(~causal, float("-inf"))
        heads.append(scores.softmax(dim=-1) @ v[:, g])
    out = torch.stack(heads, dim=1).transpose(1, 2).reshape(bsz, t, n_heads * d)
    return out @ attn.wo.weight.T


@pytest.mark.parametrize(("n_heads", "n_kv_heads"), [(4, 4), (4, 2)])
def test_gqa_matches_reference(n_heads, n_kv_heads):
    torch.manual_seed(0)
    attn = GroupedQueryAttention(32, n_heads, n_kv_heads, max_seq_len=64).to(DEVICE)
    x = torch.randn(2, 11, 32, device=DEVICE)
    out = attn(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, _reference_attention(attn, x), **TOL)


@pytest.mark.parametrize(("dim", "n_heads", "n_kv_heads"), [(32, 4, 3), (30, 4, 2)])
def test_invalid_head_config_raises(dim, n_heads, n_kv_heads):
    with pytest.raises(ValueError):
        GroupedQueryAttention(dim, n_heads, n_kv_heads)


def test_kv_cache_update_returns_valid_prefix():
    cache = KVCache(2, 3, 10, 4)
    assert cache.k.shape == cache.v.shape == (2, 3, 10, 4)
    k1, v1 = torch.randn(2, 3, 6, 4), torch.randn(2, 3, 6, 4)
    k_all, v_all = cache.update(k1, v1, start_pos=0)
    assert k_all.shape == (2, 3, 6, 4)
    k2, v2 = torch.randn(2, 3, 1, 4), torch.randn(2, 3, 1, 4)
    k_all, v_all = cache.update(k2, v2, start_pos=6)
    assert k_all.shape == v_all.shape == (2, 3, 7, 4)
    torch.testing.assert_close(k_all, torch.cat([k1, k2], dim=2))
    torch.testing.assert_close(v_all, torch.cat([v1, v2], dim=2))
    with pytest.raises(ValueError):
        cache.update(torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4), start_pos=7)


def _new_cache(attn: GroupedQueryAttention, bsz: int, max_len: int) -> KVCache:
    return KVCache(bsz, attn.n_kv_heads, max_len, attn.head_dim, device=DEVICE)


@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])  # MHA, GQA, MQA
def test_incremental_decoding_matches_full_forward(n_kv_heads):
    # KEY TEST: the optimized path (cache, one token at a time) == the simple path.
    torch.manual_seed(0)
    attn = GroupedQueryAttention(32, 4, n_kv_heads, max_seq_len=64).to(DEVICE).eval()
    x = torch.randn(2, 12, 32, device=DEVICE)
    with torch.no_grad():
        full = attn(x)
        cache = _new_cache(attn, 2, 16)
        steps = [attn(x[:, t : t + 1], cache=cache, start_pos=t) for t in range(12)]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, **TOL)
    # The cache holds only n_kv_heads heads: the GQA memory saving.
    assert cache.k.shape == (2, n_kv_heads, 16, 8)


def test_prefill_then_decode_matches_full_forward():
    # Prompt prefill (T>1 at start_pos 0), a chunked prefill (T>1 at start_pos>0), then decode.
    torch.manual_seed(0)
    attn = GroupedQueryAttention(32, 8, 2, max_seq_len=64).to(DEVICE).eval()
    x = torch.randn(3, 13, 32, device=DEVICE)
    with torch.no_grad():
        full = attn(x)
        cache = _new_cache(attn, 3, 32)
        outs = [attn(x[:, :5], cache=cache, start_pos=0)]
        outs.append(attn(x[:, 5:9], cache=cache, start_pos=5))
        outs += [attn(x[:, t : t + 1], cache=cache, start_pos=t) for t in range(9, 13)]
    torch.testing.assert_close(torch.cat(outs, dim=1), full, **TOL)
