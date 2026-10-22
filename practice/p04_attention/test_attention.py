import pytest
import torch
import torch.nn.functional as F
from torch import nn

from common import get_device
from practice.p04_attention.attention import (
    MultiHeadAttention,
    attention_weights,
    scaled_dot_product_attention,
)

DEVICE = get_device()
ATOL, RTOL = 1e-5, 1e-4  # float32 on DEVICE; different summation order than PyTorch


def qkv(*lead: int, L: int = 5, S: int = 5, E: int = 8, Ev: int = 8):
    torch.manual_seed(0)
    q = torch.randn(*lead, L, E, device=DEVICE)
    k = torch.randn(*lead, S, E, device=DEVICE)
    v = torch.randn(*lead, S, Ev, device=DEVICE)
    return q, k, v


# ---------------------------------------------------------------- scaled_dot_product_attention


def test_sdpa_matches_reference_no_mask():
    q, k, v = qkv(2, 3)
    out = scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=ATOL, rtol=RTOL)


def test_sdpa_multi_dim_leading_shapes_and_l_ne_s():
    q, k, v = qkv(2, 3, 4, L=3, S=7, E=8, Ev=5)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 3, 4, 3, 5)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=ATOL, rtol=RTOL)


def test_sdpa_bool_mask_true_means_keep():
    q, k, v = qkv(2, L=4, S=6)
    torch.manual_seed(1)
    mask = torch.rand(2, 4, 6, device=DEVICE) > 0.3
    mask[..., 0] = True  # every query keeps at least one key (no all-masked rows)
    out = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


def test_sdpa_float_mask_is_added_to_logits():
    q, k, v = qkv(2, 3, L=4, S=6)
    torch.manual_seed(2)
    bias = torch.randn(4, 6, device=DEVICE)  # broadcasts over the leading dims
    bias[:, 2] = float("-inf")
    out = scaled_dot_product_attention(q, k, v, attn_mask=bias)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(("L", "S"), [(5, 5), (3, 6)])
def test_sdpa_causal_matches_reference(L: int, S: int):
    q, k, v = qkv(2, 2, L=L, S=S)
    out = scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
    # Changing a "future" value must not change earlier outputs.
    v2 = v.clone()
    v2[..., -1, :] += 100.0
    out2 = scaled_dot_product_attention(q, k, v2, is_causal=True)
    torch.testing.assert_close(out2[..., : min(L, S - 1), :], out[..., : min(L, S - 1), :])


def test_mask_and_causal_together_raises():
    q, k, v = qkv(L=4, S=4)
    mask = torch.ones(4, 4, dtype=torch.bool, device=DEVICE)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)


def test_masked_out_key_gets_zero_weight_and_rows_sum_to_one():
    q, k, _ = qkv(2, L=4, S=6)
    mask = torch.ones(4, 6, dtype=torch.bool, device=DEVICE)
    mask[:, 3] = False  # key 3 is hidden from every query
    w = attention_weights(q, k, attn_mask=mask)
    assert w.shape == (2, 4, 6)
    assert torch.all(w[..., 3] == 0)
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 4, device=DEVICE))


def test_large_logits_stay_finite():
    q, k, v = qkv(2, L=4, S=6)
    q, k = q * 1e3, k * 1e3  # logits ~1e6: exp() overflows without the max trick
    out = scaled_dot_product_attention(q, k, v)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=1e-4, rtol=1e-4)


def test_sdpa_gradcheck():
    torch.manual_seed(3)
    shape = dict(dtype=torch.float64, requires_grad=True)
    q = torch.randn(2, 3, 4, **shape)
    k = torch.randn(2, 5, 4, **shape)
    v = torch.randn(2, 5, 3, **shape)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[:, 1] = False
    assert torch.autograd.gradcheck(
        lambda q, k, v: scaled_dot_product_attention(q, k, v, attn_mask=mask), (q, k, v)
    )


# ---------------------------------------------------------------- MultiHeadAttention


def make_pair(E: int = 16, H: int = 4, bias: bool = True):
    """Build our MHA and nn.MultiheadAttention with identical weights.

    nn.MultiheadAttention stores the q/k/v projections fused: in_proj_weight is (3E, E) =
    [W_q; W_k; W_v] stacked on dim 0 (in_proj_bias likewise, (3E,)), so chunk(3) splits it back.
    """
    torch.manual_seed(4)
    ref = nn.MultiheadAttention(E, H, bias=bias, batch_first=True).to(DEVICE).eval()
    mha = MultiHeadAttention(E, H, bias=bias).to(DEVICE).eval()
    projs = (mha.q_proj, mha.k_proj, mha.v_proj)
    with torch.no_grad():
        for proj, w in zip(projs, ref.in_proj_weight.chunk(3), strict=True):
            proj.weight.copy_(w)
        mha.out_proj.weight.copy_(ref.out_proj.weight)
        if bias:
            for proj, b in zip(projs, ref.in_proj_bias.chunk(3), strict=True):
                proj.bias.copy_(b)
            mha.out_proj.bias.copy_(ref.out_proj.bias)
    return mha, ref


@pytest.mark.parametrize("bias", [True, False])
def test_mha_matches_reference_self_attention(bias: bool):
    mha, ref = make_pair(bias=bias)
    x = torch.randn(3, 7, 16, device=DEVICE)
    out = mha(x, x, x)
    assert out.shape == (3, 7, 16)
    expected, _ = ref(x, x, x, need_weights=False)
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)


def test_mha_cross_attention_with_key_padding_mask():
    mha, ref = make_pair()
    query = torch.randn(2, 4, 16, device=DEVICE)
    memory = torch.randn(2, 6, 16, device=DEVICE)
    pad = torch.zeros(2, 6, dtype=torch.bool, device=DEVICE)
    pad[0, 4:] = True  # sequence 0 has length 4
    pad[1, 5:] = True  # sequence 1 has length 5
    out = mha(query, memory, memory, key_padding_mask=pad)
    expected, _ = ref(query, memory, memory, key_padding_mask=pad, need_weights=False)
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)
    # Padded positions really are ignored: changing them changes nothing.
    memory2 = memory.clone()
    memory2[0, 4:] = 1e3
    memory2[1, 5:] = -1e3
    torch.testing.assert_close(mha(query, memory2, memory2, key_padding_mask=pad), out)


def test_mha_causal_with_key_padding_mask():
    mha, ref = make_pair()
    x = torch.randn(2, 6, 16, device=DEVICE)
    pad = torch.zeros(2, 6, dtype=torch.bool, device=DEVICE)
    pad[1, 4:] = True  # right-padding: every query still sees key 0
    out = mha(x, key_padding_mask=pad, is_causal=True)  # key/value default to query
    # Careful: nn.MultiheadAttention's bool attn_mask uses True = NOT allowed (the opposite of
    # F.scaled_dot_product_attention), and it needs the mask even when is_causal=True.
    causal = torch.ones(6, 6, dtype=torch.bool, device=DEVICE).triu(diagonal=1)
    expected, _ = ref(
        x, x, x, attn_mask=causal, key_padding_mask=pad, is_causal=True, need_weights=False
    )
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)


def test_mha_dropout_only_in_training_mode():
    torch.manual_seed(6)
    mha = MultiHeadAttention(16, 4, dropout=0.5).to(DEVICE)
    x = torch.randn(2, 5, 16, device=DEVICE)
    mha.eval()
    torch.testing.assert_close(mha(x), mha(x))  # deterministic at inference
    eval_out = mha(x)
    mha.train()
    assert not torch.allclose(mha(x), eval_out)  # attention weights are dropped while training


@pytest.mark.parametrize(("E", "H"), [(10, 3), (16, 0), (8, 16)])
def test_mha_bad_head_count_raises(E: int, H: int):
    with pytest.raises(ValueError):
        MultiHeadAttention(E, H)


def test_mha_gradients_flow_to_all_parameters():
    torch.manual_seed(5)
    mha = MultiHeadAttention(16, 4, dropout=0.1).to(DEVICE).train()
    x = torch.randn(2, 5, 16, device=DEVICE, requires_grad=True)
    mha(x, is_causal=True).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name, p in mha.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    names = {n for n, _ in mha.named_parameters()}
    assert names == {
        f"{m}.{t}" for m in ("q_proj", "k_proj", "v_proj", "out_proj") for t in ("weight", "bias")
    }
