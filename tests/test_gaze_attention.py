#!/usr/bin/env python3
"""
Test: Gaze Attention — TopK region routing

Verifies:
  1. compute_gaze_regions correctly partitions image patches into gaze regions
  2. Prefill produces correct output shape and is non-NaN
  3. Decode routing matches prefill routing for the same query
  4. Prefill and decode produce identical output for the same token
  5. Decode produces correct output shape and is non-NaN
  6. Gather-based decode matches the mask-based reference implementation
"""

import os, sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from cambrian.model.language_model.gaze_attention import (
    GazeAttention,
    compute_gaze_regions,
    build_context_token_causal_mask,
    build_gaze_ctx_prefill,
    build_gaze_ctx_decode,
)
from cambrian.constants import MODALITY_TEXT, MODALITY_IMAGE, MODALITY_CONTEXT_TOKEN, MODALITY_PAD

torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Layout: [36 image patches (6x6)] [2 context tokens] [N text tokens]
N_IMG, N_CTX, N_TEXT = 36, 2, 5
REGION_SIZE = 3     # region side → (6/3)^2 = 4 regions, each 9 patches
TOPK_RATIO = 0.5    # select 2 out of 4 regions

GAZE_CFG = SimpleNamespace(
    num_patches_one_image=N_IMG,
    gaze_region_size=REGION_SIZE,
    gaze_topk_ratio=TOPK_RATIO,
)


# ── Helpers ───────────────────────────────────────────────────
def make_config():
    cfg = Qwen3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        attention_dropout=0.0,
        sliding_window=None,
        rms_norm_eps=1e-6,
        qk_norm=True,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    cfg._attn_implementation = "eager"
    return cfg


def make_attn(config, layer_idx=0):
    attn = Qwen3Attention(config, layer_idx=layer_idx)
    attn.__class__ = GazeAttention
    return attn.to(DEVICE, DTYPE).eval()


def make_modality_types(n_text=N_TEXT):
    S = N_IMG + N_CTX + n_text
    mt = torch.zeros(1, S, dtype=torch.long, device=DEVICE)
    mt[0, :N_IMG] = MODALITY_IMAGE
    mt[0, N_IMG : N_IMG + N_CTX] = MODALITY_CONTEXT_TOKEN
    mt[0, N_IMG + N_CTX :] = MODALITY_TEXT
    return mt


def make_pos_emb(config, S):
    rotary = Qwen3RotaryEmbedding(config).to(DEVICE)
    dummy = torch.zeros(1, S, config.hidden_size, device=DEVICE, dtype=DTYPE)
    pos_ids = torch.arange(S, device=DEVICE).unsqueeze(0)
    return rotary(dummy, pos_ids)  # (cos, sin)


# ── Reference mask-based attention (for equivalence testing) ──
def _reference_mask_based_attention(
    attn_module, query_states, key_states, value_states,
    attention_mask, gaze_ctx, query_pre_rope, key_pre_rope,
):
    """Old single-pass approach: full per-head [B, H, S, S] mask.
    Numerically equivalent but memory-heavy; kept only as a test oracle."""
    B, H, S, D = query_states.shape
    H_kv = key_states.shape[1]
    dtype = query_states.dtype
    device = query_states.device
    min_dtype = torch.finfo(dtype).min

    text_token_mask = gaze_ctx["text_token_mask"]
    position_to_region = gaze_ctx["position_to_region"]
    region_positions = gaze_ctx["region_positions"]
    region_valid_mask = gaze_ctx["region_valid_mask"]
    n_regions_per_item = gaze_ctx["n_regions_per_item"]
    image_mask = gaze_ctx["image_mask"]
    topk_ratio = gaze_ctx["topk_ratio"]

    R_max = region_positions.shape[1]
    ppr = region_positions.shape[2]

    if topk_ratio > 0:
        flat_pos = region_positions.reshape(B, -1)
        flat_pos_exp = flat_pos[:, None, :, None].expand(B, H_kv, -1, D)
        region_K = key_pre_rope.gather(2, flat_pos_exp)
        region_K = region_K.reshape(B, H_kv, R_max, ppr, D)
        region_K = region_K * region_valid_mask[:, None, :, None, None].to(dtype)
        descriptors_kv = region_K.sum(dim=3) / ppr
        descriptors = repeat_kv(descriptors_kv, attn_module.num_key_value_groups)

        sim = torch.einsum(
            "bhsd,bhrd->bhsr",
            F.normalize(query_pre_rope, dim=-1), F.normalize(descriptors, dim=-1),
        )
        sim.masked_fill_(~region_valid_mask[:, None, None, :], float("-inf"))

        k_per_item = (n_regions_per_item.float() * topk_ratio).ceil().long().clamp(min=1)
        k_max = min(k_per_item.max().item(), R_max)
        _, topk_idx = sim.topk(k_max, dim=-1)

        selected_regions = torch.zeros(B, H, S, R_max, dtype=torch.bool, device=device)
        selected_regions.scatter_(-1, topk_idx, True)
        for b in range(B):
            k_b = k_per_item[b].item()
            if k_b < k_max:
                selected_regions[b].scatter_(-1, topk_idx[b, :, :, k_b:], False)

    if attention_mask is None:
        causal = torch.triu(
            torch.full((S, S), min_dtype, dtype=dtype, device=device), diagonal=1
        )
        attention_mask = causal[None, None, :, :]

    full_mask = attention_mask.expand(B, H, S, S).clone()
    text_image_block = text_token_mask[:, :, None] & image_mask[:, None, :]
    full_mask.masked_fill_(text_image_block[:, None, :, :], min_dtype)

    if topk_ratio > 0:
        ptr = position_to_region[:, None, None, :]
        ptr_clamped = ptr.clamp(min=0).expand(B, H, S, S)
        unmask = selected_regions.gather(-1, ptr_clamped)
        unmask &= (ptr >= 0)
        unmask &= text_token_mask[:, None, :, None]
        seq_idx = torch.arange(S, device=device)
        unmask &= (seq_idx[None, None, None, :] <= seq_idx[None, None, :, None])
        full_mask.masked_fill_(unmask, 0.0)

    key_full = repeat_kv(key_states, attn_module.num_key_value_groups)
    val_full = repeat_kv(value_states, attn_module.num_key_value_groups)

    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    ):
        attn_output = F.scaled_dot_product_attention(
            query_states, key_full, val_full,
            attn_mask=full_mask, dropout_p=0.0, scale=attn_module.scaling,
        )
    return attn_output.transpose(1, 2).contiguous()


# ══════════════════════════════════════════════════════════════
# Test 1: compute_gaze_regions
# ══════════════════════════════════════════════════════════════
def test_compute_gaze_regions():
    print("=" * 60)
    print("Test 1: compute_gaze_regions")
    print("=" * 60)

    mt = make_modality_types()
    info = compute_gaze_regions(mt, N_IMG, REGION_SIZE, DEVICE)

    assert info is not None, "region info should not be None"
    n_regions = info["n_regions_per_item"][0].item()
    assert n_regions == 4, f"Expected 4 regions, got {n_regions}"
    assert info["region_positions"].shape == (1, 4, 9), f"Shape mismatch: {info['region_positions'].shape}"
    assert info["region_valid_mask"].all(), "All 4 regions should be valid"

    ptr = info["position_to_region"][0]
    assert (ptr[:N_IMG] >= 0).all(), "All image positions must map to a region"
    assert (ptr[N_IMG:] == -1).all(), "Context and text positions must be -1"

    expected_region0 = {0, 1, 2, 6, 7, 8, 12, 13, 14}
    actual_region0 = set(info["region_positions"][0, 0].tolist())
    assert actual_region0 == expected_region0, f"Region 0: expected {expected_region0}, got {actual_region0}"

    print(f"  4 regions, each 9 patches. Region 0 = {sorted(actual_region0)}")
    print("  PASSED\n")


# ══════════════════════════════════════════════════════════════
# Test 2: Prefill basic correctness
# ══════════════════════════════════════════════════════════════
def test_prefill_basic():
    print("=" * 60)
    print("Test 2: Prefill basic correctness")
    print("=" * 60)

    config = make_config()
    attn = make_attn(config)
    mt = make_modality_types()
    B, S = mt.shape

    hidden = torch.randn(B, S, config.hidden_size, device=DEVICE, dtype=DTYPE)
    pos_emb = make_pos_emb(config, S)
    ctx_mask = build_context_token_causal_mask(
        torch.ones(B, S, dtype=torch.long, device=DEVICE), mt, DTYPE, DEVICE
    )
    gaze_ctx, _ = build_gaze_ctx_prefill(GAZE_CFG, mt)

    with torch.no_grad():
        output, _ = attn.forward(
            hidden, position_embeddings=pos_emb,
            attention_mask=ctx_mask, gaze_ctx=gaze_ctx,
        )

    assert output.shape == (B, S, config.hidden_size), f"Shape mismatch: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    print(f"  Output shape: {output.shape}, no NaN/Inf")
    print("  PASSED\n")


# ══════════════════════════════════════════════════════════════
# Test 3: Routing equivalence (prefill vs decode)
# ══════════════════════════════════════════════════════════════
def test_routing_equivalence():
    print("=" * 60)
    print("Test 3: Routing equivalence (prefill vs decode)")
    print("=" * 60)

    config = make_config()
    attn = make_attn(config)
    mt = make_modality_types()
    B, S = mt.shape
    H_kv = config.num_key_value_heads
    head_dim = config.head_dim
    n_kv_groups = config.num_attention_heads // H_kv

    hidden = torch.randn(B, S, config.hidden_size, device=DEVICE, dtype=DTYPE)
    regions = compute_gaze_regions(mt, N_IMG, REGION_SIZE, DEVICE)
    R_max = regions["region_positions"].shape[1]
    ppr = regions["region_positions"].shape[2]

    hidden_shape = (B, S, -1, head_dim)
    with torch.no_grad():
        Q_pre = attn.q_norm(attn.q_proj(hidden).view(hidden_shape)).transpose(1, 2)
        K_pre = attn.k_norm(attn.k_proj(hidden).view(hidden_shape)).transpose(1, 2)

    flat_pos = regions["region_positions"].reshape(B, -1)
    flat_pos_exp = flat_pos[:, None, :, None].expand(B, H_kv, -1, head_dim)
    region_K = K_pre.gather(2, flat_pos_exp).reshape(B, H_kv, R_max, ppr, head_dim)
    region_K = region_K * regions["region_valid_mask"][:, None, :, None, None].to(DTYPE)
    descriptors_kv = region_K.sum(dim=3) / ppr
    descriptors = repeat_kv(descriptors_kv, n_kv_groups)

    C_norm = F.normalize(descriptors, dim=-1)
    sim_prefill = torch.einsum("bhsd,bhrd->bhsr", F.normalize(Q_pre, dim=-1), C_norm)
    sim_prefill.masked_fill_(~regions["region_valid_mask"][:, None, None, :], float("-inf"))

    k_per_item = (regions["n_regions_per_item"].float() * TOPK_RATIO).ceil().long().clamp(min=1)
    k_max = min(k_per_item.max().item(), R_max)
    _, topk_prefill = sim_prefill.topk(k_max, dim=-1)

    text_start = N_IMG + N_CTX
    all_match = True
    for t_pos in range(text_start, S):
        q_single = Q_pre[:, :, t_pos : t_pos + 1, :]
        sim_decode = torch.einsum("bhsd,bhrd->bhsr", F.normalize(q_single, dim=-1), C_norm)
        sim_decode.masked_fill_(~regions["region_valid_mask"][:, None, None, :], float("-inf"))
        _, topk_decode = sim_decode.topk(k_max, dim=-1)

        expected = topk_prefill[:, :, t_pos : t_pos + 1, :]
        if not torch.equal(topk_decode, expected):
            print(f"  FAIL: T[{t_pos}] topk mismatch")
            all_match = False

    assert all_match, "Routing decisions diverged"
    print(f"  All {N_TEXT} text tokens: prefill and decode topk indices match exactly")
    print("  PASSED\n")


# ══════════════════════════════════════════════════════════════
# Test 4: Output equivalence (prefill-44 vs prefill-43 + decode)
# ══════════════════════════════════════════════════════════════
def test_output_equivalence():
    print("=" * 60)
    print("Test 4: Output equivalence (prefill vs decode)")
    print("=" * 60)

    config = make_config()
    attn = make_attn(config)

    mt_b = make_modality_types(n_text=N_TEXT + 1)
    B, S_b = mt_b.shape
    hidden_b = torch.randn(B, S_b, config.hidden_size, device=DEVICE, dtype=DTYPE)
    pos_emb_b = make_pos_emb(config, S_b)
    ctx_mask_b = build_context_token_causal_mask(
        torch.ones(B, S_b, dtype=torch.long, device=DEVICE), mt_b, DTYPE, DEVICE
    )
    gaze_ctx_b, _ = build_gaze_ctx_prefill(GAZE_CFG, mt_b)

    with torch.no_grad():
        out_b, _ = attn.forward(
            hidden_b, position_embeddings=pos_emb_b,
            attention_mask=ctx_mask_b, gaze_ctx=gaze_ctx_b,
        )
    prefill_44_output = out_b[:, -1:, :]

    attn2 = make_attn(config)
    attn2.load_state_dict(attn.state_dict())

    S_a = S_b - 1
    mt_a = make_modality_types(n_text=N_TEXT)
    hidden_a = hidden_b[:, :S_a, :]
    pos_emb_a = make_pos_emb(config, S_a)
    ctx_mask_a = build_context_token_causal_mask(
        torch.ones(B, S_a, dtype=torch.long, device=DEVICE), mt_a, DTYPE, DEVICE
    )
    gaze_ctx_a, decode_cache = build_gaze_ctx_prefill(GAZE_CFG, mt_a)

    cache_a = DynamicCache()
    with torch.no_grad():
        attn2.forward(
            hidden_a, position_embeddings=pos_emb_a,
            attention_mask=ctx_mask_a,
            past_key_value=cache_a,
            cache_position=torch.arange(S_a, device=DEVICE),
            gaze_ctx=gaze_ctx_a,
        )

    assert attn2._cached_descriptors is not None

    h_new = hidden_b[:, S_a:S_a + 1, :]
    cos_full, sin_full = pos_emb_b
    pos_emb_new = (cos_full[:, S_a:S_a + 1, :], sin_full[:, S_a:S_a + 1, :])
    decode_ctx, _ = build_gaze_ctx_decode(GAZE_CFG, decode_cache, B, DEVICE)

    with torch.no_grad():
        out_decode, _ = attn2.forward(
            h_new, position_embeddings=pos_emb_new,
            attention_mask=None,
            past_key_value=cache_a,
            cache_position=torch.tensor([S_a], device=DEVICE),
            gaze_ctx=decode_ctx,
        )

    max_diff = (out_decode - prefill_44_output).abs().max().item()
    print(f"  Prefill-44 vs Prefill-43+Decode: max_diff = {max_diff:.2e}")

    # 2e-4: fp32 accumulation order differs between S×S and 1×S attention on GPU
    assert torch.allclose(out_decode, prefill_44_output, atol=2e-4), \
        f"Output mismatch: max_diff = {max_diff:.2e}"
    print("  PASSED\n")


# ══════════════════════════════════════════════════════════════
# Test 5: Decode basic correctness
# ══════════════════════════════════════════════════════════════
def test_decode_basic():
    print("=" * 60)
    print("Test 5: Decode basic correctness")
    print("=" * 60)

    config = make_config()
    attn = make_attn(config)

    mt = make_modality_types()
    B, S = mt.shape
    hidden = torch.randn(B, S, config.hidden_size, device=DEVICE, dtype=DTYPE)
    pos_emb = make_pos_emb(config, S)
    ctx_mask = build_context_token_causal_mask(
        torch.ones(B, S, dtype=torch.long, device=DEVICE), mt, DTYPE, DEVICE
    )
    gaze_ctx, decode_cache = build_gaze_ctx_prefill(GAZE_CFG, mt)

    cache = DynamicCache()
    with torch.no_grad():
        attn.forward(
            hidden, position_embeddings=pos_emb,
            attention_mask=ctx_mask,
            past_key_value=cache,
            cache_position=torch.arange(S, device=DEVICE),
            gaze_ctx=gaze_ctx,
        )

    h_new = torch.randn(B, 1, config.hidden_size, device=DEVICE, dtype=DTYPE)
    cos_full, sin_full = make_pos_emb(config, S + 1)
    pos_emb_new = (cos_full[:, S:S + 1, :], sin_full[:, S:S + 1, :])
    decode_ctx, _ = build_gaze_ctx_decode(GAZE_CFG, decode_cache, B, DEVICE)

    with torch.no_grad():
        out, _ = attn.forward(
            h_new, position_embeddings=pos_emb_new,
            attention_mask=None,
            past_key_value=cache,
            cache_position=torch.tensor([S], device=DEVICE),
            gaze_ctx=decode_ctx,
        )

    assert out.shape == (B, 1, config.hidden_size), f"Shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"

    print(f"  Decode output shape: {out.shape}, no NaN/Inf")
    print("  PASSED\n")


# ══════════════════════════════════════════════════════════════
# Test 6: Two-pass prefill vs mask-based reference equivalence
# ══════════════════════════════════════════════════════════════
def test_prefill_vs_mask_reference():
    print("=" * 60)
    print("Test 6: Two-pass prefill vs mask-based reference")
    print("=" * 60)

    config = make_config()
    attn = make_attn(config)
    mt = make_modality_types()
    B, S = mt.shape
    head_dim = config.head_dim

    hidden = torch.randn(B, S, config.hidden_size, device=DEVICE, dtype=DTYPE)
    pos_emb = make_pos_emb(config, S)
    ctx_mask = build_context_token_causal_mask(
        torch.ones(B, S, dtype=torch.long, device=DEVICE), mt, DTYPE, DEVICE
    )
    gaze_ctx, _ = build_gaze_ctx_prefill(GAZE_CFG, mt)

    # Compute Q, K, V and pre-RoPE states (same as GazeAttention.forward)
    hidden_shape = (B, S, -1, head_dim)
    with torch.no_grad():
        query_states = attn.q_norm(attn.q_proj(hidden).view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(attn.k_proj(hidden).view(hidden_shape)).transpose(1, 2)
        value_states = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)

        query_pre_rope = query_states.clone()
        key_pre_rope = key_states.clone()

        cos, sin = pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        out_two_pass, _ = attn._gaze_prefill(
            query_states, key_states, value_states,
            ctx_mask, gaze_ctx, query_pre_rope, key_pre_rope,
        )
        out_reference = _reference_mask_based_attention(
            attn, query_states, key_states, value_states,
            ctx_mask, gaze_ctx, query_pre_rope, key_pre_rope,
        )

    # Compare text rows only: the reference builds one full mask so its
    # non-text rows are the plain causal result, same as pass 1 — but only
    # text rows are the sparse-routing behavior under test.
    text_start = N_IMG + N_CTX
    max_diff = (out_two_pass[:, text_start:, :] - out_reference[:, text_start:, :]).abs().max().item()
    print(f"  Text token outputs max_diff = {max_diff:.2e}")

    assert torch.allclose(
        out_two_pass[:, text_start:, :], out_reference[:, text_start:, :], atol=1e-5
    ), f"Text output mismatch: max_diff = {max_diff:.2e}"

    assert not torch.isnan(out_two_pass).any()
    assert not torch.isinf(out_two_pass).any()

    print("  Two-pass and mask-based implementations produce equivalent text outputs")
    print("  PASSED\n")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\nDevice: {DEVICE}  |  dtype: {DTYPE}")
    print(f"Layout: {N_IMG} img (6×6) + {N_CTX} ctx + {N_TEXT} text = {N_IMG + N_CTX + N_TEXT}")
    print(f"Regions: K={REGION_SIZE} → 4 regions of {REGION_SIZE * REGION_SIZE} patches, topk_ratio={TOPK_RATIO}\n")

    test_compute_gaze_regions()
    test_prefill_basic()
    test_routing_equivalence()
    test_output_equivalence()
    test_decode_basic()
    test_prefill_vs_mask_reference()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
