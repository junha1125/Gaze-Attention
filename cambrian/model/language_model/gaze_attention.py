"""Gaze Attention: dynamic TopK region routing for text-to-visual attention.

Paper: "Gaze Attention: Query-Adaptive Visual Routing for Efficient Multimodal LLMs"

  * Visual patches are spatially grouped into K x K "gaze regions"; each region
    is summarized by a mean-pooled key descriptor (pre-RoPE).
  * Every text query scores the descriptors (cosine similarity) and attends to
    only its TopK regions, per layer, per head and per token, so the gaze
    shifts across the image/video during generation. The full KV cache is
    retained; nothing is evicted.
  * A few learnable context tokens per image/frame summarize the whole scene
    and stay visible to text tokens throughout; each context token only attends
    to its own frame (see build_context_token_causal_mask).

Implementation notes:
  * Prefill/training: two-pass split. Pass 1 runs standard SDPA for all tokens
    with the head-uniform [B, 1, S, S] mask; Pass 2 re-runs attention for text
    rows only with a per-head [B, H, T, S] mask and overwrites their outputs.
    (A single-pass per-head [B, H, S, S] mask or gathered 5D KV runs out of memory.)
  * Decode: the selected regions' KV entries are physically gathered from the
    cache, so attention runs over S_small << S_kv.
"""

import torch
import torch.nn.functional as F
from typing import Callable, Dict, Optional, Tuple

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from cambrian.constants import MODALITY_IMAGE, MODALITY_CONTEXT_TOKEN, MODALITY_TEXT, MODALITY_PAD


# ---------------------------------------------------------------------------
# Legacy config migration (checkpoints saved with pre-release config key names)
# ---------------------------------------------------------------------------

_LEGACY_CONFIG_KEYS = {
    "enable_moc_sparse_attn": "enable_gaze_attn",
    "attn_chunk_select_ratio": "gaze_topk_ratio",
    "patch_wh_one_chunk": "gaze_region_size",
    "num_compressed_context": "num_context_tokens",
    "compressed_context_token_ids": "context_token_ids",
    "enable_gradual_sparsity": "enable_progressive_topk",
    "gradual_sparsity_start_ratio": "progressive_topk_start_ratio",
    "gradual_sparsity_end_ratio": "progressive_topk_end_ratio",
    "gradual_sparsity_decay_fraction": "progressive_topk_decay_fraction",
}


def migrate_legacy_gaze_config(config):
    """Rename pre-release config keys in place so older checkpoints keep working."""
    for old, new in _LEGACY_CONFIG_KEYS.items():
        if hasattr(config, old):
            if not hasattr(config, new):
                setattr(config, new, getattr(config, old))
            delattr(config, old)


# ---------------------------------------------------------------------------
# Gaze region partitioning
# ---------------------------------------------------------------------------

def compute_gaze_regions(
    modality_types: torch.Tensor,
    num_patches_one_image: int,
    region_size: int,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Partition image/video patches into K x K gaze regions.

    Each frame is a W x W patch grid (W = sqrt(num_patches_one_image)) stored in
    raster order; regions are K x K blocks of that grid, numbered consecutively
    across frames. Context tokens between frames are skipped automatically
    because only MODALITY_IMAGE positions are collected.

    Args:
        modality_types: [B, S] per-token modality (TEXT/IMAGE/CONTEXT/PAD).
        num_patches_one_image: patches per image or per video frame.
        region_size: K, the side length of one region in patches.
        device: device for the returned tensors.

    Returns None if the batch has no image tokens, else a dict:
        position_to_region  [B, S]             region index per position (-1 = not image)
        region_positions    [B, R_max, K*K]    sequence positions of each region's patches
        region_valid_mask   [B, R_max]         False for batch-padding regions
        n_regions_per_item  [B]                region count per batch item
        image_mask          [B, S]             MODALITY_IMAGE positions
    """
    B, S = modality_types.shape
    K = region_size

    image_mask = (modality_types.to(device) == MODALITY_IMAGE)
    if not image_mask.any():
        return None

    patches_per_frame = num_patches_one_image
    W = int(patches_per_frame ** 0.5)
    assert W * W == patches_per_frame, f"num_patches_one_image={patches_per_frame} is not a perfect square"
    assert W % K == 0, f"Grid width {W} not divisible by region size K={K}"

    R = W // K  # regions per axis (e.g. 24/6 -> 4)
    # Flat raster indices of one frame, grouped by region: [R*R regions, K*K patches]
    cr = torch.arange(R, device=device)
    r = torch.arange(K, device=device)
    frame_grid = (
        (cr[:, None, None, None] * K + r[None, None, :, None]) * W
        + cr[None, :, None, None] * K + r[None, None, None, :]
    ).reshape(R * R, K * K)

    position_to_region = torch.full((B, S), -1, dtype=torch.long, device=device)
    per_item_positions = []
    n_regions_list = []

    for b in range(B):
        image_pos = image_mask[b].nonzero(as_tuple=True)[0]
        n_patches = image_pos.numel()
        if n_patches == 0:
            per_item_positions.append(None)
            n_regions_list.append(0)
            continue

        n_frames = n_patches // patches_per_frame
        assert n_patches == n_frames * patches_per_frame, (
            f"Batch {b}: {n_patches} image patches not divisible by patches_per_frame={patches_per_frame}. "
            "An input that is longer than tokenizer_model_max_length is truncated, which cuts an image or frame: "
            "use fewer images / frames or a larger tokenizer_model_max_length."
        )

        # [F, R*R, K*K] -> [n_regions, K*K] sequence positions, regions numbered across frames
        region_pos = image_pos.view(n_frames, patches_per_frame)[:, frame_grid]
        region_pos = region_pos.reshape(n_frames * R * R, K * K)
        n_regions = region_pos.shape[0]

        region_ids = torch.arange(n_regions, device=device).repeat_interleave(K * K)
        position_to_region[b, region_pos.reshape(-1)] = region_ids

        per_item_positions.append(region_pos)
        n_regions_list.append(n_regions)

    R_max = max(n_regions_list)
    if R_max == 0:
        return None

    # Pad ragged region counts to R_max. Padding regions keep position 0; they are
    # zeroed out of the mean-pool and masked to -inf in similarity, so they can
    # never be selected.
    region_positions = torch.zeros(B, R_max, K * K, dtype=torch.long, device=device)
    region_valid_mask = torch.zeros(B, R_max, dtype=torch.bool, device=device)
    for b, region_pos in enumerate(per_item_positions):
        if region_pos is not None:
            region_positions[b, : region_pos.shape[0]] = region_pos
            region_valid_mask[b, : region_pos.shape[0]] = True

    return {
        "position_to_region": position_to_region,
        "region_positions": region_positions,
        "region_valid_mask": region_valid_mask,
        "n_regions_per_item": torch.tensor(n_regions_list, dtype=torch.long, device=device),
        "image_mask": image_mask,
    }


# ---------------------------------------------------------------------------
# Context-token restricted causal mask
# ---------------------------------------------------------------------------

def build_context_token_causal_mask(
    attention_mask_2d: Optional[torch.Tensor],
    modality_types: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Causal mask where each context token attends only to its own frame.

    Context tokens (C) see their frame's image patches and earlier same-frame
    context tokens; all other rows keep the standard causal mask::

        Q\\K   T0  I1  I2  C3  I4  I5  C6  T7
         T0    .   X   X   X   X   X   X   X
         I1    .   .   X   X   X   X   X   X
         I2    .   .   .   X   X   X   X   X
         C3    X   .   .   .   X   X   X   X
         I4    .   .   .   .   .   X   X   X
         I5    .   .   .   .   .   .   X   X
         C6    X   X   X   X   .   .   .   X
         T7    .   .   .   .   .   .   .   .

    Args:
        attention_mask_2d: [B, S] with 1=valid / 0=pad, or None.
        modality_types:    [B, S] modality per token.

    Returns:
        [B, 1, S, S] additive mask (0 = attend, min_dtype = masked).
    """
    batch_size, seq_len = modality_types.shape
    min_dtype = torch.finfo(dtype).min

    row_idx = torch.arange(seq_len, device=device)
    causal_mask = torch.where(
        row_idx.unsqueeze(1) >= row_idx.unsqueeze(0),
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(min_dtype, dtype=dtype, device=device),
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).clone()

    if attention_mask_2d is not None:
        causal_mask.masked_fill_((attention_mask_2d == 0)[:, None, None, :], min_dtype)

    # Frame ids: a new frame starts where an IMAGE token follows a non-IMAGE token.
    # cumsum gives 1-based ids for visual tokens; non-visual tokens stay 0.
    is_image = (modality_types == MODALITY_IMAGE)
    is_ctx = (modality_types == MODALITY_CONTEXT_TOKEN)
    prev_is_image = F.pad(is_image[:, :-1], (1, 0), value=False)
    frame_starts = is_image & ~prev_is_image
    frame_ids = frame_starts.long().cumsum(dim=1) * (is_image | is_ctx).long()

    # For context-token query rows, mask keys outside the same frame.
    same_frame = (frame_ids.unsqueeze(2) == frame_ids.unsqueeze(1))
    restrict = is_ctx.unsqueeze(2) & ~same_frame
    causal_mask.masked_fill_(restrict.unsqueeze(1), min_dtype)

    return causal_mask


# ---------------------------------------------------------------------------
# Per-forward routing context (shared glue for CambrianQwen / CambrianQwen3)
# ---------------------------------------------------------------------------

def build_gaze_ctx_prefill(config, modality_types) -> Tuple[Optional[dict], Optional[dict]]:
    """Build the routing context for training/prefill.

    Returns (gaze_ctx, decode_cache); both None when gaze attention does not
    apply to this batch (no text or no image tokens).
    """
    if modality_types is None:
        return None, None
    text_token_mask = (modality_types == MODALITY_TEXT)
    if not text_token_mask.any():
        return None, None

    regions = compute_gaze_regions(
        modality_types, config.num_patches_one_image, config.gaze_region_size, modality_types.device
    )
    if regions is None:
        return None, None

    gaze_ctx = {
        "text_token_mask": text_token_mask,
        "topk_ratio": config.gaze_topk_ratio,
        "decode_mode": False,
        "valid_token_mask": (modality_types != MODALITY_PAD),
        **regions,
    }
    decode_cache = {k: v for k, v in gaze_ctx.items() if k not in ("text_token_mask", "topk_ratio", "decode_mode")}
    decode_cache["S_prefill"] = modality_types.shape[1]
    return gaze_ctx, decode_cache


def build_gaze_ctx_decode(config, decode_cache, batch_size, device) -> Tuple[Optional[dict], Optional[dict]]:
    """Rebuild the routing context from the prefill cache for one decode step.

    Expands the cache along batch dim when generation runs with multiple beams.
    Returns (gaze_ctx, possibly-expanded decode_cache).
    """
    if decode_cache is None:
        return None, None

    cached_B = decode_cache["n_regions_per_item"].shape[0]
    if batch_size != cached_B and batch_size % cached_B == 0:
        num_beams = batch_size // cached_B
        decode_cache = {
            k: v.repeat_interleave(num_beams, dim=0) if isinstance(v, torch.Tensor) else v
            for k, v in decode_cache.items()
        }

    gaze_ctx = {
        "text_token_mask": torch.ones(batch_size, 1, dtype=torch.bool, device=device),
        "topk_ratio": config.gaze_topk_ratio,
        "decode_mode": True,
        **decode_cache,
    }
    return gaze_ctx, decode_cache


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class _GazeAttentionMixin:
    """Gaze routing shared by the Qwen2/Qwen3 attention subclasses.

    Subclasses implement `_compute_qkv` and `_get_sliding_window`. Behaviour:
    image/context tokens always use standard (context-restricted) causal
    attention; text tokens attend to non-image tokens plus their TopK regions.
    """

    def _compute_qkv(self, hidden_states, hidden_shape):
        raise NotImplementedError

    def _get_sliding_window(self):
        raise NotImplementedError

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        gaze_ctx = kwargs.pop("gaze_ctx", None)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, key_states, value_states = self._compute_qkv(hidden_states, hidden_shape)

        # Pre-RoPE Q/K are the routing features (region scoring is position-free)
        query_pre_rope = query_states
        key_pre_rope = key_states

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        should_route = (
            gaze_ctx is not None
            and gaze_ctx["text_token_mask"].any()
            and gaze_ctx["n_regions_per_item"].sum() > 0
        )

        if not should_route:
            attn_output, attn_weights = self._standard_attention(
                query_states, key_states, value_states, attention_mask, **kwargs
            )
        elif gaze_ctx["decode_mode"]:
            attn_output, attn_weights = self._gaze_decode(
                query_states, key_states, value_states, gaze_ctx, query_pre_rope
            )
        else:
            attn_output, attn_weights = self._gaze_prefill(
                query_states, key_states, value_states, attention_mask, gaze_ctx,
                query_pre_rope, key_pre_rope,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    # ------------------------------------------------------------------
    def _standard_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                pass  # fall back to eager
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        return attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self._get_sliding_window(),
            **kwargs,
        )

    # ------------------------------------------------------------------
    def _gaze_prefill(
        self,
        query_states: torch.Tensor,              # [B, H, S, D] post-RoPE
        key_states: torch.Tensor,                # [B, H_kv, S, D] post-RoPE
        value_states: torch.Tensor,              # [B, H_kv, S, D]
        attention_mask: Optional[torch.Tensor],  # [B, 1, S, S] or None
        gaze_ctx: Dict[str, torch.Tensor],
        query_pre_rope: torch.Tensor,            # [B, H, S, D]
        key_pre_rope: torch.Tensor,              # [B, H_kv, S, D]
    ) -> Tuple[torch.Tensor, None]:
        """Two-pass prefill: standard attention for all tokens, then sparse
        re-attention for text rows (T << S) whose outputs overwrite pass 1."""
        B, H, S, D = query_states.shape
        H_kv = key_states.shape[1]
        dtype = query_states.dtype
        device = query_states.device
        min_dtype = torch.finfo(dtype).min

        text_token_mask = gaze_ctx["text_token_mask"]        # [B, S]
        position_to_region = gaze_ctx["position_to_region"]  # [B, S]
        region_positions = gaze_ctx["region_positions"]      # [B, R_max, K*K]
        region_valid_mask = gaze_ctx["region_valid_mask"]    # [B, R_max]
        n_regions_per_item = gaze_ctx["n_regions_per_item"]  # [B]
        image_mask = gaze_ctx["image_mask"]                  # [B, S]
        topk_ratio = gaze_ctx["topk_ratio"]

        R_max = region_positions.shape[1]
        patches_per_region = region_positions.shape[2]

        # ==== Pass 1: standard attention for ALL tokens ====================
        # The context-restricted mask is head-uniform -> Flash/MemEfficient OK
        attn_output_std, _ = self._standard_attention(
            query_states, key_states, value_states, attention_mask,
        )                                                    # [B, S, H, D]

        # ==== Text token positions (padded to max_T; pad rows use pos 0) ===
        n_text_per_item = text_token_mask.sum(dim=1)         # [B]
        max_T = n_text_per_item.max().item()
        if max_T == 0:
            return attn_output_std, None

        text_positions = torch.zeros(B, max_T, dtype=torch.long, device=device)
        text_valid = torch.zeros(B, max_T, dtype=torch.bool, device=device)
        for b in range(B):
            pos = text_token_mask[b].nonzero(as_tuple=True)[0]
            text_positions[b, : pos.shape[0]] = pos
            text_valid[b, : pos.shape[0]] = True

        tp_exp = text_positions[:, None, :, None].expand(B, H, max_T, D)
        Q_text = query_states.gather(2, tp_exp)              # [B, H, max_T, D]

        # ==== Routing: descriptors -> similarity -> TopK regions ===========
        if topk_ratio > 0:
            # Region descriptors: mean-pooled pre-RoPE keys ([B, H_kv, R_max, D])
            flat_pos = region_positions.reshape(B, -1)
            flat_pos_exp = flat_pos[:, None, :, None].expand(B, H_kv, -1, D)
            region_K = key_pre_rope.gather(2, flat_pos_exp)
            region_K = region_K.reshape(B, H_kv, R_max, patches_per_region, D)
            region_K = region_K * region_valid_mask[:, None, :, None, None].to(dtype)
            descriptors_kv = region_K.sum(dim=3) / patches_per_region

            # Cache for decode steps
            self._cached_descriptors = descriptors_kv.detach()

            descriptors = repeat_kv(descriptors_kv, self.num_key_value_groups)  # [B, H, R_max, D]

            # Cosine similarity of text queries vs descriptors (pre-RoPE)
            Q_text_pre = query_pre_rope.gather(2, tp_exp)
            sim = torch.einsum(
                "bhtd,bhrd->bhtr", F.normalize(Q_text_pre, dim=-1), F.normalize(descriptors, dim=-1)
            )                                                # [B, H, max_T, R_max]
            sim.masked_fill_(~region_valid_mask[:, None, None, :], float("-inf"))
            sim.masked_fill_(~text_valid[:, None, :, None], float("-inf"))

            # TopK selection; k varies per item, so mark k_max then un-mark extras
            k_per_item = (n_regions_per_item.float() * topk_ratio).ceil().long().clamp(min=1)
            k_max = min(k_per_item.max().item(), R_max)
            _, topk_idx = sim.topk(k_max, dim=-1)            # [B, H, max_T, k_max]

            selected_regions = torch.zeros(B, H, max_T, R_max, dtype=torch.bool, device=device)
            selected_regions.scatter_(-1, topk_idx, True)
            for b in range(B):
                k_b = k_per_item[b].item()
                if k_b < k_max:
                    selected_regions[b].scatter_(-1, topk_idx[b, :, :, k_b:], False)

        # ==== Per-head text mask [B, H, max_T, S] ==========================
        if attention_mask is None:
            causal = torch.triu(
                torch.full((S, S), min_dtype, dtype=dtype, device=device), diagonal=1
            )
            attention_mask = causal[None, None, :, :]

        base_mask = attention_mask.expand(B, 1, S, S)
        tp_row_exp = text_positions[:, None, :, None].expand(B, 1, max_T, S)
        text_mask = base_mask.gather(2, tp_row_exp)          # text rows of the base mask
        text_mask = text_mask.expand(B, H, max_T, S).clone()

        # Close all image-patch columns for valid text rows...
        img_col_mask = text_valid[:, None, :, None] & image_mask[:, None, None, :]
        text_mask.masked_fill_(img_col_mask, min_dtype)

        # ...then reopen the selected regions' patches (within the causal bound)
        if topk_ratio > 0:
            ptr = position_to_region[:, None, None, :]       # [B, 1, 1, S]
            ptr_clamped = ptr.clamp(min=0).expand(B, H, max_T, S)
            unmask = selected_regions.gather(-1, ptr_clamped)
            unmask &= (ptr >= 0)
            unmask &= text_valid[:, None, :, None]
            seq_idx = torch.arange(S, device=device)
            unmask &= (seq_idx[None, None, None, :] <= text_positions[:, None, :, None])
            text_mask.masked_fill_(unmask, 0.0)
            del unmask

        # ==== Pass 2: sparse attention for text rows (math SDPA) ===========
        key_states_full = repeat_kv(key_states, self.num_key_value_groups)
        value_states_full = repeat_kv(value_states, self.num_key_value_groups)

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            sparse_output = F.scaled_dot_product_attention(
                Q_text.contiguous(),
                key_states_full.contiguous(),
                value_states_full.contiguous(),
                attn_mask=text_mask,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                scale=self.scaling,
            )                                                # [B, H, max_T, D]

        # ==== Merge: overwrite text positions in the pass-1 output =========
        attn_output = attn_output_std.transpose(1, 2).clone()  # [B, H, S, D]
        for b in range(B):
            n = n_text_per_item[b].item()
            if n > 0:
                attn_output[b, :, text_positions[b, :n], :] = sparse_output[b, :, :n, :]

        return attn_output.transpose(1, 2).contiguous(), None

    # ------------------------------------------------------------------
    def _gaze_decode(
        self,
        query_states: torch.Tensor,       # [B, H, 1, D] post-RoPE
        key_states: torch.Tensor,         # [B, H_kv, S_kv, D] full cache
        value_states: torch.Tensor,       # [B, H_kv, S_kv, D]
        gaze_ctx: Dict[str, torch.Tensor],
        query_pre_rope: torch.Tensor,     # [B, H, 1, D]
    ) -> Tuple[torch.Tensor, None]:
        """Gather-based decode: physically collect the selected regions' KV plus
        all mandatory (non-image) KV and attend over S_small << S_kv."""
        B, H, _, D = query_states.shape
        H_kv = key_states.shape[1]
        S_kv = key_states.shape[2]
        dtype = query_states.dtype
        device = query_states.device
        min_dtype = torch.finfo(dtype).min

        topk_ratio = gaze_ctx["topk_ratio"]
        region_valid_mask = gaze_ctx["region_valid_mask"]    # [B, R_max]
        n_regions_per_item = gaze_ctx["n_regions_per_item"]  # [B]
        image_mask_prefill = gaze_ctx["image_mask"]          # [B, S_prefill]
        valid_token_mask_prefill = gaze_ctx["valid_token_mask"]
        S_prefill = gaze_ctx["S_prefill"]

        R_max = region_valid_mask.shape[1]

        # Extend prefill masks to S_kv: decoded tokens are valid non-image
        pad_len = S_kv - S_prefill
        if pad_len > 0:
            image_mask = F.pad(image_mask_prefill, (0, pad_len), value=False)
            valid_token_mask = F.pad(valid_token_mask_prefill, (0, pad_len), value=True)
        else:
            image_mask = image_mask_prefill
            valid_token_mask = valid_token_mask_prefill

        # ==== Routing: score cached descriptors with the current query =====
        if topk_ratio > 0:
            descriptors_kv = self._cached_descriptors  # pre-RoPE, from prefill
            if descriptors_kv.shape[0] != B and B % descriptors_kv.shape[0] == 0:
                num_beams = B // descriptors_kv.shape[0]
                descriptors_kv = descriptors_kv.repeat_interleave(num_beams, dim=0)
                self._cached_descriptors = descriptors_kv

            descriptors = repeat_kv(descriptors_kv, self.num_key_value_groups)
            sim = torch.einsum(
                "bhsd,bhrd->bhsr", F.normalize(query_pre_rope, dim=-1), F.normalize(descriptors, dim=-1)
            )
            sim.masked_fill_(~region_valid_mask[:, None, None, :], float("-inf"))

            k_per_item = (n_regions_per_item.float() * topk_ratio).ceil().long()
            k_per_item = torch.where(n_regions_per_item > 0, k_per_item.clamp(min=1), k_per_item)
            k_max = min(k_per_item.max().item(), R_max)
            _, topk_idx = sim.topk(k_max, dim=-1)            # [B, H, 1, k_max]

            for b in range(B):
                k_b = k_per_item[b].item()
                if k_b < k_max:
                    topk_idx[b, :, :, k_b:] = 0

        # ==== Gather selected regions' K, V ================================
        key_states_full = repeat_kv(key_states, self.num_key_value_groups)
        value_states_full = repeat_kv(value_states, self.num_key_value_groups)

        if topk_ratio > 0:
            region_positions = gaze_ctx["region_positions"]
            patches_per_region = region_positions.shape[2]
            n_sel = k_max * patches_per_region

            topk_idx_exp = topk_idx.unsqueeze(-1).expand(B, H, 1, k_max, patches_per_region)
            rp_exp = region_positions[:, None, None, :, :].expand(B, H, 1, R_max, patches_per_region)
            selected_patch_pos = rp_exp.gather(3, topk_idx_exp).reshape(B, H, 1, n_sel)

            K_flat = key_states_full.reshape(B * H, S_kv, D)
            V_flat = value_states_full.reshape(B * H, S_kv, D)
            del key_states_full, value_states_full

            spp_exp = selected_patch_pos.reshape(B * H, n_sel).unsqueeze(-1).expand(-1, -1, D)
            gathered_K_img = K_flat.gather(1, spp_exp).reshape(B, H, 1, n_sel, D)
            gathered_V_img = V_flat.gather(1, spp_exp).reshape(B, H, 1, n_sel, D)
            del K_flat, V_flat

            # Invalidate slots past each item's own k (k_b < k_max)
            img_valid = torch.ones(B, 1, 1, n_sel, dtype=torch.bool, device=device)
            for b in range(B):
                k_b = k_per_item[b].item()
                if k_b < k_max:
                    img_valid[b, :, :, k_b * patches_per_region:] = False
        else:
            del key_states_full, value_states_full

        # ==== Gather mandatory K, V (all valid non-image tokens) ===========
        mandatory_mask = valid_token_mask & ~image_mask      # context + text + decoded
        n_mandatory_per_item = mandatory_mask.sum(dim=1)
        max_n_mandatory = n_mandatory_per_item.max().item()

        mandatory_positions = torch.zeros(B, max_n_mandatory, dtype=torch.long, device=device)
        mandatory_valid = torch.zeros(B, max_n_mandatory, dtype=torch.bool, device=device)
        for b in range(B):
            pos = mandatory_mask[b].nonzero(as_tuple=True)[0]
            mandatory_positions[b, : pos.shape[0]] = pos
            mandatory_valid[b, : pos.shape[0]] = True

        mp_exp = mandatory_positions[:, None, :, None].expand(B, H_kv, max_n_mandatory, D)
        mandatory_K = repeat_kv(key_states.gather(2, mp_exp), self.num_key_value_groups).unsqueeze(2)
        mandatory_V = repeat_kv(value_states.gather(2, mp_exp), self.num_key_value_groups).unsqueeze(2)

        # ==== Concat + attention over the small KV set =====================
        if topk_ratio > 0:
            S_small = n_sel + max_n_mandatory
            all_K = torch.cat([gathered_K_img, mandatory_K], dim=3)  # [B, H, 1, S_small, D]
            all_V = torch.cat([gathered_V_img, mandatory_V], dim=3)
        else:
            n_sel = 0
            S_small = max_n_mandatory
            all_K = mandatory_K.contiguous()
            all_V = mandatory_V.contiguous()

        attn_mask = torch.full((B, H, 1, S_small), min_dtype, dtype=dtype, device=device)
        if topk_ratio > 0:
            attn_mask[:, :, :, :n_sel].masked_fill_(img_valid.expand(B, H, 1, n_sel), 0.0)
        attn_mask[:, :, :, n_sel:].masked_fill_(
            mandatory_valid[:, None, None, :].expand(B, H, 1, max_n_mandatory), 0.0
        )

        scores = torch.einsum("bhtd,bhtkd->bhtk", query_states, all_K) * self.scaling
        scores = scores + attn_mask
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(dtype)
        attn_output = torch.einsum("bhtk,bhtkd->bhtd", attn_weights, all_V)

        return attn_output.transpose(1, 2).contiguous(), None


class GazeAttention(_GazeAttentionMixin, Qwen3Attention):
    """Qwen3Attention + gaze routing."""

    def _compute_qkv(self, hidden_states, hidden_shape):
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    def _get_sliding_window(self):
        return self.sliding_window


class GazeAttentionQwen2(_GazeAttentionMixin, Qwen2Attention):
    """Qwen2Attention + gaze routing."""

    def _compute_qkv(self, hidden_states, hidden_shape):
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    def _get_sliding_window(self):
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            return self.config.sliding_window
        return None
