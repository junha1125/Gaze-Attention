# Gaze Attention: implementation notes

The whole method lives in one file, [`gaze_attention.py`](gaze_attention.py).

| File | Role |
|---|---|
| `gaze_attention.py` | Gaze regions, context-token attention mask, routing context builders, `GazeAttention` (Qwen3) / `GazeAttentionQwen2` (Qwen2.5) |
| `cambrian_qwen3.py`, `cambrian_qwen.py` | Cambrian model classes. `forward()` builds the mask and the routing context and hands them to the attention layers |

## Terms (paper -> code)

| Paper | Code |
|---|---|
| Gaze region (`r^h x r^w` patches) | `gaze_region_size` (K): K x K patch blocks, built by `compute_gaze_regions()` |
| Region descriptor | Mean of the pre-RoPE key vectors of a region (`descriptors_kv`) |
| Visual routing (TopK) | Cosine similarity between the text query and the descriptors, then TopK. Selected per layer, per head, per token. `gaze_topk_ratio` sets `TopK = ceil(ratio * n_regions)` |
| Learnable context tokens | `num_context_tokens` tokens appended after every image / frame; restricted to their own frame by `build_context_token_causal_mask()` |
| Progressive TopK schedule | `enable_progressive_topk`, implemented in `cambrian/train/cambrian_trainer.py` |

## Sequence layout

```
position  [frame0 patches x576] [ctx x4] [frame1 patches x576] [ctx x4] [question ... answer ...]
modality   IMAGE ...             CTX      IMAGE ...             CTX      TEXT ...
```

`prepare_inputs_labels_for_multimodal()` in `cambrian/model/cambrian_arch.py` produces this layout together with a
per-token `modality_types` tensor (`MODALITY_TEXT / IMAGE / CONTEXT_TOKEN / PAD`, see `cambrian/constants.py`).
With 576 patches (24 x 24) and K = 6, every frame has 4 x 4 = 16 gaze regions.
When batch items have different numbers of regions, regions are padded to `R_max` and excluded through
`region_valid_mask` (similarity `-inf`, so they can never be selected).

## Forward paths

| Phase | When | Routing context | Queries |
|---|---|---|---|
| Training / prefill | `modality_types` is available | `build_gaze_ctx_prefill()` (also returns the cache used while decoding) | all text tokens of the sequence |
| Decoding | one new token, KV cache present | `build_gaze_ctx_decode()` (reuses the cache, expands it for beam search) | the new token |

The routing context is passed as the keyword argument `gaze_ctx` through the Hugging Face decoder layers to every
`GazeAttention.forward()`.

## Attention per token type

- **Image and context tokens**: standard causal attention (with the context-token restriction in the mask).
- **Text tokens, prefill / training**: two passes. Pass 1 is the standard attention for all tokens with the shared
  `[B, 1, S, S]` mask. Pass 2 recomputes the text rows only, with a per-head `[B, H, T, S]` mask that hides every image
  patch outside the selected regions, and overwrites the outputs of pass 1 for those rows.
  A single pass with a per-head `[B, H, S, S]` mask does not fit in memory for long visual sequences.
- **Text tokens, decoding**: the key/value entries of the selected regions and of all non-image tokens
  (context tokens, text, generated tokens) are gathered from the cache, and attention runs over this small set
  (`S_small << S_kv`).

## Compatibility notes

- The vocabulary strings of the context tokens are `<compressed_context_i>` and the corresponding embedding rows are
  stored in `mm_projector.bin` under the key `compressed_context_embeddings`.
- Checkpoints saved with pre-release config names (`attn_chunk_select_ratio`, `patch_wh_one_chunk`, ...) are converted
  on load by `migrate_legacy_gaze_config()`.

## Tests

```bash
python tests/test_gaze_attention.py   # routing, prefill/decode consistency, equivalence with a mask-based reference
python tests/test_gaze_regions.py     # region partitioning for 384px / 512px / token compression / video
```
