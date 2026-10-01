#!/bin/bash
# Stage 1-2: learnable context tokens (512px, 1024 visual tokens per image)
#
# Continues from the stage 1-1 projector and adds 4 learnable context tokens per image.
# Gaze attention is enabled with TopK ratio 0, i.e. text tokens cannot attend to image
# patches at all and have to read the image through the context tokens.
# Trained: projector + context-token embeddings. Frozen: vision encoder, language model.
# Output: ${OUTPUT_ROOT}/projectors/so400m512-${LLM_TAG}-stage1-2-ctx4/mm_projector.bin
#
# Usage (from the repository root):
#   bash scripts/train/image_512/2_pretrain_context_tokens.sh [additional training arguments]
# Additional arguments override the ones below, e.g. --save_steps 500 or --max_steps 1000.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

VISION_TOWER="${VISION_TOWER:-google/siglip2-so400m-patch16-512}"
INIT_PROJECTOR="${INIT_PROJECTOR:-${OUTPUT_ROOT}/projectors/so400m512-${LLM_TAG}-stage1-1/mm_projector.bin}"
RUN_NAME="so400m512-${LLM_TAG}-stage1-2-ctx4"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-16}"
TOTAL_BATCH="${TOTAL_BATCH:-512}"
GRAD_ACCUM="$(grad_accum "${TOTAL_BATCH}" "${PER_DEVICE_BATCH}")"

launch_training \
    --data_path "${DATA_ROOT}/Cambrian-Alignment/jsons/alignment_filtered.jsonl" \
    --image_folder "${DATA_ROOT}/Cambrian-Alignment" \
    --version plain \
    --vision_tower "${VISION_TOWER}" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_patch_merge_type flat \
    --model_name_or_path "${LLM_VERSION}" \
    --pretrain_mm_mlp_adapter "${INIT_PROJECTOR}" \
    --mm_tunable_parts "mm_mlp_adapter" \
    --num_context_tokens 4 \
    --enable_gaze_attn True \
    --gaze_region_size 8 \
    --gaze_topk_ratio 0.0 \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_train_epochs 1 \
    --learning_rate 1e-3 \
    --model_max_length 8192 \
    --save_strategy "no" \
    --output_dir "${OUTPUT_ROOT}/projectors/${RUN_NAME}" \
    --run_name "${RUN_NAME}" \
    "$@"
