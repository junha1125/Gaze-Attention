#!/bin/bash
# Stage 1-1: projector pretraining (384px, 576 visual tokens per image)
#
# Trains only the multimodal projector on Cambrian-Alignment with dense attention.
# The vision encoder and the language model are frozen.
# Output: ${OUTPUT_ROOT}/projectors/so400m384-${LLM_TAG}-stage1-1/mm_projector.bin
#
# Usage (from the repository root):
#   bash scripts/train/image_384/1_pretrain_projector.sh [additional training arguments]
# Additional arguments override the ones below, e.g. --save_steps 500 or --max_steps 1000.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

VISION_TOWER="${VISION_TOWER:-google/siglip2-so400m-patch16-384}"
RUN_NAME="so400m384-${LLM_TAG}-stage1-1"
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
    --mm_tunable_parts "mm_mlp_adapter" \
    --num_context_tokens 0 \
    --enable_gaze_attn False \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_train_epochs 1 \
    --learning_rate 1e-3 \
    --model_max_length 8192 \
    --save_strategy "no" \
    --output_dir "${OUTPUT_ROOT}/projectors/${RUN_NAME}" \
    --run_name "${RUN_NAME}" \
    "$@"
