#!/bin/bash
# Dense-attention video baseline, stage 3: video finetuning
#
# Continues from the image-finetuned baseline on the Cambrian-S-3M subset with dense attention.
# Output: ${OUTPUT_ROOT}/finetune/video-so400m384-${LLM_TAG}-baseline-stage3
#
# Usage (from the repository root):
#   bash scripts/train/video_384/baseline_3_finetune_video.sh [additional training arguments]
# Additional arguments override the ones below, e.g. --save_steps 500 or --max_steps 1000.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

VISION_TOWER="${VISION_TOWER:-google/siglip2-so400m-patch16-384}"
INIT_MODEL="${INIT_MODEL:-${OUTPUT_ROOT}/finetune/video-so400m384-${LLM_TAG}-baseline-stage2}"
VIDEO_DATA_PATH="${VIDEO_DATA_PATH:-${DATA_ROOT}/Cambrian-S-3M/cambrian_s_3m_clean_under60m_30p_with_text.jsonl}"
RUN_NAME="video-so400m384-${LLM_TAG}-baseline-stage3"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
TOTAL_BATCH="${TOTAL_BATCH:-256}"
GRAD_ACCUM="$(grad_accum "${TOTAL_BATCH}" "${PER_DEVICE_BATCH}")"

launch_training \
    --data_path "${VIDEO_DATA_PATH}" \
    --image_folder "${DATA_ROOT}/Cambrian-S-3M" \
    --video_folder "${DATA_ROOT}/Cambrian-S-3M" \
    --video_fps 1 \
    --frames_upbound 64 \
    --version qwen_3 \
    --vision_tower "${VISION_TOWER}" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu_conv2x \
    --mm_patch_merge_type flat \
    --model_name_or_path "${INIT_MODEL}" \
    --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --mm_vision_tower_lr 2e-6 \
    --model_max_length 12288 \
    --save_strategy "steps" \
    --save_steps 400 \
    --output_dir "${OUTPUT_ROOT}/finetune/${RUN_NAME}" \
    --run_name "${RUN_NAME}" \
    "$@"
