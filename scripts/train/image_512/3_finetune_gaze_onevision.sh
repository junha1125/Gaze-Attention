#!/bin/bash
# Stage 2: full finetuning with Gaze Attention on LLaVA-OneVision + Cambrian-7M data (512px)
#
# Same as 3_finetune_gaze_cambrian7m.sh, but trained on the LLaVA-OneVision single-image data
# combined with a 40% subset of Cambrian-7M (6.05M samples in total).
# data_preparation/README.md explains how to create the yaml file.
# Output: ${OUTPUT_ROOT}/finetune/so400m512-${LLM_TAG}-stage2-gaze-onevision
#
# Usage (from the repository root):
#   bash scripts/train/image_512/3_finetune_gaze_onevision.sh [additional training arguments]
# Additional arguments override the ones below, e.g. --save_steps 500 or --max_steps 1000.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

VISION_TOWER="${VISION_TOWER:-google/siglip2-so400m-patch16-512}"
INIT_PROJECTOR="${INIT_PROJECTOR:-${OUTPUT_ROOT}/projectors/so400m512-${LLM_TAG}-stage1-2-ctx4/mm_projector.bin}"
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/LLaVA-OneVision/onevision_cambrian.yaml}"
RUN_NAME="so400m512-${LLM_TAG}-stage2-gaze-onevision"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
TOTAL_BATCH="${TOTAL_BATCH:-256}"
GRAD_ACCUM="$(grad_accum "${TOTAL_BATCH}" "${PER_DEVICE_BATCH}")"

launch_training \
    --data_path "${DATA_PATH}" \
    --image_folder "${DATA_ROOT}/LLaVA-OneVision/datasets" \
    --version qwen_3 \
    --vision_tower "${VISION_TOWER}" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_patch_merge_type flat \
    --model_name_or_path "${LLM_VERSION}" \
    --pretrain_mm_mlp_adapter "${INIT_PROJECTOR}" \
    --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --num_context_tokens 4 \
    --enable_gaze_attn True \
    --gaze_region_size 8 \
    --gaze_topk_ratio 0.1 \
    --enable_progressive_topk True \
    --progressive_topk_start_ratio 0.9 \
    --progressive_topk_end_ratio 0.1 \
    --progressive_topk_decay_fraction 0.6 \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --mm_vision_tower_lr 2e-6 \
    --model_max_length 8192 \
    --save_strategy "steps" \
    --save_steps 1500 \
    --output_dir "${OUTPUT_ROOT}/finetune/${RUN_NAME}" \
    --run_name "${RUN_NAME}" \
    "$@"
