#!/bin/bash
# Dense-attention baseline, stage 2: full finetuning (512px)
#
# Full-parameter finetuning with standard dense attention (no context tokens, no routing).
# DATA_PATH / IMAGE_FOLDER select the training set (default: Cambrian-7M).
# Output: ${OUTPUT_ROOT}/finetune/so400m512-${LLM_TAG}-baseline
#
# Usage (from the repository root):
#   bash scripts/train/image_512/baseline_2_finetune.sh [additional training arguments]
# Additional arguments override the ones below, e.g. --save_steps 500 or --max_steps 1000.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

VISION_TOWER="${VISION_TOWER:-google/siglip2-so400m-patch16-512}"
INIT_PROJECTOR="${INIT_PROJECTOR:-${OUTPUT_ROOT}/projectors/so400m512-${LLM_TAG}-baseline/mm_projector.bin}"
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/Cambrian-10M/jsons/Cambrian7M_withsystemprompt_filtered.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_ROOT}/Cambrian-10M}"
RUN_NAME="so400m512-${LLM_TAG}-baseline"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
TOTAL_BATCH="${TOTAL_BATCH:-256}"
GRAD_ACCUM="$(grad_accum "${TOTAL_BATCH}" "${PER_DEVICE_BATCH}")"

launch_training \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --version qwen_3 \
    --vision_tower "${VISION_TOWER}" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_patch_merge_type flat \
    --model_name_or_path "${LLM_VERSION}" \
    --pretrain_mm_mlp_adapter "${INIT_PROJECTOR}" \
    --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model" \
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
