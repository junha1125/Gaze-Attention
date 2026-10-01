#!/bin/bash
# Stage 2: full finetuning with Gaze Attention on Cambrian-7M (512px)
#
# Full-parameter finetuning (vision encoder, projector, language model) with Gaze Attention.
# The TopK ratio follows the progressive schedule: it starts at 0.9 and decays to 0.1 during
# the first 60% of training, then stays at 0.1.
# Output: ${OUTPUT_ROOT}/finetune/so400m512-${LLM_TAG}-stage2-gaze-cambrian7m
#
# Usage (from the repository root):
#   bash scripts/train/image_512/3_finetune_gaze_cambrian7m.sh [additional training arguments]
# Additional arguments override the ones below, e.g. --save_steps 500 or --max_steps 1000.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

VISION_TOWER="${VISION_TOWER:-google/siglip2-so400m-patch16-512}"
INIT_PROJECTOR="${INIT_PROJECTOR:-${OUTPUT_ROOT}/projectors/so400m512-${LLM_TAG}-stage1-2-ctx4/mm_projector.bin}"
RUN_NAME="so400m512-${LLM_TAG}-stage2-gaze-cambrian7m"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
TOTAL_BATCH="${TOTAL_BATCH:-512}"
GRAD_ACCUM="$(grad_accum "${TOTAL_BATCH}" "${PER_DEVICE_BATCH}")"

launch_training \
    --data_path "${DATA_ROOT}/Cambrian-10M/jsons/Cambrian7M_withsystemprompt_filtered.jsonl" \
    --image_folder "${DATA_ROOT}/Cambrian-10M" \
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
