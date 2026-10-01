#!/bin/bash
###############################################################################
# Additional image benchmarks (appendix of the paper) with lmms-eval.
#
# Usage:
#   bash lmms_eval_integration/run_image_benchmarks.sh <checkpoint> [gaze_topk_ratio]
#
# Examples:
#   NUM_GPUS=8 bash lmms_eval_integration/run_image_benchmarks.sh /path/to/checkpoint
#   LIMIT=8 NUM_GPUS=2 bash lmms_eval_integration/run_image_benchmarks.sh /path/to/checkpoint     # smoke test
#
# Environment variables: NUM_GPUS (2), PORT (12400), TASKS, OUTPUT_ROOT (./logs/lmms_eval), LIMIT, HF_HOME,
#   MODEL_MAX_LENGTH  maximum number of input tokens incl. visual tokens (default: value of the checkpoint).
#                     MMSI-Bench has up to 10 images per question: the 512px models (1,024 tokens per
#                     image) need MODEL_MAX_LENGTH=12288 for this task.
###############################################################################
set -o pipefail

# The evaluation runs on one machine: ignore the torchrun settings of job launchers
# (e.g. Kubeflow PyTorchJob exports PET_* variables), which `accelerate launch` would pick up.
unset PET_NNODES PET_NPROC_PER_NODE PET_RDZV_BACKEND PET_RDZV_ENDPOINT PET_RDZV_ID

if [ -z "$1" ]; then
    echo "Usage: bash lmms_eval_integration/run_image_benchmarks.sh <checkpoint> [gaze_topk_ratio]"
    exit 1
fi

MODEL_PATH="$1"
GAZE_TOPK_RATIO="${2:-}"
NUM_GPUS="${NUM_GPUS:-2}"
PORT="${PORT:-12400}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./logs/lmms_eval}"
LIMIT="${LIMIT:-}"
read -r -a TASK_LIST <<< "${TASKS:-countbench blink_counting blink_spatial_relation blink_object_localization mme mmstar mmerealworld_lite mmsi_bench}"

PARENT_DIR=$(basename "$(dirname "${MODEL_PATH%/}")")
CKPT_DIR=$(basename "${MODEL_PATH%/}")
RUN_NAME="${PARENT_DIR}-${CKPT_DIR}-image${GAZE_TOPK_RATIO:+-ratio${GAZE_TOPK_RATIO}}"

OUTPUT_PATH="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${OUTPUT_PATH}/run_logs"
TIMING_FILE="${LOG_DIR}/timing_summary.txt"
mkdir -p "${LOG_DIR}"

MODEL_ARGS="pretrained=${MODEL_PATH},conv_template=qwen_3,device_map=auto,model_name=cambrian_qwen3"
if [ -n "${GAZE_TOPK_RATIO}" ]; then
    MODEL_ARGS="${MODEL_ARGS},gaze_topk_ratio=${GAZE_TOPK_RATIO}"
fi
if [ -n "${MODEL_MAX_LENGTH:-}" ]; then
    MODEL_ARGS="${MODEL_ARGS},model_max_length=${MODEL_MAX_LENGTH}"
fi

EXTRA_ARGS=()
if [ -n "${LIMIT}" ]; then
    EXTRA_ARGS+=(--limit "${LIMIT}")
fi

{
    echo "============================================="
    echo "Evaluation started: $(date)"
    echo "Model: ${MODEL_PATH}"
    echo "Model args: ${MODEL_ARGS}"
    echo "Output: ${OUTPUT_PATH}"
    echo "GPUs: ${NUM_GPUS} | Port: ${PORT}"
    echo "Tasks (${#TASK_LIST[@]}): ${TASK_LIST[*]}"
    echo "============================================="
} | tee "${TIMING_FILE}"

FAILED_TASKS=()
for task in "${TASK_LIST[@]}"; do
    log_file="${LOG_DIR}/${task}.log"
    start_seconds=$SECONDS
    marker="${LOG_DIR}/.start_${task}"
    touch "${marker}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: ${task}" | tee -a "${log_file}"

    accelerate launch \
        --num_processes="${NUM_GPUS}" \
        --main_process_port "${PORT}" \
        -m lmms_eval \
        --model=llava_onevision \
        --model_args="${MODEL_ARGS}" \
        --tasks="${task}" \
        --batch_size=1 \
        --log_samples \
        --gen_kwargs="max_new_tokens=16" \
        --output_path "${OUTPUT_PATH}" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "${log_file}"
    exit_code=${PIPESTATUS[0]}
    # lmms-eval also exits with 0 when the evaluation raised an error: check that results were written
    if [ -z "$(find "${OUTPUT_PATH}" -name "*_results.json" -newer "${marker}" 2>/dev/null)" ] && [ "${exit_code}" -eq 0 ]; then
        exit_code=1
    fi
    rm -f "${marker}"

    elapsed=$(( SECONDS - start_seconds ))
    time_str="$(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m $(( elapsed % 60 ))s"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] END: ${task} | Exit: ${exit_code} | Elapsed: ${time_str}" | tee -a "${log_file}"
    echo "${task} | GPUs: ${NUM_GPUS} | Exit: ${exit_code} | Elapsed: ${time_str}" >> "${TIMING_FILE}"
    if [ "${exit_code}" -ne 0 ]; then
        FAILED_TASKS+=("${task}")
    fi
done

{
    echo "============================================="
    echo "All tasks finished: $(date)"
    if [ "${#FAILED_TASKS[@]}" -gt 0 ]; then
        echo "Failed tasks (${#FAILED_TASKS[@]}): ${FAILED_TASKS[*]}"
    else
        echo "Failed tasks: none"
    fi
    echo "============================================="
} | tee -a "${TIMING_FILE}"

echo "Logs: ${LOG_DIR}/"
echo "Results: ${OUTPUT_PATH}/"

if [ "${#FAILED_TASKS[@]}" -gt 0 ]; then
    exit 1
fi
