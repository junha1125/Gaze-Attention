#!/bin/bash
###############################################################################
# Video benchmarks of the paper with lmms-eval (tasks run one after another).
#
# Usage:
#   bash lmms_eval_integration/run_video_benchmarks.sh <checkpoint> [max_frames] [gaze_topk_ratio]
#
# Examples:
#   bash lmms_eval_integration/run_video_benchmarks.sh /path/to/checkpoint              # 64 frames
#   bash lmms_eval_integration/run_video_benchmarks.sh /path/to/checkpoint 32           # 32 frames
#   bash lmms_eval_integration/run_video_benchmarks.sh /path/to/checkpoint 64 0.2       # attend to 20% of the gaze regions
#   NUM_GPUS=8 bash lmms_eval_integration/run_video_benchmarks.sh /path/to/checkpoint
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 PORT=12401 bash lmms_eval_integration/run_video_benchmarks.sh /path/to/checkpoint
#
# Environment variables:
#   NUM_GPUS     number of processes / GPUs (default: 2)
#   PORT         port used by accelerate (default: 12400); use different ports for concurrent runs
#   TASKS        space-separated task list (default: the six video benchmarks of the paper;
#                EgoSchema is evaluated on the 500 questions with public answers, task egoschema_subset)
#   OUTPUT_ROOT  where results are written (default: ./logs/lmms_eval)
#   LIMIT        evaluate only the first N samples of every task (smoke test)
#   MODEL_MAX_LENGTH  maximum number of input tokens incl. visual tokens (default: value of the checkpoint)
#   HF_HOME      Hugging Face cache; the benchmark videos are downloaded there (several 100 GB)
#
# Frames are sampled uniformly. With 4x token compression every frame has 144 visual
# tokens (+4 context tokens), so 64 frames give 9.5K visual tokens in the KV cache.
# The released video models accept 12,288 input tokens (about 80 frames); set
# MODEL_MAX_LENGTH to evaluate with more frames, e.g. MODEL_MAX_LENGTH=24576 for 128 frames.
###############################################################################
set -o pipefail

# The evaluation runs on one machine: ignore the torchrun settings of job launchers
# (e.g. Kubeflow PyTorchJob exports PET_* variables), which `accelerate launch` would pick up.
unset PET_NNODES PET_NPROC_PER_NODE PET_RDZV_BACKEND PET_RDZV_ENDPOINT PET_RDZV_ID

if [ -z "$1" ]; then
    echo "Usage: bash lmms_eval_integration/run_video_benchmarks.sh <checkpoint> [max_frames] [gaze_topk_ratio]"
    exit 1
fi

MODEL_PATH="$1"
MAX_FRAMES="${2:-64}"
GAZE_TOPK_RATIO="${3:-}"
NUM_GPUS="${NUM_GPUS:-2}"
PORT="${PORT:-12400}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./logs/lmms_eval}"
LIMIT="${LIMIT:-}"
read -r -a TASK_LIST <<< "${TASKS:-videomme mlvu_dev egoschema_subset longvideobench_val_v nextqa_mc_test tempcompass_multi_choice}"

# Run name: last two path components + settings
PARENT_DIR=$(basename "$(dirname "${MODEL_PATH%/}")")
CKPT_DIR=$(basename "${MODEL_PATH%/}")
RUN_NAME="${PARENT_DIR}-${CKPT_DIR}-f${MAX_FRAMES}${GAZE_TOPK_RATIO:+-ratio${GAZE_TOPK_RATIO}}"

OUTPUT_PATH="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${OUTPUT_PATH}/run_logs"
TIMING_FILE="${LOG_DIR}/timing_summary.txt"
mkdir -p "${LOG_DIR}"

MODEL_ARGS="pretrained=${MODEL_PATH},conv_template=qwen_3,device_map=auto,model_name=cambrian_qwen3,max_frames_num=${MAX_FRAMES}"
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

run_task() {
    local task=$1
    local log_file="${LOG_DIR}/${task}.log"
    local start_seconds=$SECONDS
    local marker="${LOG_DIR}/.start_${task}"
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

    local exit_code=${PIPESTATUS[0]}
    # lmms-eval also exits with 0 when the evaluation raised an error: check that results were written
    if [ -z "$(find "${OUTPUT_PATH}" -name "*_results.json" -newer "${marker}" 2>/dev/null)" ] && [ "${exit_code}" -eq 0 ]; then
        exit_code=1
    fi
    rm -f "${marker}"
    local elapsed=$(( SECONDS - start_seconds ))
    local time_str="$(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m $(( elapsed % 60 ))s"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] END: ${task} | Exit: ${exit_code} | Elapsed: ${time_str}" | tee -a "${log_file}"
    echo "${task} | GPUs: ${NUM_GPUS} | Exit: ${exit_code} | Elapsed: ${time_str}" >> "${TIMING_FILE}"
    return ${exit_code}
}

FAILED_TASKS=()
for task in "${TASK_LIST[@]}"; do
    if ! run_task "${task}"; then
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
