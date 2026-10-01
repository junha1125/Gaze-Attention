#!/bin/bash
# Shared settings of the training scripts. Every value can be overridden from the
# environment, e.g.
#   NUM_GPUS=4 DATA_ROOT=/data bash scripts/train/image_384/1_pretrain_projector.sh
#
# Multi-node training: run the same script on every node with
#   NNODES=4 NODE_RANK=<0..3> MASTER_ADDR=<ip of node 0> MASTER_PORT=29500
# Two trainings on the same machine need different values of MASTER_PORT.
#
# Job launchers such as Kubeflow PyTorchJob configure torchrun through PET_* environment
# variables (PET_NNODES, PET_RDZV_ENDPOINT, ...). When they are set and NNODES is not, torchrun
# uses them and the number of nodes is read from PET_NNODES; unset them to use the variables above.

# ---- Paths ---------------------------------------------------------------
export DATA_ROOT="${DATA_ROOT:-./datasets}"          # see data_preparation/README.md
export OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints}"   # projectors/ and finetune/ are created inside

# ---- Hardware ------------------------------------------------------------
export NUM_GPUS="${NUM_GPUS:-8}"                     # GPUs per node
if [ -z "${NNODES:-}" ] && [ -n "${PET_NNODES:-}" ]; then
    NNODES="${PET_NNODES##*:}"                       # "min:max" -> max
    NNODES_FROM_LAUNCHER=1
fi
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
TOTAL_GPUS=$((NUM_GPUS * NNODES))

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NUM_WORKERS="${NUM_WORKERS:-8}"               # dataloader workers per GPU

# ---- Logging -------------------------------------------------------------
# "wandb" (needs `wandb login`) or "none"
export REPORT_TO="${REPORT_TO:-none}"

# ---- Models --------------------------------------------------------------
# Language model: Qwen/Qwen3-4B-Instruct-2507 or a Qwen2.5 model, e.g. Qwen/Qwen2.5-7B-Instruct
export LLM_VERSION="${LLM_VERSION:-Qwen/Qwen3-4B-Instruct-2507}"
# Short name of the language model in the names of the output folders, e.g. qwen3-4b
LLM_TAG="${LLM_TAG:-$(basename "${LLM_VERSION}" | tr '[:upper:]' '[:lower:]' | sed -e 's/-instruct.*$//')}"

# grad_accum <total batch size> <per-GPU batch size>
# Number of gradient accumulation steps needed to reach the total batch size.
grad_accum() {
    local total=$1 per_device=$2
    local per_step=$((TOTAL_GPUS * per_device))
    if (( total % per_step != 0 )); then
        echo "Total batch size ${total} is not divisible by #GPUs (${TOTAL_GPUS}) x per-GPU batch size (${per_device})" >&2
        exit 1
    fi
    echo $((total / per_step))
}

# Launch cambrian/train/train_mem.py with torchrun on this node.
launch_training() {
    local dist_args=(--nproc_per_node="${NUM_GPUS}")
    if [ -n "${NNODES_FROM_LAUNCHER:-}" ]; then
        echo "torchrun is configured by the PET_* environment variables of the job launcher (${NNODES} node(s) x ${NUM_GPUS} GPUs)." >&2
    elif (( NNODES > 1 )); then
        dist_args+=(--nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
    else
        dist_args+=(--master_port="${MASTER_PORT}")
    fi
    ACCELERATE_CPU_AFFINITY=1 torchrun "${dist_args[@]}" cambrian/train/train_mem.py \
        --deepspeed scripts/train/zero2.json \
        --attn_implementation sdpa \
        --fp16 True \
        --tf32 True \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --gradient_checkpointing True \
        --dataloader_num_workers "${NUM_WORKERS}" \
        --lazy_preprocess True \
        --report_to "${REPORT_TO}" \
        "$@"
}
