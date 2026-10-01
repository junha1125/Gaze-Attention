#!/bin/bash
# Run one benchmark: inference (one process per visible GPU) followed by scoring.
#
# Usage:
#   bash scripts/run_benchmark.sh --benchmark mme --ckpt <checkpoint> [--conv_mode qwen_1_5] [--gaze_topk_ratio 0.25]
#
# GPUs are taken from CUDA_VISIBLE_DEVICES (default: GPU 0), e.g.
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_benchmark.sh --benchmark mme --ckpt <checkpoint>
set -e

echo "> run_benchmark.sh $@"

################# Parse Arguments #################

conv_mode="qwen_1_5"
gaze_topk_ratio=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
        benchmark="$2"
        shift 2
        ;;
    --ckpt)
        ckpt="$2"
        shift 2
        ;;
    --conv_mode)
        conv_mode="$2"
        shift 2
        ;;
    --gaze_topk_ratio)
        gaze_topk_ratio="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
  esac
done

if [[ -z "$benchmark" || -z "$ckpt" ]]; then
  echo "Error: --benchmark and --ckpt arguments are required."
  exit 1
fi

echo "Benchmark: $benchmark"
echo "Checkpoint path: $ckpt"
echo "Conversation mode: $conv_mode"
echo "Gaze TopK ratio: ${gaze_topk_ratio:-<from checkpoint>}"

################# Process Arguments #################

script_dir=$(dirname "$(realpath "$0")")
benchmark_dir="${script_dir}/../eval/${benchmark}"
eval_file="${benchmark_dir}/${benchmark}_eval.py"
test_file="${benchmark_dir}/${benchmark}_test.py"

if [ ! -f "$eval_file" ]; then
    echo "Error: Eval file $eval_file does not exist."
    exit 1
fi
if [ ! -f "$test_file" ]; then
    echo "Error: Test file $test_file does not exist."
    exit 1
fi

# A local checkpoint path must be absolute because we change directory below
if [ -d "$ckpt" ]; then
    ckpt=$(realpath "$ckpt")
fi

cd "$benchmark_dir"

# Name of this run: last 2 path components (3 for .../<run>/checkpoint-<step>)
ckpt_clean="${ckpt%/}"
name1=$(basename "$ckpt_clean")
name2=$(basename "$(dirname "$ckpt_clean")")
name3=$(basename "$(dirname "$(dirname "$ckpt_clean")")")
if [[ "$name1" == checkpoint-* ]]; then
    model_basename="${name3}-${name2}-${name1}"
else
    model_basename="${name2}-${name1}"
fi
extra_args=()
if [[ -n "$gaze_topk_ratio" ]]; then
    model_basename="${model_basename}-ratio${gaze_topk_ratio}"
    extra_args+=(--gaze_topk_ratio "$gaze_topk_ratio")
fi

answers_file="./answers/answers_${model_basename}.jsonl"
echo "Answers file: $answers_file"

################# Handle Chunking #################
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
echo "Number of GPUs / chunks: $CHUNKS"

################# Run Evaluation #################

echo "$(date "+%Y-%m-%d %H:%M:%S")"

pids=()
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" python "$eval_file" \
        --model_path "$ckpt" \
        --num_chunks "$CHUNKS" \
        --chunk_idx "$IDX" \
        --answers_file "$answers_file" \
        --conv_mode "$conv_mode" \
        "${extra_args[@]}" &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
    echo "Error: at least one inference process failed for $benchmark. Not scoring partial results."
    exit 1
fi

################# Combine Results #################

echo "$(date "+%Y-%m-%d %H:%M:%S")"

# If the answers file already exists, move it to a backup file with the current timestamp
if [ -f "$answers_file" ]; then
    backup="${answers_file}.bak.$(date +%s)"
    mv "$answers_file" "$backup"
    echo "Moved existing answers file to $backup"
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    idx_file=./answers/answers_${model_basename}_${IDX}.jsonl
    cat "$idx_file" >> "$answers_file"
    rm "$idx_file"
done

################# Run Testing #####################

python "$test_file" --answers_file "$answers_file"

echo "Done evaluation and testing for $benchmark on model at $ckpt with conversation mode $conv_mode"
