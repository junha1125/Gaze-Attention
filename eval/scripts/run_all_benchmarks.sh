#!/bin/bash
# Run all image benchmarks sequentially for one checkpoint.
#
# Usage:
#   bash scripts/run_all_benchmarks.sh <checkpoint> [conv_mode] [gaze_topk_ratio]
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_all_benchmarks.sh /path/to/checkpoint
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_all_benchmarks.sh /path/to/checkpoint qwen_1_5 0.25
#
# Finished benchmarks are recorded in progress/; re-running the same command
# runs the benchmarks that are not finished. A failed benchmark does not stop
# the following ones; the failed benchmarks are listed at the end (exit code 1).
set -e

echo "> run_all_benchmarks.sh $@"

if [[ -z "$1" ]]; then
    echo "Usage: bash scripts/run_all_benchmarks.sh <checkpoint> [conv_mode] [gaze_topk_ratio]"
    exit 1
fi

ckpt="$1"
conv_mode="${2:-qwen_1_5}"
gaze_topk_ratio="${3:-}"

benchmarks=(
    # General
    mme
    mmbench_en
    mmbench_cn
    seed
    gqa

    # Science & Math
    scienceqa
    mmmu
    mathvista
    ai2d

    # Chart & OCR
    chartqa
    ocrbench
    textvqa

    # Vision-centric
    mmvp
    realworldqa
    coco
    ade
    omni
)

script_dir=$(dirname "$(realpath "$0")")
progress_dir="${script_dir}/../progress"
mkdir -p "$progress_dir"

# Name of the progress file, derived from the checkpoint path and the settings
ckpt_clean="${ckpt%/}"
ckpt_name="$(basename "$(dirname "$ckpt_clean")")_$(basename "$ckpt_clean")"
if [[ "$(basename "$ckpt_clean")" != checkpoint-* ]]; then
    ckpt_name="$(basename "$ckpt_clean")"
fi
progress_file="$progress_dir/progress_${ckpt_name}_${conv_mode}${gaze_topk_ratio:+_ratio${gaze_topk_ratio}}.txt"

extra_args=()
if [[ -n "$gaze_topk_ratio" ]]; then
    extra_args+=(--gaze_topk_ratio "$gaze_topk_ratio")
fi

if [[ -f "$progress_file" ]]; then
    completed_benchmarks=($(cat "$progress_file"))
    echo "Resuming. Completed benchmarks: ${completed_benchmarks[@]}"
else
    completed_benchmarks=()
fi

start_time=$(date +%s)
failed_benchmarks=()
for benchmark in "${benchmarks[@]}"; do
    if [[ " ${completed_benchmarks[@]} " =~ " $benchmark " ]]; then
        echo "Skipping completed benchmark: $benchmark"
        continue
    fi

    echo "Running benchmark: $benchmark"
    if bash "$script_dir/run_benchmark.sh" --benchmark "$benchmark" --ckpt "$ckpt" --conv_mode "$conv_mode" "${extra_args[@]}"; then
        echo "Finished benchmark: $benchmark"
        echo "$benchmark" >> "$progress_file"
    else
        echo "FAILED benchmark: $benchmark"
        failed_benchmarks+=("$benchmark")
    fi

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    printf "Elapsed time: %02d:%02d \n\n" "$((elapsed / 60))" "$((elapsed % 60))"
done

total_elapsed=$(( $(date +%s) - start_time ))
printf "Total elapsed time: %02d:%02d (MM:SS)\n" "$((total_elapsed / 60))" "$((total_elapsed % 60))"
if [[ ${#failed_benchmarks[@]} -gt 0 ]]; then
    echo "Failed benchmarks (${#failed_benchmarks[@]}): ${failed_benchmarks[*]}"
    echo "Run the same command again to repeat them."
    exit 1
fi
echo "Finished all benchmarks"
