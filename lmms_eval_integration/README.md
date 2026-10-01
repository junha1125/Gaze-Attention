# Video benchmarks with lmms-eval

The video benchmarks of the paper (and the additional image benchmarks of its appendix) are evaluated with
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). lmms-eval is not part of this repository:
clone it and apply the small patch of this folder.

| File | Purpose |
|---|---|
| `setup.sh` | Clones lmms-eval at the commit we used, applies the patch and installs it |
| `gaze_attention.patch` | Changes to lmms-eval (see below) |
| `constraints.txt` | Package versions that must stay unchanged when lmms-eval is installed |
| `run_video_benchmarks.sh` | Runs the six video benchmarks of the paper for one checkpoint |
| `run_image_benchmarks.sh` | Runs the additional image benchmarks (appendix of the paper) for one checkpoint |
| `summarize_image_benchmarks.py` | Computes the sub-scores of the appendix tables from the per-sample logs |

## Setup

Install the `cambrian` package of this repository first (see the main README), then:

```bash
bash lmms_eval_integration/setup.sh        # clones into third_party/lmms-eval
```

which is equivalent to

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git third_party/lmms-eval
cd third_party/lmms-eval
git checkout 40a369db5e8c2b02ba437d2e6a60471e8d7e4da3
git apply ../../lmms_eval_integration/gaze_attention.patch
pip install -e . -c ../../lmms_eval_integration/constraints.txt
```

### What the patch changes

| File | Change |
|---|---|
| `lmms_eval/models/simple/llava_onevision.py` | Imports the `cambrian` package of this repository instead of `llava`; new model arguments `gaze_topk_ratio` (fraction of gaze regions attended per query) and `model_max_length` (maximum input length); removes `<think>...</think>` blocks from Qwen3 outputs |
| `lmms_eval/tasks/longvideobench/utils.py` | Adds a missing import (`VideoReader`) |
| `lmms_eval/tasks/tempcompass/utils.py` | Runs without `OPENAI_API_KEY`: answers that cannot be matched by rules are counted as wrong instead of being sent to GPT |
| `lmms_eval/tasks/videomme/utils.py` | Reads the video metadata for subtitles with decord instead of OpenCV |

Checkpoints of this repository are loaded by the existing `llava_onevision` model of lmms-eval, which (with the patch)
imports the `cambrian` package from the current environment.

## Datasets

lmms-eval downloads the benchmark datasets from the Hugging Face Hub into `HF_HOME` when a task runs for the first
time. The six video benchmarks need about 1 TB:

| Task | Benchmark | Dataset | Samples |
|---|---|---|---:|
| `videomme` | Video-MME | lmms-lab/Video-MME | 2,700 |
| `mlvu_dev` | MLVU (dev) | sy1998/MLVU_dev | 2,174 |
| `egoschema_subset` | EgoSchema (subset with public answers) | lmms-lab/egoschema | 500 |
| `longvideobench_val_v` | LongVideoBench (val) | longvideobench/LongVideoBench (gated) | 1,337 |
| `nextqa_mc_test` | NExT-QA (multiple choice, test) | lmms-lab/NExTQA | 8,564 |
| `tempcompass_multi_choice` | TempCompass (multi-choice) | lmms-lab/TempCompass | 1,580 |

```bash
export HF_HOME=/path/to/large/disk/hf_home
huggingface-cli login      # LongVideoBench is gated: accept its terms on the dataset page first
```

EgoSchema is evaluated locally on the 500 questions with public answers (`egoschema_subset`). The task `egoschema`
runs the full test set (5,031 questions) and writes a submission file for the
[EgoSchema evaluation server](https://github.com/egoschema/EgoSchema).

## Usage

```bash
# <checkpoint> [max_frames] [gaze_topk_ratio]
NUM_GPUS=8 bash lmms_eval_integration/run_video_benchmarks.sh /path/to/checkpoint 64

# attend to 20% of the gaze regions instead of the ratio stored in the checkpoint
NUM_GPUS=8 bash lmms_eval_integration/run_video_benchmarks.sh /path/to/checkpoint 64 0.2

# smoke test: 8 samples per task
LIMIT=8 NUM_GPUS=2 bash lmms_eval_integration/run_video_benchmarks.sh /path/to/checkpoint 32

# other tasks
TASKS="egoschema mvbench" NUM_GPUS=8 bash lmms_eval_integration/run_video_benchmarks.sh /path/to/checkpoint 64
```

Results are written to `logs/lmms_eval/<run name>/`:

- `<model>/<timestamp>_results.json`, `<model>/<timestamp>_samples_<task>.jsonl`: scores and per-sample outputs,
- `run_logs/<task>.log`, `run_logs/timing_summary.txt`: logs, exit code and run time of every task.

The script runs the tasks one after another and continues with the next task when one fails; failed tasks are
listed at the end and in `timing_summary.txt`.
With six H100 GPUs and 64 frames, the six video benchmarks take about four hours.
Use different `PORT` values (default 12400) and disjoint `CUDA_VISIBLE_DEVICES` to run several evaluations at the same time.

### Running lmms-eval directly

```bash
accelerate launch --num_processes=8 --main_process_port 12400 -m lmms_eval \
    --model llava_onevision \
    --model_args "pretrained=/path/to/checkpoint,conv_template=qwen_3,device_map=auto,model_name=cambrian_qwen3,max_frames_num=64" \
    --tasks videomme,mlvu_dev \
    --batch_size 1 \
    --log_samples \
    --gen_kwargs "max_new_tokens=16" \
    --output_path ./logs/lmms_eval/my_run
```

| Model argument | Meaning |
|---|---|
| `pretrained` | Checkpoint folder or Hugging Face model id |
| `conv_template` | `qwen_3` (the same ChatML template is used for Qwen2.5) |
| `model_name` | `cambrian_qwen3` for Qwen3 models, `cambrian_qwen` for Qwen2.5 models |
| `max_frames_num` | Number of uniformly sampled frames per video |
| `gaze_topk_ratio` | Optional; fraction of gaze regions attended per query |
| `model_max_length` | Optional; maximum number of input tokens including the visual tokens |

Image tasks of lmms-eval run with the same command without `max_frames_num`.

### Additional image benchmarks

The appendix of the paper reports counting, fine-grained perception and spatial reasoning scores. They are sub-scores
of eight lmms-eval tasks (`countbench`, `blink_counting`, `blink_spatial_relation`, `blink_object_localization`,
`mme`, `mmstar`, `mmerealworld_lite`, `mmsi_bench`):

```bash
# <checkpoint> [gaze_topk_ratio]
NUM_GPUS=8 bash lmms_eval_integration/run_image_benchmarks.sh /path/to/checkpoint

# tables of the sub-scores, one row per run
python lmms_eval_integration/summarize_image_benchmarks.py logs/lmms_eval/<run name> [more runs ...]
```

MMSI-Bench has up to 10 images per question. The 512px models (1,024 visual tokens per image) need a larger input
limit than the 8,192 tokens of their config for this task:

```bash
MODEL_MAX_LENGTH=12288 TASKS=mmsi_bench NUM_GPUS=8 bash lmms_eval_integration/run_image_benchmarks.sh /path/to/checkpoint
```

### Visual tokens in the KV cache

With 4x token compression every frame has 144 visual tokens and 4 context tokens, and one gaze region has 36 tokens
(4 regions per frame). For `F` frames and TopK ratio `r`:

| | Visual KV entries per query |
|---|---|
| Dense attention | `144 x F` |
| Gaze Attention | `36 x ceil(r x 4 x F) + 4 x F` |

For example, 64 frames: 9,216 (dense), 1,192 (`r = 0.1`), 2,128 (`r = 0.2`).

**Maximum number of frames.** Inputs longer than `tokenizer_model_max_length` of the checkpoint config are truncated
(a warning is printed). The released video models were trained with 12,288 tokens, which is enough for about 80 frames.
To evaluate with more frames, raise the limit with the model argument `model_max_length`
(`MODEL_MAX_LENGTH=24576` for `run_video_benchmarks.sh`).
