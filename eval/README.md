# Image benchmarks

Evaluation suite for the image benchmarks of the paper. It is adapted from the evaluation code of
[Cambrian-1](https://github.com/cambrian-mllm/cambrian/tree/main/eval), with two changes:

- the models are loaded with the `cambrian` package of this repository,
- all benchmark datasets can be downloaded in advance into a local folder and are loaded from there.

Video benchmarks are evaluated with lmms-eval, see [`../lmms_eval_integration`](../lmms_eval_integration).

## Layout

```
eval/
├── scripts/
│   ├── run_all_benchmarks.sh    # all benchmarks, one after another
│   ├── run_benchmark.sh         # one benchmark: inference on all visible GPUs, then scoring
│   ├── tabulate.py              # collect the scores of all benchmarks into one table
│   └── clean_results.py         # delete all answers and scores
├── eval/
│   ├── dataset_utils.py         # dataset download and loading
│   ├── download_all_datasets.py # download all benchmark datasets
│   └── <benchmark>/
│       ├── <benchmark>_eval.py  # inference: writes answers/answers_<model>.jsonl
│       └── <benchmark>_test.py  # scoring: appends a row to experiments.csv
├── datasets_eval/               # local copy of the benchmark datasets (created by the download script)
└── progress/                    # finished benchmarks per checkpoint (used to resume)
```

## Benchmarks

| Benchmark | Dataset (Hugging Face) | Split | Reported score |
|---|---|---|---|
| `mme` | lmms-lab/MME | test | perception score (MME<sup>P</sup>) |
| `mmbench_en` | lmms-lab/MMBench_EN | dev | circular accuracy |
| `mmbench_cn` | lmms-lab/MMBench_CN | dev | circular accuracy |
| `seed` | lmms-lab/SEED-Bench | test | accuracy, every 5th image question |
| `gqa` | lmms-lab/GQA | testdev balanced | accuracy |
| `scienceqa` | derek-thomas/ScienceQA | test | accuracy on the questions with an image |
| `mmmu` | lmms-lab/MMMU | validation | accuracy |
| `mathvista` | AI4Math/MathVista | testmini | accuracy, strict matching of the raw answer |
| `ai2d` | lmms-lab/ai2d | test | accuracy |
| `chartqa` | lmms-lab/ChartQA | test | relaxed accuracy |
| `ocrbench` | echo840/OCRBench | test | accuracy |
| `textvqa` | lmms-lab/textvqa | validation | VQA accuracy |
| `mmvp` | MMVP/MMVP | - | pair accuracy |
| `realworldqa` | lmms-lab/RealWorldQA | test | accuracy |
| `coco` | SaiCharithaAkula21/benchmark_coco_filtered | train | accuracy (CV-Bench 2D, COCO) |
| `ade` | SaiCharithaAkula21/benchmark_ade_manual | train | accuracy (CV-Bench 2D, ADE) |
| `omni` | nyu-visionx/CV-Bench (3D) | test | accuracy (CV-Bench 3D, Omni3D) |

The paper reports 13 of them: MME, MMBench (EN and CN), MMVP, RealWorldQA, ADE (`ade`), OmniBench (`omni`), GQA,
ScienceQA, MMMU, MathVista, ChartQA, OCRBench and TextVQA.

Notes on the scores:

- No benchmark needs an API key; all answers are scored by rules.
- MathVista answers are compared with the ground truth without an answer-extraction step, which gives lower numbers
  than the official protocol (which extracts the answer with GPT). Use the numbers only for comparisons within this suite.
- SEED-Bench is evaluated on every 5th image question (2,851 questions) to save time
  (`--subsample_stride` of `seed_eval.py`).

## Setup

```bash
pip install -e ".[eval]"        # from the repository root
```

### Download the datasets (optional)

```bash
python eval/eval/download_all_datasets.py
```

The datasets are stored in `eval/datasets_eval/<org>--<name>/` (about 200 GB). Set `EVAL_DATASETS_DIR` to use another
folder. Datasets that are not there are downloaded when a benchmark needs them, so this step is only required
when the evaluation machine has no internet access. Each process waits for a file lock, so several GPU workers can
start at the same time.

Log in to the Hugging Face Hub first (`huggingface-cli login`): anonymous downloads are limited to few requests.
When the Hub answers with a rate limit, the download waits and continues.

## Usage

Run from the `eval/` folder. One inference process is started per GPU in `CUDA_VISIBLE_DEVICES`.

```bash
cd eval

# all benchmarks
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_all_benchmarks.sh /path/to/checkpoint

# one benchmark
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_benchmark.sh --benchmark mme --ckpt /path/to/checkpoint
```

`/path/to/checkpoint` is a local folder or a Hugging Face model id.
With eight H100 GPUs, all benchmarks of one checkpoint take about 35 minutes when the datasets are already downloaded.

**Resuming.** Finished benchmarks are recorded in `progress/`. Running the same command again skips them.
A benchmark that fails does not stop the following ones; the failed benchmarks are listed at the end.

**Number of attended gaze regions.** A Gaze Attention checkpoint attends to the fraction of gaze regions stored in its
config (`gaze_topk_ratio`, 0.1 for the released models). The ratio can be changed for evaluation:

```bash
# 576 visual tokens = 16 regions of 36 tokens: ratio 0.1 -> 2 regions (72 tokens), 0.25 -> 4 regions (144 tokens)
bash scripts/run_all_benchmarks.sh /path/to/checkpoint qwen_1_5 0.25
bash scripts/run_benchmark.sh --benchmark mme --ckpt /path/to/checkpoint --gaze_topk_ratio 0.25
```

## Results

Every benchmark folder gets

- `answers/answers_<model>.jsonl`: the model answers,
- `incorrect/`: the wrongly answered questions,
- `experiments.csv`: one row of scores per evaluated model.

Collect the scores of all benchmarks:

```bash
python scripts/tabulate.py --eval_dir eval --out_pivot pivot.xlsx --out_all_results all_results.csv
```

`all_results.csv` has one row per benchmark and model, `pivot.xlsx` is a table of models x benchmarks.
