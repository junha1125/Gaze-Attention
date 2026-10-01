"""Download the benchmark datasets used by scripts/run_all_benchmarks.sh.

Usage:
    python eval/eval/download_all_datasets.py                       # all benchmarks
    python eval/eval/download_all_datasets.py --only mme mmvp       # selected benchmarks

Each dataset is downloaded with snapshot_download into datasets_eval/ (or EVAL_DATASETS_DIR),
then loaded once so that the Arrow cache is created. Datasets that are already there are skipped.
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_utils import ensure_dataset, load_hf_dataset

# (label, repo_id, name_or_None, split_or_None, use_ensure_only)
DATASETS = [
    # --- mme ---
    ("mme",          "lmms-lab/MME",                                   None,                            "test",       False),
    # --- mmbench_en ---
    ("mmbench_en",   "lmms-lab/MMBench_EN",                            None,                            "dev",        False),
    # --- mmbench_cn ---
    ("mmbench_cn",   "lmms-lab/MMBench_CN",                            "default",                       "dev",        False),
    # --- seed ---
    ("seed",         "lmms-lab/SEED-Bench",                            None,                            "test",       False),
    # --- gqa (2 configs) ---
    ("gqa-images",   "lmms-lab/GQA",                                   "testdev_balanced_images",       "testdev",    False),
    ("gqa-instrs",   "lmms-lab/GQA",                                   "testdev_balanced_instructions", "testdev",    False),
    # --- scienceqa ---
    ("scienceqa",    "derek-thomas/ScienceQA",                         None,                            "test",       False),
    # --- mmmu (2 splits) ---
    ("mmmu-val",     "lmms-lab/MMMU",                                  None,                            "validation", False),
    ("mmmu-dev",     "lmms-lab/MMMU",                                  None,                            "dev",        False),
    # --- mathvista ---
    ("mathvista",    "AI4Math/MathVista",                               None,                            "testmini",   False),
    # --- ai2d ---
    ("ai2d",         "lmms-lab/ai2d",                                  None,                            "test",       False),
    # --- chartqa ---
    ("chartqa",      "lmms-lab/ChartQA",                               None,                            "test",       False),
    # --- ocrbench ---
    ("ocrbench",     "echo840/OCRBench",                               None,                            "test",       False),
    # --- textvqa ---
    ("textvqa",      "lmms-lab/textvqa",                               None,                            "validation", False),
    # --- mmvp (ensure_dataset only, no load_dataset) ---
    ("mmvp",         "MMVP/MMVP",                                      None,                            None,         True),
    # --- realworldqa ---
    ("realworldqa",  "lmms-lab/RealWorldQA",                           None,                            "test",       False),
    # --- coco ---
    ("coco",         "SaiCharithaAkula21/benchmark_coco_filtered",     None,                            "train",      False),
    # --- ade ---
    ("ade",          "SaiCharithaAkula21/benchmark_ade_manual",        None,                            "train",      False),
    # --- omni (CV-Bench 3d) ---
    ("omni",         "nyu-visionx/CV-Bench",                           "3D",                            "test",       False),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="Benchmarks to download, e.g. --only mme gqa (default: all)")
    args = parser.parse_args()

    datasets = DATASETS
    if args.only:
        datasets = [d for d in DATASETS if d[0].split("-")[0] in args.only]
        if not datasets:
            raise SystemExit(f"No benchmark matches {args.only}. Available: {sorted({d[0].split('-')[0] for d in DATASETS})}")

    total = len(datasets)
    success = 0
    failed = []
    start = time.time()

    for i, (label, repo_id, name, split, ensure_only) in enumerate(datasets, 1):
        header = f"[{i}/{total}] {label} ({repo_id})"
        print(f"\n{'='*60}")
        print(header)
        print(f"{'='*60}")

        try:
            if ensure_only:
                ensure_dataset(repo_id)
            else:
                kwargs = {}
                if name is not None:
                    kwargs["name"] = name
                ds = load_hf_dataset(repo_id, split=split, **kwargs)
                print(f"  -> {len(ds)} examples loaded")
            success += 1
        except Exception as e:
            print(f"  !! FAILED: {e}")
            failed.append(label)

    elapsed = time.time() - start
    minutes = int(elapsed) // 60
    seconds = int(elapsed) % 60

    print(f"\n{'='*60}")
    print(f"Done: {success}/{total} succeeded in {minutes:02d}:{seconds:02d}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"{'='*60}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
