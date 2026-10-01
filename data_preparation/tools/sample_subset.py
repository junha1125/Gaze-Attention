#!/usr/bin/env python3
"""Draw random subsets of a .jsonl annotation file (reproducible).

Examples:
    # one subset with 40% of the samples -> <name>_40p.jsonl
    python data_preparation/tools/sample_subset.py --input data.jsonl --percent 40

    # two disjoint subsets with 17% of the samples each -> <name>_17p_A.jsonl, <name>_17p_B.jsonl
    python data_preparation/tools/sample_subset.py --input data.jsonl --percent 17 --num-subsets 2
"""

import argparse
import math
import os
import random
import string


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input .jsonl file")
    parser.add_argument("--percent", type=float, required=True, help="Size of every subset in percent of the input")
    parser.add_argument("--num-subsets", type=int, default=1, help="Number of disjoint subsets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None, help="Output folder (default: folder of the input file)")
    args = parser.parse_args()

    if args.percent * args.num_subsets > 100:
        raise ValueError("percent x num-subsets must not exceed 100")

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    subset_size = math.ceil(len(lines) * args.percent / 100)
    print(f"Loaded {len(lines):,} samples; every subset has {subset_size:,} samples")

    random.Random(args.seed).shuffle(lines)

    base, ext = os.path.splitext(os.path.basename(args.input))
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(output_dir, exist_ok=True)
    percent = int(args.percent) if float(args.percent).is_integer() else args.percent
    for i in range(args.num_subsets):
        suffix = f"_{string.ascii_uppercase[i]}" if args.num_subsets > 1 else ""
        path = os.path.join(output_dir, f"{base}_{percent}p{suffix}{ext}")
        subset = lines[i * subset_size : (i + 1) * subset_size]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(line if line.endswith("\n") else line + "\n" for line in subset)
        print(f"Saved {len(subset):,} samples to {path}")


if __name__ == "__main__":
    main()
