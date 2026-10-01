#!/usr/bin/env python3
r"""Prepend a folder name to the image / video paths of a .jsonl annotation file.

Used to train on several datasets that live under one common media root, e.g.
LLaVA-OneVision (paths relative to <root>) and Cambrian-10M linked as <root>/Cambrian-10M:

    python data_preparation/tools/add_path_prefix.py \
        --input datasets/Cambrian-10M/jsons/Cambrian7M_filtered_40p.jsonl \
        --output datasets/Cambrian-10M/jsons/Cambrian7M_filtered_40p_onevision_path.jsonl \
        --prefix Cambrian-10M
"""

import argparse
import json
import os


def add_prefix(value, prefix):
    if isinstance(value, str):
        return os.path.join(prefix, value)
    if isinstance(value, list):
        return [add_prefix(v, prefix) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input .jsonl file")
    parser.add_argument("--output", required=True, help="Output .jsonl file")
    parser.add_argument("--prefix", required=True, help="Folder name to prepend")
    args = parser.parse_args()

    n = 0
    with open(args.input, "r", encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            for key in ("image", "video"):
                if key in sample:
                    sample[key] = add_prefix(sample[key], args.prefix)
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n += 1
    print(f"Saved {n:,} samples to {args.output}")


if __name__ == "__main__":
    main()
