#!/usr/bin/env python3
r"""Remove malformed and over-long samples from a .jsonl annotation file.

Example:
    python data_preparation/tools/filter_by_length.py \
        --input datasets/Cambrian-10M/jsons/Cambrian7M_withsystemprompt.jsonl \
        --output datasets/Cambrian-10M/jsons/Cambrian7M_withsystemprompt_filtered.jsonl \
        --max-tokens 800

A sample is removed when
  1. the number of "<image>" placeholders differs from the number of image paths, or
  2. its conversation has `--max-tokens` text tokens or more (sum over all turns,
     placeholders and chat template excluded).
The order of the samples is preserved.
"""

import argparse
import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from tqdm import tqdm
from transformers import AutoTokenizer

DEFAULT_IMAGE_TOKEN = "<image>"


def count_image_paths(sample):
    image = sample.get("image")
    if image is None:
        return 0
    return len(image) if isinstance(image, list) else 1


def read_chunks(path, chunk_size):
    chunk = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input .jsonl file")
    parser.add_argument("--output", required=True, help="Output .jsonl file")
    parser.add_argument("--max-tokens", type=int, default=800, help="Samples with this many text tokens or more are removed")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507", help="Tokenizer name or path")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Samples tokenized per batch")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    total = kept = mismatch = too_long = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for lines in tqdm(read_chunks(args.input, args.chunk_size), desc="Chunks"):
            samples = [json.loads(line) for line in lines]
            total += len(samples)

            # Filter 1: number of <image> placeholders vs. number of image paths
            candidates = []
            for line, sample in zip(lines, samples):
                n_placeholders = sum(turn["value"].count(DEFAULT_IMAGE_TOKEN) for turn in sample["conversations"])
                if n_placeholders != count_image_paths(sample):
                    mismatch += 1
                else:
                    candidates.append((line, sample))

            # Filter 2: number of text tokens, all turns of all samples are tokenized in one batch
            texts = [turn["value"].replace(DEFAULT_IMAGE_TOKEN, "") for _, sample in candidates for turn in sample["conversations"]]
            lengths = iter(len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]) if texts else iter(())
            for line, sample in candidates:
                n_tokens = sum(next(lengths) for _ in sample["conversations"])
                if n_tokens >= args.max_tokens:
                    too_long += 1
                else:
                    out.write(line + "\n")
                    kept += 1

    print(f"Total                          : {total:,}")
    print(f"Removed (placeholder mismatch) : {mismatch:,}")
    print(f"Removed (>= {args.max_tokens} tokens)        : {too_long:,}")
    print(f"Kept                           : {kept:,}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
