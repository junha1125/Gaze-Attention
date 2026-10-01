#!/usr/bin/env python3
r"""Extract the text-only samples (no image, no video) of a .jsonl annotation file.

Example:
    python data_preparation/tools/extract_text_only.py \
        --input datasets/Cambrian-10M/jsons/Cambrian7M_withsystemprompt_filtered_15p.jsonl \
        --output datasets/Cambrian-10M/jsons/Cambrian7M_withsystemprompt_filtered_15p_text_only.jsonl
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input .jsonl file")
    parser.add_argument("--output", required=True, help="Output .jsonl file with the text-only samples")
    args = parser.parse_args()

    total = text_only = 0
    with open(args.input, "r", encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            sample = json.loads(line)
            if not sample.get("image") and not sample.get("video"):
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                text_only += 1

    print(f"Total samples    : {total:,}")
    print(f"Text-only samples: {text_only:,} ({text_only / max(total, 1) * 100:.2f}%)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
