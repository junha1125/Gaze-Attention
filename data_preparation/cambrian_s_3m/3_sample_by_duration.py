#!/usr/bin/env python3
r"""Draw a random subset of the videos that are shorter than a maximum duration (reproducible).

Example (30% of the videos shorter than 60 minutes):
    python data_preparation/cambrian_s_3m/3_sample_by_duration.py \
        --input datasets/Cambrian-S-3M/cambrian_s_3m_clean.jsonl \
        --output datasets/Cambrian-S-3M/cambrian_s_3m_clean_under60m_30p.jsonl \
        --max-minutes 60 --percent 30
"""

import argparse
import collections
import json
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input .jsonl file with a video_duration entry per sample")
    parser.add_argument("--output", required=True, help="Output .jsonl file")
    parser.add_argument("--max-minutes", type=float, default=60, help="Only videos shorter than this are used")
    parser.add_argument("--percent", type=float, default=30, help="Percentage of the remaining videos to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidates = []
    histogram = collections.Counter()
    total = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            duration = json.loads(line).get("video_duration")
            if duration is None:
                continue
            histogram[int(duration // 300)] += 1
            if duration < args.max_minutes * 60:
                candidates.append(line)

    print(f"{'duration':<18} {'samples':>12}")
    for bucket in sorted(histogram):
        print(f"{bucket * 5:>4} - {bucket * 5 + 5:<4} min    {histogram[bucket]:>12,}")

    random.Random(args.seed).shuffle(candidates)
    num_samples = round(len(candidates) * args.percent / 100)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in candidates[:num_samples]:
            f.write(line + "\n")

    print(f"Total samples                      : {total:,}")
    print(f"Shorter than {args.max_minutes:g} minutes          : {len(candidates):,}")
    print(f"Sampled ({args.percent:g}%)                    : {num_samples:,}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
