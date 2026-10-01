#!/usr/bin/env python3
r"""Keep the plain video question-answering samples that the dataloader can train on.

Example:
    python data_preparation/cambrian_s_3m/2_filter_samples.py \
        --input datasets/Cambrian-S-3M/cambrian_s_3m_probed.jsonl \
        --output datasets/Cambrian-S-3M/cambrian_s_3m_clean.jsonl \
        --min-duration 5

A sample is removed when
  * its video is shorter than `--min-duration` seconds,
  * it has an "audio" entry or a "<speech>" placeholder in a human turn,
  * it has an "image" entry but no "video" entry,
  * it has entries other than the core keys below. Such samples refer to a part of a video
    (start / end times, frame ranges, ...), which the dataloader does not support.
"""

import argparse
import collections
import json

CORE_KEYS = {"id", "conversations", "video", "data_source", "task", "instruction", "type", "video_duration"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input .jsonl file (output of 1_probe_videos.py)")
    parser.add_argument("--output", required=True, help="Output .jsonl file")
    parser.add_argument("--min-duration", type=float, default=5.0, help="Minimum video duration in seconds")
    args = parser.parse_args()

    stats = collections.Counter()
    extra_keys = collections.Counter()
    with open(args.input, "r", encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            sample = json.loads(line)

            if "audio" in sample:
                stats["removed: audio"] += 1
                continue
            if "video" not in sample:
                stats["removed: no video"] += 1
                continue
            if any(t.get("from") == "human" and "<speech>" in t.get("value", "") for t in sample.get("conversations", [])):
                stats["removed: speech"] += 1
                continue
            if sample.get("video_duration", 0.0) < args.min_duration:
                stats["removed: too short"] += 1
                continue
            extra = set(sample.keys()) - CORE_KEYS
            if extra:
                stats["removed: extra keys"] += 1
                extra_keys.update(extra)
                continue

            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    for key, value in stats.items():
        print(f"{key:<22}: {value:,}")
    if extra_keys:
        print("Extra keys of the removed samples:", dict(extra_keys.most_common()))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
