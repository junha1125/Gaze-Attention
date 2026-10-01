#!/usr/bin/env python3
r"""Remove samples with missing files and / or samples of unwanted sources.

Example:
    python data_preparation/tools/filter_annotations.py \
        --input datasets/Cambrian-Alignment/jsons/alignment_2.5m.jsonl \
        --missing datasets/Cambrian-Alignment/jsons/missing.jsonl \
        --exclude sbu558k \
        --output datasets/Cambrian-Alignment/jsons/alignment_filtered.jsonl

A sample is removed when
  * one of its media paths appears in the `--missing` file (output of check_missing_media.py), or
  * one of its media paths contains one of the `--exclude` substrings.
Text-only samples are always kept. The order of the samples is preserved.
"""

import argparse
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import iter_records, media_paths, write_records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Annotation file (.json or .jsonl)")
    parser.add_argument("--output", required=True, help="Filtered annotation file (.json or .jsonl)")
    parser.add_argument("--missing", default=None, help="Records with missing files (output of check_missing_media.py)")
    parser.add_argument("--exclude", default="", help="Comma-separated substrings; samples whose media path contains one of them are removed")
    args = parser.parse_args()

    missing_paths = set()
    if args.missing:
        for record in iter_records(args.missing):
            missing_paths.update(media_paths(record))
        print(f"Loaded {len(missing_paths):,} missing media paths")
    exclude = [key.strip() for key in args.exclude.split(",") if key.strip()]
    print(f"Excluded substrings: {exclude}")

    stats = {"total": 0, "missing": 0, "excluded": 0, "kept": 0}

    def kept_records():
        for record in tqdm(iter_records(args.input), desc="Filtering"):
            stats["total"] += 1
            paths = media_paths(record)
            if any(p in missing_paths for p in paths):
                stats["missing"] += 1
                continue
            if any(key in p for key in exclude for p in paths):
                stats["excluded"] += 1
                continue
            stats["kept"] += 1
            yield record

    write_records(args.output, kept_records())

    print(f"Total                  : {stats['total']:,}")
    print(f"Removed (missing file) : {stats['missing']:,}")
    print(f"Removed (excluded)     : {stats['excluded']:,}")
    print(f"Kept                   : {stats['kept']:,}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
