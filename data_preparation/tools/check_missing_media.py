#!/usr/bin/env python3
r"""List the samples whose image / video file does not exist.

Example:
    python data_preparation/tools/check_missing_media.py \
        --annotation datasets/Cambrian-Alignment/jsons/alignment_2.5m.jsonl \
        --media-root datasets/Cambrian-Alignment \
        --output datasets/Cambrian-Alignment/jsons/missing.jsonl

The output is a .jsonl file with the full records of the samples that reference at
least one missing file. It is the `--missing` input of filter_annotations.py.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import iter_records, media_paths, write_records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotation", required=True, help="Annotation file (.json or .jsonl)")
    parser.add_argument("--media-root", required=True, help="Folder that the image / video paths are relative to")
    parser.add_argument("--output", required=True, help="Output .jsonl with the records that have missing files")
    parser.add_argument("--workers", type=int, default=32, help="Number of threads")
    args = parser.parse_args()

    records = list(iter_records(args.annotation))
    print(f"Loaded {len(records):,} records from {args.annotation}")

    def is_missing(record):
        return any(not os.path.exists(os.path.join(args.media_root, p)) for p in media_paths(record))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        flags = list(tqdm(executor.map(is_missing, records), total=len(records), desc="Checking files"))

    missing = [r for r, flag in zip(records, flags) if flag]
    n_media = sum(1 for r in records if media_paths(r))
    write_records(args.output, missing)

    print(f"Records                  : {len(records):,}")
    print(f"Records with media       : {n_media:,}")
    print(f"Records with missing file: {len(missing):,}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
