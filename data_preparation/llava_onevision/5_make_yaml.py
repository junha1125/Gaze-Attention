#!/usr/bin/env python3
r"""Validate the prepared data and write the yaml file that is passed to --data_path.

Example:
    python data_preparation/llava_onevision/5_make_yaml.py \
        --dataset-root datasets/LLaVA-OneVision \
        --output datasets/LLaVA-OneVision/onevision_single_image.yaml

    # LLaVA-OneVision + a Cambrian-7M subset (see data_preparation/README.md)
    python data_preparation/llava_onevision/5_make_yaml.py \
        --dataset-root datasets/LLaVA-OneVision \
        --extra datasets/Cambrian-10M/jsons/Cambrian7M_filtered_40p_onevision_path.jsonl \
        --output datasets/LLaVA-OneVision/onevision_cambrian.yaml

The output lists every annotation file with its absolute path. With --check-images (default)
the script also verifies that all image files exist under <dataset-root>/datasets.
"""

import argparse
import json
import math
import os

import yaml

DEFAULT_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "single_image.yaml")


def load_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def num_used(total, strategy):
    """Number of samples used by the dataloader for a sampling strategy."""
    if ":" not in strategy:
        return total
    _, number = strategy.split(":")
    if "%" in number:
        return math.ceil(int(number.split("%")[0]) * total / 100)
    return min(int(number), total)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, help="Folder that contains datasets/ (json_path entries are relative to it)")
    parser.add_argument("--output", required=True, help="Output yaml file")
    parser.add_argument("--yaml", default=DEFAULT_YAML, help="Dataset list (default: single_image.yaml next to this script)")
    parser.add_argument("--extra", nargs="*", default=[], help="Additional annotation files that are used completely")
    parser.add_argument("--no-check-images", dest="check_images", action="store_false", help="Skip the image existence check")
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    image_root = os.path.join(dataset_root, "datasets")

    with open(args.yaml, "r", encoding="utf-8") as f:
        entries = yaml.safe_load(f)["datasets"]
    entries = [{"json_path": os.path.join(dataset_root, e["json_path"]), "sampling_strategy": str(e.get("sampling_strategy", "all"))} for e in entries]
    entries += [{"json_path": os.path.abspath(path), "sampling_strategy": "all"} for path in args.extra]

    print(f"{'annotation file':<72} {'total':>10} {'used':>10} {'images':>10} {'missing':>8}")
    total = used = with_image = missing = 0
    not_found = []
    for entry in entries:
        path = entry["json_path"]
        if not os.path.isfile(path):
            not_found.append(path)
            print(f"{'[NOT FOUND] ' + os.path.basename(path):<72}")
            continue
        samples = load_samples(path)
        n_image = n_missing = 0
        for sample in samples:
            image = sample.get("image")
            if not image:
                continue
            n_image += 1
            if args.check_images:
                images = image if isinstance(image, list) else [image]
                if any(not os.path.isfile(os.path.join(image_root, p)) for p in images):
                    n_missing += 1
        n_used = num_used(len(samples), entry["sampling_strategy"])
        print(f"{os.path.basename(path):<72} {len(samples):>10,} {n_used:>10,} {n_image:>10,} {n_missing:>8,}")
        total += len(samples)
        used += n_used
        with_image += n_image
        missing += n_missing

    print("-" * 114)
    print(f"{'TOTAL (' + str(len(entries)) + ' files)':<72} {total:>10,} {used:>10,} {with_image:>10,} {missing:>8,}")

    if not_found or missing:
        raise SystemExit(f"\n{len(not_found)} annotation files and {missing:,} images are missing. The yaml file was not written.")

    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump({"datasets": entries}, f, sort_keys=False, width=1000)
    print(f"\nSaved {args.output}")
    print(f"Train with: --data_path {os.path.abspath(args.output)} --image_folder {image_root}")


if __name__ == "__main__":
    main()
