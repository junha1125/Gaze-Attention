#!/usr/bin/env python3
r"""Prepare the LLaVA-OneVision-Data subsets that are distributed as JSON + tar.gz archives.

Example:
    python data_preparation/llava_onevision/3_extract_archives.py \
        --source-root datasets/LLaVA-OneVision/datasets/OneVisionData_origin \
        --target-root datasets/LLaVA-OneVision/datasets/OneVisionData

Handles the subsets "cambrian(filtered)", "ureader_kg" and "ureader_qa":
  * the image archives (*images_*.tar.gz) are extracted into <target-root>/<subset>/image
  * the annotation JSONs are copied with image paths rewritten to
    "<name of target-root>/<subset>/image/<original path>"
"""

import argparse
import json
import os
import tarfile

from tqdm import tqdm

ARCHIVE_SUBSETS = ("cambrian", "ureader_kg", "ureader_qa")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", required=True, help="Folder of the downloaded LLaVA-OneVision-Data repository")
    parser.add_argument("--target-root", required=True, help="Output folder (must differ from --source-root)")
    args = parser.parse_args()

    source_root = os.path.abspath(args.source_root)
    target_root = os.path.abspath(args.target_root)
    if source_root == target_root:
        raise ValueError("--source-root and --target-root must be different folders")
    path_prefix = os.path.basename(target_root.rstrip("/"))

    subsets = [d for d in sorted(os.listdir(source_root)) if os.path.isdir(os.path.join(source_root, d)) and any(key in d for key in ARCHIVE_SUBSETS)]
    for subset in subsets:
        source_dir = os.path.join(source_root, subset)
        target_dir = os.path.join(target_root, subset)
        os.makedirs(target_dir, exist_ok=True)

        for name in sorted(os.listdir(source_dir)):
            source_path = os.path.join(source_dir, name)

            if name.endswith(".json"):
                with open(source_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    print(f"[WARN] {source_path} is not a list of samples, skipped")
                    continue
                for item in tqdm(data, desc=f"{subset}/{name}"):
                    if "image" in item:
                        item["image"] = f"{path_prefix}/{subset}/image/{item['image']}"
                with open(os.path.join(target_dir, name), "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"[DONE] {len(data):,} samples -> {os.path.join(target_dir, name)}")

            elif "images_" in name and name.endswith(".tar.gz"):
                image_dir = os.path.join(target_dir, "image")
                os.makedirs(image_dir, exist_ok=True)
                with tarfile.open(source_path, "r:gz") as tar:
                    tar.extractall(path=image_dir)
                print(f"[DONE] extracted {source_path} -> {image_dir}")


if __name__ == "__main__":
    main()
