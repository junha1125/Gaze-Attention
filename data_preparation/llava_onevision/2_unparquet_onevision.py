#!/usr/bin/env python3
r"""Convert the parquet subsets of LLaVA-OneVision-Data into annotation JSONs and image folders.

Example:
    python data_preparation/llava_onevision/2_unparquet_onevision.py \
        --source-root datasets/LLaVA-OneVision/datasets/OneVisionData_origin \
        --target-root datasets/LLaVA-OneVision/datasets/OneVisionData

Input : <source-root>/<subset>/*.parquet            (downloaded from lmms-lab/LLaVA-OneVision-Data)
Output: <target-root>/<subset>/image/<id>.jpg
        <target-root>/<subset>/<json name>           (names are listed in json_mapping.csv)
The image paths in the JSONs are relative to the parent folder of <target-root>, e.g.
"OneVisionData/llavar_gpt4_20k/image/100000803014.jpg".

Subsets whose target folder already exists are skipped, so an interrupted run can be resumed
after deleting the folder of the subset that was being converted.
Subsets without parquet files (cambrian, ureader_kg, ureader_qa) are handled by 3_extract_archives.py.
"""

import argparse
import csv
import glob
import json
import os
from io import BytesIO

from datasets import load_dataset, Image as HFImage
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def load_json_mapping():
    mapping_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json_mapping.csv")
    with open(mapping_csv, newline="", encoding="utf-8") as f:
        return {row["folder_name"]: row["json_file"] for row in csv.DictReader(f)}


def convert_subset(source_dir, target_root, json_mapping):
    subset = os.path.basename(source_dir.rstrip("/"))
    path_prefix = os.path.basename(target_root.rstrip("/"))

    parquet_files = sorted(glob.glob(os.path.join(source_dir, "*.parquet")))
    if not parquet_files:
        parquet_files = sorted(glob.glob(os.path.join(source_dir, "**", "*.parquet"), recursive=True))
    if not parquet_files:
        print(f"[SKIP] '{subset}' has no parquet files")
        return

    target_dir = os.path.join(target_root, subset)
    image_dir = os.path.join(target_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    print(f"[START] '{subset}': {len(parquet_files)} parquet files")

    ds = load_dataset("parquet", data_files={"train": parquet_files}, split="train").cast_column("image", HFImage(decode=False))

    converted, bad_samples = [], []
    for row in tqdm(ds, desc=f"  {subset}"):
        record = {"id": row["id"], "conversations": row.get("conversations", [])}

        image_bytes = row["image"].get("bytes") if row.get("image") else None
        if image_bytes:
            try:
                image = Image.open(BytesIO(image_bytes))
                image.load()
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image_name = f"{row['id']}.jpg"
                save_path = os.path.join(image_dir, image_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                image.save(save_path)
                record["image"] = os.path.join(path_prefix, subset, "image", image_name)
            except (UnidentifiedImageError, OSError) as e:
                bad_samples.append({"id": row["id"], "reason": str(e)})
                continue

        converted.append(record)

    json_path = os.path.join(target_dir, json_mapping.get(subset, f"{subset}.json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    if bad_samples:
        log_path = os.path.join(target_dir, f"{subset}_bad_samples.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "reason"])
            writer.writeheader()
            writer.writerows(bad_samples)
        print(f"[WARN] {len(bad_samples):,} undecodable samples were skipped, see {log_path}")

    print(f"[DONE] '{subset}': {len(converted):,} samples -> {json_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", required=True, help="Folder of the downloaded LLaVA-OneVision-Data repository")
    parser.add_argument("--target-root", required=True, help="Output folder (must differ from --source-root)")
    parser.add_argument("--subsets", nargs="*", default=None, help="Only convert these subsets (default: all)")
    args = parser.parse_args()

    source_root = os.path.abspath(args.source_root)
    target_root = os.path.abspath(args.target_root)
    if source_root == target_root:
        raise ValueError("--source-root and --target-root must be different folders")
    os.makedirs(target_root, exist_ok=True)

    json_mapping = load_json_mapping()
    for entry in sorted(os.listdir(source_root)):
        source_dir = os.path.join(source_root, entry)
        if entry.startswith(".") or not os.path.isdir(source_dir):
            continue
        if args.subsets and entry not in args.subsets:
            continue
        if os.path.exists(os.path.join(target_root, entry)):
            print(f"[SKIP] '{entry}' already exists in the target folder")
            continue
        convert_subset(source_dir, target_root, json_mapping)


if __name__ == "__main__":
    main()
