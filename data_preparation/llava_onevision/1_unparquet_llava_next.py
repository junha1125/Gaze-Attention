#!/usr/bin/env python3
r"""Convert LLaVA-NeXT-Data (parquet) into one annotation JSON and an image folder.

Example:
    python data_preparation/llava_onevision/1_unparquet_llava_next.py \
        --source datasets/LLaVA-OneVision/datasets/LLaVA-Next

Input : <source>/data/*.parquet                     (downloaded from lmms-lab/LLaVA-NeXT-Data)
Output: <source>/image/<id>.jpg
        <source>/llava_next_fit_mix_filtered_text_wild_738590.json
The image paths in the JSON are relative to the parent folder of <source>, e.g.
"LLaVA-Next/image/000000033471.jpg".
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Folder of the downloaded LLaVA-NeXT-Data repository")
    parser.add_argument("--json-name", default="llava_next_fit_mix_filtered_text_wild_738590.json")
    args = parser.parse_args()

    source_root = os.path.abspath(args.source)
    base_name = os.path.basename(source_root)
    image_dir = os.path.join(source_root, "image")
    os.makedirs(image_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(source_root, "data", "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {os.path.join(source_root, 'data')}")

    ds = load_dataset("parquet", data_files={"train": parquet_files}, split="train").cast_column("image", HFImage(decode=False))

    converted, bad_samples = [], []
    for row in tqdm(ds, desc=f"Converting {base_name}"):
        record = {"id": row["id"], "conversations": row["conversations"]}

        image_bytes = row["image"]["bytes"] if row.get("image") else None
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
                record["image"] = os.path.join(base_name, "image", image_name)
            except (UnidentifiedImageError, OSError) as e:
                bad_samples.append({"id": row["id"], "reason": str(e)})
                continue

        converted.append(record)

    json_path = os.path.join(source_root, args.json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=4, ensure_ascii=False)

    if bad_samples:
        log_path = os.path.join(source_root, f"{base_name}_bad_samples.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "reason"])
            writer.writeheader()
            writer.writerows(bad_samples)
        print(f"{len(bad_samples):,} undecodable samples were skipped, see {log_path}")

    print(f"Saved {len(converted):,} samples to {json_path}")
    print(f"Images written to {image_dir}")


if __name__ == "__main__":
    main()
