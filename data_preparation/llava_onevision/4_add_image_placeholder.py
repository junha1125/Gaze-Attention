#!/usr/bin/env python3
r"""Make sure every image sample has exactly one "<image>" placeholder.

Example:
    python data_preparation/llava_onevision/4_add_image_placeholder.py \
        --dataset-root datasets/LLaVA-OneVision

For all annotation JSONs listed in single_image.yaml (paths relative to --dataset-root):
  * no placeholder in the human turns  -> "<image>\\n" is prepended to the first human turn
  * more than one placeholder          -> all are removed and one is prepended to the first human turn
Files are rewritten only when something changed.
"""

import argparse
import json
import os

import yaml
from tqdm import tqdm

DEFAULT_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "single_image.yaml")


def process_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    added = cleaned = 0

    for item in tqdm(data, desc=os.path.basename(path)):
        convs = item.get("conversations")
        if "image" not in item or not isinstance(convs, list):
            continue
        human_idxs = [i for i, c in enumerate(convs) if c.get("from") == "human"]
        if not human_idxs:
            continue

        n_placeholders = sum(convs[i].get("value", "").count("<image>") for i in human_idxs)
        first = human_idxs[0]
        if n_placeholders == 0:
            convs[first]["value"] = "<image>\n" + convs[first].get("value", "")
            added += 1
        elif n_placeholders > 1:
            for i in human_idxs:
                convs[i]["value"] = convs[i]["value"].replace("<image>", "").strip()
            convs[first]["value"] = "<image>\n" + convs[first]["value"]
            cleaned += 1

    if added or cleaned:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"{os.path.basename(path)}: {added} placeholders added, {cleaned} samples with duplicated placeholders fixed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, help="Folder that contains datasets/ (json_path entries are relative to it)")
    parser.add_argument("--yaml", default=DEFAULT_YAML, help="Dataset list (default: single_image.yaml next to this script)")
    args = parser.parse_args()

    with open(args.yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    missing = []
    for ds in cfg.get("datasets", []):
        json_path = os.path.join(args.dataset_root, ds["json_path"])
        if os.path.isfile(json_path):
            process_json(json_path)
        else:
            missing.append(json_path)
    if missing:
        print(f"\n[WARN] {len(missing)} annotation files were not found:")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
