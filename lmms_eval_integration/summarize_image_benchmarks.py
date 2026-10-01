"""Sub-scores of the additional image benchmarks (counting, fine-grained perception, spatial reasoning).

Usage:
    python lmms_eval_integration/summarize_image_benchmarks.py logs/lmms_eval/<run name> [more runs ...]

Reads the per-sample logs (*_samples_<task>.jsonl) that lmms-eval writes with --log_samples and prints
the scores of the three tables in the appendix of the paper. "Avg" is the mean of the columns of a
table; the MME category scores (0-200) are divided by 2 before averaging.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

TASKS = ["countbench", "blink_counting", "blink_spatial_relation", "blink_object_localization", "mme", "mmstar", "mmerealworld_lite", "mmsi_bench"]


def load_samples(run_dir, task):
    """Samples of the latest run of a task, or None when the task was not evaluated."""
    files = sorted(glob.glob(os.path.join(run_dir, "**", f"*_samples_{task}.jsonl"), recursive=True))
    if not files:
        return None
    with open(files[-1]) as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values):
    values = list(values)
    return 100.0 * sum(values) / len(values) if values else None


def mme_category_score(samples, category):
    """accuracy + accuracy+ of one MME category (0-200); accuracy+ counts images with both questions correct."""
    per_image = defaultdict(list)
    for s in samples:
        entry = s.get("mme_perception_score") or s.get("mme_cognition_score")
        if entry["category"] == category:
            per_image[entry["question_id"]].append(entry["score"])
    if not per_image:
        return None
    scores = [x for image_scores in per_image.values() for x in image_scores]
    acc = 100.0 * sum(scores) / len(scores)
    acc_plus = 100.0 * sum(1 for image_scores in per_image.values() if len(image_scores) == 2 and sum(image_scores) == 2) / len(per_image)
    return acc + acc_plus


def l2_category_score(samples, category):
    return mean(s["average"]["score"] for s in samples if s["average"]["l2_category"].lower() == category.lower())


def realworld_score(samples, category=None, sub_category=None, task_category=None):
    selected = []
    for s in samples:
        e = s["mme_realworld_score"]
        if category is not None and e["category"].lower() != category.lower():
            continue
        if sub_category is not None and e["sub_category"].lower() != sub_category.lower():
            continue
        if task_category is not None and e["task_category"].lower() != task_category.lower():
            continue
        selected.append(float(e["pred_answer"] == e["answer"]))
    return mean(selected)


def summarize(run_dir):
    data = {task: load_samples(run_dir, task) for task in TASKS}

    def get(task, fn):
        return fn(data[task]) if data[task] is not None else None

    blink = lambda samples: mean(float(s["blink_acc"]["is_correct"]) for s in samples)
    perception = dict(category="Perception")

    counting = [
        ("CountBench", get("countbench", lambda x: mean(s["acc"] for s in x)), 1),
        ("BLINK Counting", get("blink_counting", blink), 1),
        ("MME Count", get("mme", lambda x: mme_category_score(x, "count")), 2),
        ("MMStar Obj. Count", get("mmstar", lambda x: l2_category_score(x, "object counting")), 1),
        ("MME-RW Vehicle", get("mmerealworld_lite", lambda x: realworld_score(x, sub_category="Monitoring", task_category="Vehicle/counting", **perception)), 1),
        ("MME-RW Person", get("mmerealworld_lite", lambda x: realworld_score(x, sub_category="Monitoring", task_category="Person/counting", **perception)), 1),
        ("MME-RW Remote", get("mmerealworld_lite", lambda x: realworld_score(x, sub_category="Remote Sensing", task_category="count", **perception)), 1),
    ]
    fine_grained = [
        ("MME-RW Overall", get("mmerealworld_lite", realworld_score), 1),
        ("MME-RW Perception", get("mmerealworld_lite", lambda x: realworld_score(x, **perception)), 1),
        ("MME-RW Monitoring", get("mmerealworld_lite", lambda x: realworld_score(x, sub_category="Monitoring", **perception)), 1),
        ("MME-RW Driving", get("mmerealworld_lite", lambda x: realworld_score(x, sub_category="Autonomous_Driving", **perception)), 1),
        ("MME-RW Remote", get("mmerealworld_lite", lambda x: realworld_score(x, sub_category="Remote Sensing", **perception)), 1),
        ("MME-RW Vehicle Loc.", get("mmerealworld_lite", lambda x: realworld_score(x, sub_category="Monitoring", task_category="Vehicle/location", **perception)), 1),
    ]
    spatial = [
        ("BLINK Spatial Rel.", get("blink_spatial_relation", blink), 1),
        ("BLINK Obj. Loc.", get("blink_object_localization", blink), 1),
        ("MME Position", get("mme", lambda x: mme_category_score(x, "position")), 2),
        ("MMStar Localization", get("mmstar", lambda x: l2_category_score(x, "localization")), 1),
        ("MMSI Obj.-Reg.", get("mmsi_bench", lambda x: l2_category_score(x, "Positional Relationship (Obj.–Reg.)")), 1),
        ("MMSI Cam.-Reg.", get("mmsi_bench", lambda x: l2_category_score(x, "Positional Relationship (Cam.–Reg.)")), 1),
        ("MMSI Reg.-Reg.", get("mmsi_bench", lambda x: l2_category_score(x, "Positional Relationship (Reg.–Reg.)")), 1),
    ]
    return {"Counting": counting, "Fine-grained perception": fine_grained, "Localization and spatial reasoning": spatial}


def print_table(title, rows):
    names = [name for name, _, _ in next(iter(rows.values()))]
    print(f"\n### {title}\n")
    print("| Run | Avg | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * (len(names) + 1))
    for run, columns in rows.items():
        values = [value for _, value, _ in columns]
        if all(v is not None for v in values):
            avg = f"{sum(value / scale for _, value, scale in columns) / len(columns):.2f}"
        else:
            avg = "-"
        print(f"| {run} | {avg} | " + " | ".join("-" if v is None else f"{v:.2f}" for v in values) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="output folders of run_image_benchmarks.sh")
    args = parser.parse_args()

    summaries = {os.path.basename(os.path.normpath(run)): summarize(run) for run in args.runs}
    for title in ["Counting", "Fine-grained perception", "Localization and spatial reasoning"]:
        print_table(title, {run: tables[title] for run, tables in summaries.items()})


if __name__ == "__main__":
    main()
