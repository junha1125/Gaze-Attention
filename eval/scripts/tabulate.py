"""Collect the scores of all benchmarks into one table.

Usage (from the eval/ folder):
    python scripts/tabulate.py --eval_dir eval --out_pivot pivot.xlsx --out_all_results all_results.csv

Reads eval/<benchmark>/experiments.csv. When a model was evaluated several times on a benchmark,
the latest result is used.
"""

import os
import json
import pandas as pd
import argparse

# benchmark -> column of experiments.csv that is reported ("100x_" = multiply by 100)
EVALS = {
    "mme": "Perception",
    "mmbench_en": "100x_circular_accuracy",
    "mmbench_cn": "100x_circular_accuracy",
    "seed": "100x_accuracy",
    "gqa": "accuracy",

    "scienceqa": "100x_multimodal_acc",
    "mmmu": "100x_accuracy",
    "mathvista": "100x_accuracy",
    "ai2d": "accuracy",

    "chartqa": "accuracy",
    "ocrbench": "total_accuracy['accuracy']",
    "textvqa": "accuracy",

    "mmvp": "accuracy",
    "realworldqa": "accuracy",
    "coco": "100x_accuracy",
    "ade": "100x_accuracy",
    "omni": "100x_accuracy",
}


def tabulate_results(eval_dir, experiment_csv_fname, out_pivot_fname, out_all_results_fname):
    if not os.path.exists(eval_dir):
        raise ValueError(f"eval_dir {eval_dir} does not exist")
    print(f"eval_dir: {eval_dir}")

    # gather results from each eval
    dfs = []
    for eval_name, column in EVALS.items():
        results_path = os.path.join(eval_dir, eval_name, experiment_csv_fname)
        if not os.path.exists(results_path):
            print(f"Skipping {eval_name} as no results file found")
            continue

        df = pd.read_csv(results_path)
        if column.startswith("100x_"):
            df["accuracy"] = df[column[5:]] * 100
        elif column == "total_accuracy['accuracy']":
            df["accuracy"] = df["total_accuracy"].apply(lambda x: json.loads(x.replace("'", '"'))["accuracy"])
        else:
            df["accuracy"] = df[column]

        df["eval_name"] = eval_name
        df = df.sort_values("time")
        df = df.drop_duplicates("model", keep="last")
        dfs.append(df[["time", "eval_name", "model", "accuracy"]])

    if not dfs:
        raise ValueError(f"No {experiment_csv_fname} found in {eval_dir}")

    all_results = pd.concat(dfs)
    all_results.sort_values("time").to_csv(out_all_results_fname, index=False)
    print(f"Saved all results to {out_all_results_fname}")

    pivot = all_results.pivot(index="model", columns="eval_name", values="accuracy")
    pivot = pivot[[name for name in EVALS if name in pivot.columns]]

    if out_pivot_fname.endswith(".xlsx"):
        pivot.to_excel(out_pivot_fname)
    else:
        pivot.to_csv(out_pivot_fname)
    print(f"Saved pivot table to {out_pivot_fname}")
    print(pivot.round(1).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabulate experiment results")
    parser.add_argument("--eval_dir", type=str, default="eval", help="Directory containing evaluation results")
    parser.add_argument("--experiment_csv", type=str, default="experiments.csv", help="Name of the CSV file containing experiment results")
    parser.add_argument("--out_pivot", type=str, default="pivot.xlsx", help="Name of the output file (Excel or CSV)")
    parser.add_argument("--out_all_results", type=str, default="all_results.csv", help="Name of the CSV file to save all results")

    args = parser.parse_args()

    tabulate_results(args.eval_dir, args.experiment_csv, args.out_pivot, args.out_all_results)
