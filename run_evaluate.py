import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import config
from src.evaluate import evaluate
from src.visualize import generate_all_plots


def _abs(rel_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)


def _load_jsonl_df(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _print_confusion_matrix(cm_dict: dict) -> None:
    true_labels = list(config.classes)
    all_pred = set()
    for row in cm_dict.values():
        all_pred.update(row.keys())
    pred_labels = [l for l in true_labels if l in all_pred]
    if "__invalid__" in all_pred:
        pred_labels.append("__invalid__")

    col_w = max(len(l) for l in pred_labels + true_labels) + 2
    header = f"{'':>{col_w}}" + "".join(f"{l:>{col_w}}" for l in pred_labels)
    print("  " + "-" * len(header))
    print("  " + header)
    print("  " + "-" * len(header))
    for tl in true_labels:
        row_vals = "".join(f"{cm_dict[tl].get(pl, 0):>{col_w}}" for pl in pred_labels)
        print(f"  {tl:>{col_w}}{row_vals}")
    print("  " + "-" * len(header))


if __name__ == "__main__":
    results_path = _abs(config.paths.results)
    ground_truth_path = _abs(config.paths.ground_truth)
    report_path = os.path.join(_abs(config.paths.results_dir), "evaluation_report.json")

    print("=" * 60)
    print("PHASE 4 — Evaluation")
    print("=" * 60)

    report = evaluate(results_path, ground_truth_path)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written: {report_path}\n")

    cls = report["classification"]
    ops = report["operational"]
    hyp = report["hypothesis"]

    h1_flag = "H1 SUPPORTED" if hyp["h1_supported"] else "H1 NOT SUPPORTED"
    h2_flag = "H2 SUPPORTED" if hyp["h2_supported"] else "H2 NOT SUPPORTED"

    print("=" * 60)
    print("INTENT CLASSIFICATION — EVALUATION SUMMARY")
    print("-" * 60)
    print(f"Macro F1:          {cls['macro_f1']:.3f}     → {h1_flag} (threshold: {config.plots.h1_threshold})")
    print(f"Schema failures:   {ops['schema_failures']} ({ops['schema_failure_rate']*100:.1f}%)  → {h2_flag} (threshold: <5%)")
    print("-" * 60)
    print(f"Cost per call:     ${ops['mean_cost_usd']:.6f}")
    print(f"Total ({ops['total_calls']} calls): ${ops['total_cost_usd']:.3f}")
    print(f"Projected ({config.costs.industrial_scale_benchmark:,}):   ${ops['projected_cost_usd_10k']:.2f}")
    print("=" * 60)

    print("\nConfusion Matrix:")
    _print_confusion_matrix(cls["confusion_matrix"])

    df = _load_jsonl_df(results_path)
    generate_all_plots(report, df)

    print("\nEvaluation complete.")
