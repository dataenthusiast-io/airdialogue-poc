import json

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import config


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(results_path: str, ground_truth_path: str) -> dict:
    results = pd.DataFrame(_load_jsonl(results_path))
    gt = pd.DataFrame(_load_jsonl(ground_truth_path))

    df = results.merge(gt, on="id", how="inner")

    # Schema failures register as misclassifications
    df["predicted_intent"] = df.apply(
        lambda row: row["predicted_intent"] if row["schema_valid"] else "__invalid__",
        axis=1,
    )

    y_true = df["goal"].tolist()
    y_pred = df["predicted_intent"].tolist()

    labels = list(config.classes)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    macro_f1 = report["macro avg"]["f1-score"]

    per_class = {}
    for cls in labels:
        per_class[cls] = {
            "precision": round(report[cls]["precision"], 6),
            "recall": round(report[cls]["recall"], 6),
            "f1": round(report[cls]["f1-score"], 6),
            "support": int(report[cls]["support"]),
        }

    has_invalid = (df["predicted_intent"] == "__invalid__").any()
    cm_labels = labels + (["__invalid__"] if has_invalid else [])
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    cm_dict = {
        tl: {pl: int(cm[i][j]) for j, pl in enumerate(cm_labels)}
        for i, tl in enumerate(labels)
    }

    schema_failures = int((~df["schema_valid"]).sum())
    schema_failure_rate = schema_failures / len(df)
    mean_prompt_tokens = float(df["prompt_tokens"].mean())
    mean_completion_tokens = float(df["completion_tokens"].mean())
    mean_cost_usd = float(df["cost_usd"].mean())
    total_cost_usd = float(df["cost_usd"].sum())
    projected_cost = mean_cost_usd * config.costs.industrial_scale_benchmark

    return {
        "classification": {
            "macro_f1": round(macro_f1, 6),
            "per_class": per_class,
            "confusion_matrix": cm_dict,
        },
        "operational": {
            "total_calls": len(df),
            "schema_failures": schema_failures,
            "schema_failure_rate": round(schema_failure_rate, 6),
            "mean_prompt_tokens": round(mean_prompt_tokens, 2),
            "mean_completion_tokens": round(mean_completion_tokens, 2),
            "mean_cost_usd": round(mean_cost_usd, 8),
            "total_cost_usd": round(total_cost_usd, 6),
            "projected_cost_usd_10k": round(projected_cost, 4),
        },
        "hypothesis": {
            "h1_supported": macro_f1 >= config.plots.h1_threshold,
            "h2_supported": schema_failure_rate < 0.05,
        },
    }
