import os
import json
import random

from datasets import load_dataset

from src.config import config


def _base_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(rel_path: str) -> str:
    return os.path.join(_base_dir(), rel_path)


def _write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run() -> None:
    print("Loading dataset...")
    ds = load_dataset(
        config.dataset.name,
        config.dataset.config,
        split=config.dataset.split,
        cache_dir=_abs(config.paths.raw_data_cache),
    )

    print(f"Total records loaded: {len(ds)}")

    if config.dataset.filter_correct_sample:
        ds = ds.filter(lambda x: x["correct_sample"] is True)
        print(f"After correct_sample filter: {len(ds)}")

    valid_classes = set(config.classes)
    ds = ds.filter(lambda x: x["intent"]["goal"] in valid_classes)
    print(f"After intent class filter: {len(ds)}")

    ds = ds.filter(lambda x: x["dialogue"] is not None and len(x["dialogue"]) > 0)
    print(f"After empty dialogue filter: {len(ds)}")

    # Group by class
    by_class: dict[str, list[dict]] = {c: [] for c in config.classes}
    for idx, record in enumerate(ds):
        goal = record["intent"]["goal"]
        by_class[goal].append({"_idx": idx, "record": record})

    print("\nPer-class counts after filtering:")
    for cls, items in by_class.items():
        print(f"  {cls}: {len(items)}")

    # Build evaluation sample
    rng = random.Random(config.sampling.random_seed)
    sample_rows: list[dict] = []
    ground_truth_rows: list[dict] = []

    for cls in config.classes:
        chosen = rng.sample(by_class[cls], config.sampling.per_class)
        for item in chosen:
            row_id = str(item["_idx"]).zfill(6)
            sample_rows.append({
                "id": row_id,
                "dialogue": "\n".join(item["record"]["dialogue"]),
            })
            ground_truth_rows.append({
                "id": row_id,
                "goal": cls,
            })

    _write_jsonl(sample_rows, _abs(config.paths.sample))
    _write_jsonl(ground_truth_rows, _abs(config.paths.ground_truth))

    print(f"\nSample written: {len(sample_rows)} rows")
    print(f"Ground truth written: {len(ground_truth_rows)} rows")

    # Validate
    assert len(sample_rows) == config.sampling.total, \
        f"Expected {config.sampling.total} rows, got {len(sample_rows)}"

    class_counts: dict[str, int] = {}
    for row in ground_truth_rows:
        class_counts[row["goal"]] = class_counts.get(row["goal"], 0) + 1
    for cls in config.classes:
        assert class_counts[cls] == config.sampling.per_class, \
            f"Expected {config.sampling.per_class} rows for '{cls}', got {class_counts[cls]}"

    for row in sample_rows:
        assert "goal" not in row, f"'goal' field found in sample row {row['id']}"

    print("\n--- Validation Summary ---")
    print(f"{'File':<30} {'Rows':<8} {'Fields'}")
    print("-" * 55)
    print(f"{'sample.jsonl':<30} {len(sample_rows):<8} id, dialogue")
    print(f"{'ground_truth.jsonl':<30} {len(ground_truth_rows):<8} id, goal")
    print(f"\nClass distribution:")
    for cls in config.classes:
        print(f"  {cls}: {class_counts[cls]}")
    print("\nAll assertions passed.")


if __name__ == "__main__":
    run()
