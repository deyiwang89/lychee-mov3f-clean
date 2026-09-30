"""Audit the archived experiment protocol before rebuilding the rebuttal results.

This script is read-only. It compares annotation counts, logged training counts,
seed directories, and prediction CSV sizes, then writes a machine-readable report.
Run it from the project root or pass --project-root explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def count_lines(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def csv_summary(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors = [row for row in rows if row.get("Status") == "Incorrect"]
    return {
        "exists": True,
        "rows": len(rows),
        "errors": len(errors),
        "true_labels": sorted({row.get("True Label") for row in rows}),
    }


def logged_counts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    if "num_train" not in text:
        text = raw.decode("utf-16", errors="replace")
    seeds = [int(value) for value in re.findall(r"SEED\s*=\s*(\d+)", text)]
    train_counts = [int(value) for value in re.findall(r"num_train\s*\|\s*(\d+)", text)]
    val_counts = [int(value) for value in re.findall(r"num_val\s*\|\s*(\d+)", text)]
    return [{
        "seed_mentions": seeds,
        "num_train_values": train_counts,
        "num_val_values": val_counts,
    }]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()

    dataset_counts = {}
    for name in ("cls_train.txt", "cls_test.txt"):
        path = root / name
        dataset_counts[name] = count_lines(path) if path.exists() else None

    all_images = []
    for split in ("train", "test"):
        directory = root / "datasets" / split
        if directory.exists():
            all_images.extend(directory.rglob("*.jpg"))

    seed_dirs = sorted(p.name for p in root.glob("logs_mobilenetv3_fractal_seed_*"))
    report = {
        "project_root": str(root),
        "dataset": {
            "annotation_counts": dataset_counts,
            "image_count_in_datasets_train_test": len(all_images),
            "effective_annotation_train_fraction": (
                dataset_counts.get("cls_train.txt", 0) / len(all_images) if all_images else None
            ),
        },
        "seed_directories": seed_dirs,
        "logged_training_counts": {
            "3seed_experiment_log.txt": logged_counts(root / "3seed_experiment_log.txt"),
            "ratio_experiment_log.txt": logged_counts(root / "ratio_experiment_log.txt"),
        },
        "prediction_csv": csv_summary(
            root / "predict_results" / "mobilenetv3_fractal" / "mobilenetv3_fractal_prediction_results.csv"
        ),
        "paper_claims": {
            "claimed_train_protocol": "20% training / 80% evaluation",
            "claimed_seeds": [42, 43, 44],
            "claimed_top1": 97.64,
        },
        "status": "audit_only_no_files_modified",
    }
    output = args.output or root  / "archive_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Audit report written to: {output}")


if __name__ == "__main__":
    main()
