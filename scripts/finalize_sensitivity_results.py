"""Summarize pre-specified seed-42 sensitivity runs without test-based selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SETTINGS = {
    "main": {"variant_dir": "fractal_seed_42", "box_sizes": [1, 2, 4], "threshold": 0.5, "fusion": "concatenation", "location": "final"},
    "scales_12": {"variant_dir": "fractal_seed_42", "box_sizes": [1, 2], "threshold": 0.5, "fusion": "concatenation", "location": "final"},
    "threshold_03": {"variant_dir": "fractal_seed_42", "box_sizes": [1, 2, 4], "threshold": 0.3, "fusion": "concatenation", "location": "final"},
    "threshold_07": {"variant_dir": "fractal_seed_42", "box_sizes": [1, 2, 4], "threshold": 0.7, "fusion": "concatenation", "location": "final"},
    "gated": {"variant_dir": "gated_seed_42", "box_sizes": [1, 2, 4], "threshold": 0.5, "fusion": "gated", "location": "final"},
    "penultimate": {"variant_dir": "penultimate_seed_42", "box_sizes": [1, 2, 4], "threshold": 0.5, "fusion": "concatenation", "location": "penultimate"},
}
METRICS = ("accuracy", "top5_accuracy", "macro_precision", "macro_recall", "macro_f1")


def best_validation(weight_dir: Path) -> tuple[int, float]:
    loss_dirs = sorted(weight_dir.glob("loss_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not loss_dirs:
        raise SystemExit(f"Missing loss log: {weight_dir}")
    values = [float(value) for value in (loss_dirs[0] / "epoch_val_loss.txt").read_text().splitlines() if value.strip()]
    index = int(np.argmin(values))
    return index + 1, values[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-results", type=Path, required=True)
    parser.add_argument("--core-weights", type=Path, required=True)
    parser.add_argument("--sensitivity-results", type=Path, required=True)
    parser.add_argument("--sensitivity-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirmation-seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, setting in SETTINGS.items():
        if name == "main":
            result_dir = args.core_results / setting["variant_dir"]
            weight_dir = args.core_weights / setting["variant_dir"]
        else:
            result_dir = args.sensitivity_results / name / setting["variant_dir"]
            weight_dir = args.sensitivity_weights / name / setting["variant_dir"]
        metrics_path = result_dir / "metrics.json"
        if not metrics_path.exists():
            raise SystemExit(f"Missing sensitivity metrics: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        best_epoch, best_val_loss = best_validation(weight_dir)
        rows.append({
            "setting": name,
            "box_sizes": ",".join(map(str, setting["box_sizes"])),
            "threshold": setting["threshold"],
            "fusion": setting["fusion"],
            "location": setting["location"],
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "parameter_count": metrics["parameter_count"],
            "weights_size_mb": metrics["weights_size_bytes"] / 1024 ** 2,
            **{metric: metrics[metric] for metric in METRICS},
        })
    alternatives = [row for row in rows if row["setting"] != "main"]
    selected = min(alternatives, key=lambda row: (row["best_val_loss"], row["setting"]))
    report = {
        "screening_seed": 42,
        "selection_metric": "minimum validation cross-entropy; test metrics are not used for selection",
        "settings": rows,
        "selected_alternative_for_three_seed_confirmation": selected["setting"],
    }
    selected_setting = SETTINGS[selected["setting"]]
    confirmation = []
    for seed in args.confirmation_seeds:
        run_name = selected_setting["variant_dir"].replace("_seed_42", f"_seed_{seed}")
        result_dir = args.sensitivity_results / selected["setting"] / run_name
        weight_dir = args.sensitivity_weights / selected["setting"] / run_name
        metrics_path = result_dir / "metrics.json"
        if not metrics_path.exists():
            report["selected_alternative_confirmation_pending_seeds"] = [
                pending_seed for pending_seed in args.confirmation_seeds
                if not (
                    args.sensitivity_results
                    / selected["setting"]
                    / selected_setting["variant_dir"].replace("_seed_42", f"_seed_{pending_seed}")
                    / "metrics.json"
                ).exists()
            ]
            confirmation = []
            break
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        best_epoch, best_val_loss = best_validation(weight_dir)
        confirmation.append({
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            **{metric: metrics[metric] for metric in METRICS},
        })
    if confirmation:
        report["selected_alternative_three_seed_confirmation"] = {
            "setting": selected["setting"],
            "seeds": args.confirmation_seeds,
            "per_seed": confirmation,
            "mean": {
                metric: float(np.mean([run[metric] for run in confirmation]))
                for metric in METRICS
            },
            "sample_std": {
                metric: float(np.std([run[metric] for run in confirmation], ddof=1))
                for metric in METRICS
            },
        }
    (args.output_dir / "sensitivity_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "sensitivity_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "selected_alternative_for_three_seed_confirmation": selected["setting"],
        "best_val_loss": selected["best_val_loss"],
    }, indent=2))


if __name__ == "__main__":
    main()
