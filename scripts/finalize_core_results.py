"""Finalize core three-seed tables, paired tests, and representative seed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from aggregate_results import mcnemar_exact, paired_bootstrap, read_predictions


VARIANTS = ("gap", "gap_matched", "fractal", "fractal_only")
METRICS = ("accuracy", "top5_accuracy", "macro_precision", "macro_recall", "macro_f1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = {}
    summary = {"seeds": args.seeds, "variants": {}}
    for variant in VARIANTS:
        runs = []
        for seed in args.seeds:
            directory = args.results_root / f"{variant}_seed_{seed}"
            metrics_path = directory / "metrics.json"
            predictions_path = directory / "predictions.csv"
            if not metrics_path.exists() or not predictions_path.exists():
                raise SystemExit(f"Missing completed result for {variant}, seed {seed}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            weight_dir = args.weights_root / f"{variant}_seed_{seed}"
            loss_dirs = sorted(weight_dir.glob("loss_*"), key=lambda path: path.stat().st_mtime, reverse=True)
            if not loss_dirs:
                raise SystemExit(f"Missing loss directory for {variant}, seed {seed}")
            val_losses = [float(value) for value in (loss_dirs[0] / "epoch_val_loss.txt").read_text().splitlines() if value.strip()]
            best_epoch = int(np.argmin(val_losses)) + 1
            true, predicted = read_predictions(predictions_path)
            runs.append({
                "seed": seed, "metrics": metrics, "true": true, "predicted": predicted,
                "best_epoch": best_epoch, "best_val_loss": val_losses[best_epoch - 1],
            })
        loaded[variant] = runs
        summary["variants"][variant] = {
            "per_seed": [
                {"seed": run["seed"], "best_epoch": run["best_epoch"], "best_val_loss": run["best_val_loss"], **{name: run["metrics"][name] for name in METRICS}}
                for run in runs
            ],
            "mean": {name: float(np.mean([run["metrics"][name] for run in runs])) for name in METRICS},
            "sample_std": {name: float(np.std([run["metrics"][name] for run in runs], ddof=1)) for name in METRICS},
            "parameter_count": runs[0]["metrics"]["parameter_count"],
            "weights_size_bytes": [run["metrics"]["weights_size_bytes"] for run in runs],
            "test_samples": [run["metrics"]["samples"] for run in runs],
        }

    comparisons = {}
    for baseline, candidate in (
        ("gap", "fractal"),
        ("gap_matched", "fractal"),
        ("fractal_only", "fractal"),
    ):
        comparison_name = f"{baseline}_vs_{candidate}"
        comparisons[comparison_name] = {}
        for baseline_run, candidate_run in zip(loaded[baseline], loaded[candidate]):
            seed = baseline_run["seed"]
            if seed != candidate_run["seed"] or not np.array_equal(baseline_run["true"], candidate_run["true"]):
                raise SystemExit(f"Paired prediction mismatch for {comparison_name}, seed {seed}")
            comparisons[comparison_name][str(seed)] = {
                "candidate_minus_baseline_bootstrap": paired_bootstrap(
                    baseline_run["true"], baseline_run["predicted"], candidate_run["predicted"],
                    args.bootstrap_iterations, seed,
                ),
                "mcnemar_exact": mcnemar_exact(
                    baseline_run["true"], baseline_run["predicted"], candidate_run["predicted"],
                ),
            }
    summary["paired_tests"] = comparisons

    fractal_mean = summary["variants"]["fractal"]["mean"]["accuracy"]
    representative = min(
        loaded["fractal"],
        key=lambda run: (abs(run["metrics"]["accuracy"] - fractal_mean), run["seed"]),
    )
    summary["representative_fractal_seed"] = {
        "selection_rule": "seed whose test accuracy is closest to the three-seed mean; ties use the smaller seed",
        "seed": representative["seed"],
        "accuracy": representative["metrics"]["accuracy"],
    }
    summary_path = args.output_dir / "core_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (args.output_dir / "core_table.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["variant", *[f"{metric}_mean" for metric in METRICS], *[f"{metric}_std" for metric in METRICS], "parameter_count", "mean_weights_size_mb"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant in VARIANTS:
            item = summary["variants"][variant]
            row = {"variant": variant, "parameter_count": item["parameter_count"], "mean_weights_size_mb": np.mean(item["weights_size_bytes"]) / 1024 ** 2}
            row.update({f"{metric}_mean": item["mean"][metric] for metric in METRICS})
            row.update({f"{metric}_std": item["sample_std"][metric] for metric in METRICS})
            writer.writerow(row)
    print(json.dumps({
        "core_summary": str(summary_path),
        "representative_fractal_seed": summary["representative_fractal_seed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
