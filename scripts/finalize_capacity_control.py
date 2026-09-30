"""Finalize the capacity-controlled ablation evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from aggregate_results import mcnemar_exact, paired_bootstrap, read_predictions


VARIANTS = ("gap", "wide_gap", "gap_gmp", "fractal", "fractal_only")
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
    summary = {"protocol": "capacity-controlled core ablation", "seeds": args.seeds, "variants": {}}
    for variant in VARIANTS:
        runs = []
        for seed in args.seeds:
            result_dir = args.results_root / f"{variant}_seed_{seed}"
            metrics_path = result_dir / "metrics.json"
            predictions_path = result_dir / "predictions.csv"
            if not metrics_path.exists() or not predictions_path.exists():
                raise SystemExit(f"Missing completed result for {variant}, seed {seed}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            true, predicted = read_predictions(predictions_path)
            runs.append({"seed": seed, "metrics": metrics, "true": true, "predicted": predicted})
        loaded[variant] = runs
        summary["variants"][variant] = {
            "per_seed": [
                {"seed": run["seed"], **{name: run["metrics"][name] for name in METRICS}}
                for run in runs
            ],
            "mean": {name: float(np.mean([run["metrics"][name] for run in runs])) for name in METRICS},
            "sample_std": {name: float(np.std([run["metrics"][name] for run in runs], ddof=1)) for name in METRICS},
            "parameter_count": runs[0]["metrics"]["parameter_count"],
            "test_samples": [run["metrics"]["samples"] for run in runs],
        }

    comparisons = {}
    for baseline, candidate in (
        ("gap", "fractal"),
        ("wide_gap", "fractal"),
        ("gap_gmp", "fractal"),
        ("fractal", "fractal_only"),
    ):
        name = f"{baseline}_vs_{candidate}"
        comparisons[name] = {}
        for baseline_run, candidate_run in zip(loaded[baseline], loaded[candidate]):
            seed = baseline_run["seed"]
            if seed != candidate_run["seed"] or not np.array_equal(baseline_run["true"], candidate_run["true"]):
                raise SystemExit(f"Prediction order mismatch for {name}, seed {seed}")
            comparisons[name][str(seed)] = {
                "candidate_minus_baseline_bootstrap": paired_bootstrap(
                    baseline_run["true"], baseline_run["predicted"], candidate_run["predicted"],
                    args.bootstrap_iterations, seed,
                ),
                "mcnemar_exact": mcnemar_exact(
                    baseline_run["true"], baseline_run["predicted"], candidate_run["predicted"],
                ),
            }
    summary["paired_tests"] = comparisons

    summary_path = args.output_dir / "capacity_control_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    table_path = args.output_dir / "capacity_control_table.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["variant", "parameter_count"]
        fields += [f"{metric}_mean" for metric in METRICS]
        fields += [f"{metric}_std" for metric in METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant in VARIANTS:
            item = summary["variants"][variant]
            row = {"variant": variant, "parameter_count": item["parameter_count"]}
            row.update({f"{metric}_mean": item["mean"][metric] for metric in METRICS})
            row.update({f"{metric}_std": item["sample_std"][metric] for metric in METRICS})
            writer.writerow(row)
    print(json.dumps({"summary": str(summary_path), "table": str(table_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
