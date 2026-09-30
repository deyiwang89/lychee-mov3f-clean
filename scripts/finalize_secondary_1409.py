"""Summarize secondary-domain frozen-model results and paired tests."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "secondary_1409_bbox"
SCOPES = {"source_group_representative": "source_group", "all_1409_merged_splits": "all_1409"}
VARIANTS = ("gap", "gap_gmp", "fractal")
SEEDS = (42, 43, 44)


def paired_bootstrap(y_true: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, iterations: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, n, size=n)
        deltas[index] = np.mean(candidate[sample] == y_true[sample]) - np.mean(baseline[sample] == y_true[sample])
    return {
        "accuracy_delta": float(np.mean(candidate == y_true) - np.mean(baseline == y_true)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def mcnemar(y_true: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict:
    baseline_correct = baseline == y_true
    candidate_correct = candidate == y_true
    baseline_wrong_candidate_correct = int(np.sum(~baseline_correct & candidate_correct))
    baseline_correct_candidate_wrong = int(np.sum(baseline_correct & ~candidate_correct))
    discordant = baseline_wrong_candidate_correct + baseline_correct_candidate_wrong
    if discordant == 0:
        p_value = 1.0
    else:
        k = min(baseline_wrong_candidate_correct, baseline_correct_candidate_wrong)
        p_value = min(1.0, 2.0 * sum(math.comb(discordant, i) for i in range(k + 1)) / (2.0 ** discordant))
    return {
        "baseline_wrong_candidate_correct": baseline_wrong_candidate_correct,
        "baseline_correct_candidate_wrong": baseline_correct_candidate_wrong,
        "p_value": float(p_value),
    }


def read_predictions(scope_dir: Path, variant: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = scope_dir / variant / f"seed_{seed}" / "predictions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    true = np.asarray([row["label"] for row in rows], dtype=object)
    pred = np.asarray([row["predicted_label"] for row in rows], dtype=object)
    return true, pred


def main() -> None:
    summary = {}
    table_rows = []
    for scope_name, scope_key in SCOPES.items():
        scope_dir = RESULTS / scope_name
        summary[scope_key] = {"variants": {}, "paired_tests": {}}
        for variant in VARIANTS:
            per_seed = []
            for seed in SEEDS:
                metrics = json.loads((scope_dir / variant / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))
                per_seed.append({"seed": seed, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "samples": metrics["samples"]})
            for metric in ("accuracy", "macro_f1"):
                values = np.asarray([row[metric] for row in per_seed], dtype=float)
                mean = float(values.mean())
                std = float(values.std(ddof=1))
                summary[scope_key]["variants"].setdefault(variant, {})[f"{metric}_mean"] = mean
                summary[scope_key]["variants"][variant][f"{metric}_std"] = std
                table_rows.append({"scope": scope_key, "variant": variant, "metric": metric, "mean": mean, "sample_std": std, "samples": per_seed[0]["samples"]})
        for baseline, candidate in (("gap", "fractal"), ("gap_gmp", "fractal")):
            comparison_key = f"{candidate}_vs_{baseline}"
            summary[scope_key]["paired_tests"][comparison_key] = {}
            for seed in SEEDS:
                y_true, baseline_pred = read_predictions(scope_dir, baseline, seed)
                _, candidate_pred = read_predictions(scope_dir, candidate, seed)
                summary[scope_key]["paired_tests"][comparison_key][str(seed)] = {
                    "paired_bootstrap": paired_bootstrap(y_true, baseline_pred, candidate_pred, seed=seed),
                    "mcnemar_exact": mcnemar(y_true, baseline_pred, candidate_pred),
                }
    (RESULTS / "secondary_1409_final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (RESULTS / "secondary_1409_summary_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
