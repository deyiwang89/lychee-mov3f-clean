"""Aggregate per-seed evaluation outputs for the rebuttal tables.

Each metric file is expected at ``<results-root>/<variant>_seed_<seed>/metrics.json``.
The script reports mean and sample standard deviation across seeds and, when a
comparison variant is supplied, paired bootstrap and exact McNemar statistics
on the identical test annotation order for every seed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    a_wrong_b_correct = int(np.sum(~a_correct & b_correct))
    a_correct_b_wrong = int(np.sum(a_correct & ~b_correct))
    discordant = a_wrong_b_correct + a_correct_b_wrong
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(a_wrong_b_correct, a_correct_b_wrong)
        # scipy evaluates the exact binomial tail without overflowing for the
        # large discordant counts produced by the fractal-only comparison.
        p_value = binomtest(lower, discordant, p=0.5, alternative="two-sided").pvalue
    return {
        "a_wrong_b_correct": a_wrong_b_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "p_value": float(p_value),
    }


def paired_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                     iterations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, n, size=n)
        deltas[index] = np.mean(pred_b[sample] == y_true[sample]) - np.mean(pred_a[sample] == y_true[sample])
    return {
        "accuracy_delta": float(np.mean(pred_b == y_true) - np.mean(pred_a == y_true)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    true = np.array([row["true_label"] for row in rows], dtype=object)
    pred = np.array([row["predicted_label"] for row in rows], dtype=object)
    return true, pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--compare", help="Variant name to compare against the first variant")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics_by_variant: dict[str, list[dict]] = {}
    missing: list[str] = []
    for variant in args.variants:
        values = []
        for seed in args.seeds:
            directory = args.results_root / f"{variant}_seed_{seed}"
            metrics_path = directory / "metrics.json"
            if not metrics_path.exists():
                missing.append(str(metrics_path))
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            values.append({"seed": seed, **metrics})
        metrics_by_variant[variant] = values

    if missing:
        raise SystemExit("Missing metrics files:\n" + "\n".join(missing))

    metric_names = ("accuracy", "top5_accuracy", "macro_precision", "macro_recall", "macro_f1")
    summary = {"seeds": args.seeds, "variants": {}, "missing": missing}
    for variant, values in metrics_by_variant.items():
        summary["variants"][variant] = {
            "per_seed": [{"seed": item["seed"], **{name: item[name] for name in metric_names}} for item in values],
            "mean": {name: float(np.mean([item[name] for item in values])) for name in metric_names},
            "std": {name: float(np.std([item[name] for item in values], ddof=1)) if len(values) > 1 else 0.0 for name in metric_names},
            "samples": [item["samples"] for item in values],
        }

    if args.compare:
        primary = args.variants[0]
        if primary == args.compare:
            raise SystemExit("--compare must differ from the first variant")
        paired = {}
        for seed in args.seeds:
            primary_csv = args.results_root / f"{primary}_seed_{seed}" / "predictions.csv"
            compare_csv = args.results_root / f"{args.compare}_seed_{seed}" / "predictions.csv"
            true_a, pred_a = read_predictions(primary_csv)
            true_b, pred_b = read_predictions(compare_csv)
            if not np.array_equal(true_a, true_b) or len(pred_a) != len(pred_b):
                raise SystemExit(f"Prediction order mismatch for seed {seed}")
            paired[str(seed)] = {
                "bootstrap": paired_bootstrap(true_a, pred_a, pred_b, args.bootstrap_iterations, seed),
                "mcnemar_exact": mcnemar_exact(true_a, pred_a, pred_b),
            }
        summary["paired_comparison"] = {"primary": primary, "comparison": args.compare, "by_seed": paired}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
