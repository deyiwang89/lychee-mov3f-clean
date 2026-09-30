"""Aggregate representative-seed controlled robustness evaluations."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "robustness_capacity"
OUTPUT = ROOT / "results" / "robustness_capacity"
MODELS = {
    "gap_seed_43.json": "GAP",
    "wide_gap_seed_43.json": "Wide-GAP",
    "gap_gmp_seed_43.json": "GAP+GMP",
    "fractal_seed_43.json": "GAP+Fractal",
}


def main() -> None:
    rows = []
    for filename, model in MODELS.items():
        path = INPUT / filename
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for condition, result in payload["conditions"].items():
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "accuracy_percent": 100.0 * result["accuracy"],
                    "change_from_clean_pp": 100.0 * result["accuracy_change_from_clean"],
                    "errors": result["errors"],
                    "samples": payload["samples"],
                }
            )

    csv_path = OUTPUT / "robustness_capacity_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_condition = {}
    for row in rows:
        by_condition.setdefault(row["condition"], {})[row["model"]] = row
    summary = {
        "protocol": "representative seed 43; deterministic controlled perturbations",
        "samples": rows[0]["samples"],
        "models": list(MODELS.values()),
        "conditions": by_condition,
    }
    (OUTPUT / "robustness_capacity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(csv_path)


if __name__ == "__main__":
    main()
