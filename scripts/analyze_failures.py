"""Summarize class-wise errors and confusion pairs from a prediction CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-errors", type=int, default=20)
    args = parser.parse_args()
    with args.predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Accept both the new evaluator schema and the archived CSV schema.
    if rows and "true_label" not in rows[0]:
        rows = [
            {
                **row,
                "true_label": row.get("True Label", ""),
                "predicted_label": row.get("Predicted Label", ""),
                "confidence": row.get("Confidence", row.get("confidence", "0")),
            }
            for row in rows
        ]
    by_class = defaultdict(lambda: {"samples": 0, "errors": 0})
    pairs = Counter()
    errors = []
    for row in rows:
        true_label, predicted = row["true_label"], row["predicted_label"]
        by_class[true_label]["samples"] += 1
        if true_label != predicted:
            by_class[true_label]["errors"] += 1
            pairs[(true_label, predicted)] += 1
            errors.append(row)
    errors.sort(key=lambda row: float(row.get("confidence", 0.0)), reverse=True)
    report = {
        "samples": len(rows),
        "errors": len(errors),
        "class_summary": {
            label: {
                **values,
                "accuracy": (values["samples"] - values["errors"]) / values["samples"] if values["samples"] else 0.0,
                "recall": (values["samples"] - values["errors"]) / values["samples"] if values["samples"] else 0.0,
            }
            for label, values in sorted(by_class.items())
        },
        "confusion_pairs": [
            {"true": true, "predicted": predicted, "count": count}
            for (true, predicted), count in pairs.most_common()
        ],
        "high_confidence_errors": errors[: args.top_errors],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
