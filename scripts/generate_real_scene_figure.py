"""Create a compact four-case figure from the real-scene diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.root / "per_image_summary.csv"
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for status in ("high_confidence_candidate", "ambiguous_multi_window_prediction"):
        selected.extend([row for row in rows if row["review_status"] == status][:2])
    if len(selected) < 4:
        raise SystemExit("Need at least four diagnostic cases")
    figure, axes = plt.subplots(2, 2, figsize=(8, 10), dpi=220)
    for index, (axis, row) in enumerate(zip(axes.flat, selected)):
        path = args.root / "diagnostic_figures" / f"{row['image_id']}.png"
        axis.imshow(Image.open(path))
        axis.axis("off")
        axis.set_title(f"({chr(97 + index)}) {row['review_status'].replace('_', ' ')}", fontsize=8)
    figure.tight_layout(pad=0.4)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
