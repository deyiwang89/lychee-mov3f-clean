"""Generate a compact multi-model controlled-robustness comparison."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "robustness_capacity" / "robustness_capacity_summary.csv"
OUT = ROOT.parent / "Submission_Archive" / "fig_rebuttal_robustness_capacity.png"
ORDER = [
    "clean", "brightness_low", "brightness_high", "gaussian_blur", "motion_blur",
    "occlusion_10", "occlusion_25", "scale_crop_1.2", "rotation_10",
    "background_clutter", "noncentered",
]
LABELS = {
    "clean": "Clean", "brightness_low": "Bright -", "brightness_high": "Bright +",
    "gaussian_blur": "Gaussian blur", "motion_blur": "Motion blur",
    "occlusion_10": "Occlusion 10%", "occlusion_25": "Occlusion 25%",
    "scale_crop_1.2": "Scale/crop", "rotation_10": "Rotation",
    "background_clutter": "Background", "noncentered": "Non-centered",
}
COLORS = {"GAP": "#4C78A8", "Wide-GAP": "#B07AA1", "GAP+GMP": "#F28E2B", "GAP+Fractal": "#59A14F"}


def main() -> None:
    values = {}
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["model"], {})[row["condition"]] = float(row["accuracy_percent"])
    models = list(COLORS)
    x = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(7.1, 3.8), constrained_layout=True)
    for model in models:
        ax.plot(x, [values[model][condition] for condition in ORDER], marker="o", markersize=3.2,
                linewidth=1.35, color=COLORS[model], label=model)
    ax.set_xticks(x, [LABELS[name] for name in ORDER], rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(65, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
