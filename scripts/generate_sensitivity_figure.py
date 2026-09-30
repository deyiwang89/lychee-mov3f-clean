"""Generate sensitivity and controlled-robustness figures from frozen JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SETTING_LABELS = {
    "main": "Main",
    "scales_12": "Scales {1,2}",
    "threshold_03": "Threshold 0.3",
    "threshold_07": "Threshold 0.7",
    "gated": "Gated fusion",
    "penultimate": "Penultimate",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity-summary", type=Path, required=True)
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sensitivity = json.loads(args.sensitivity_summary.read_text(encoding="utf-8"))
    rows = sensitivity["settings"]
    labels = [SETTING_LABELS[row["setting"]] for row in rows]
    accuracies = [100 * row["accuracy"] for row in rows]
    val_losses = [row["best_val_loss"] for row in rows]
    colors = ["#59A14F" if row["setting"] == "main" else "#4C78A8" for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), constrained_layout=True)
    y = np.arange(len(rows))
    axes[0].barh(y, accuracies, color=colors)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Top-1 accuracy on the test set (%)")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].set_xlim(max(0, min(accuracies) - 2), min(100, max(accuracies) + 1))
    for index, value in enumerate(accuracies):
        axes[0].text(value + 0.08, index, f"{value:.2f}", va="center", fontsize=8)
    axes[1].barh(y, val_losses, color=colors)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Minimum validation cross-entropy")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].set_xlim(0, max(val_losses) * 1.18)
    for index, value in enumerate(val_losses):
        axes[1].text(value, index, f" {value:.3f}", va="center", fontsize=8)
    fig.savefig(args.output_dir / "fig_rebuttal_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    robustness = json.loads(args.robustness.read_text(encoding="utf-8"))
    conditions = robustness["conditions"]
    names = list(conditions)
    robust_labels = [name.replace("_", " ") for name in names]
    values = [100 * conditions[name]["accuracy"] for name in names]
    clean = values[names.index("clean")]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.1, 4.4), constrained_layout=True)
    ax.axvline(clean, color="#555555", linewidth=1, linestyle="--", label=f"Clean: {clean:.2f}%")
    ax.hlines(y, 0, values, color="#B8B8B8", linewidth=1)
    point_colors = ["#59A14F" if name == "clean" else "#E15759" if values[index] < clean - 10 else "#4C78A8" for index, name in enumerate(names)]
    ax.scatter(values, y, color=point_colors, s=32, zorder=3)
    ax.set_yticks(y, robust_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Top-1 accuracy (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.01))
    for index, value in enumerate(values):
        if value >= 93:
            ax.text(value - 0.7, index, f"{value:.2f}", ha="right", va="center", fontsize=8)
        else:
            ax.text(value + 0.7, index, f"{value:.2f}", ha="left", va="center", fontsize=8)
    fig.savefig(args.output_dir / "fig_rebuttal_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"figures": ["fig_rebuttal_sensitivity.png", "fig_rebuttal_robustness.png"]}, indent=2))


if __name__ == "__main__":
    main()
