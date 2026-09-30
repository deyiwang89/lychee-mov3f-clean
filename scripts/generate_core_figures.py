"""Generate publication-ready core ablation, confusion, and failure figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_LABELS = {
    "gap": "GAP",
    "wide_gap": "Wide-GAP",
    "gap_gmp": "GAP + GMP",
    "fractal": "GAP + fractal",
    "fractal_only": "Fractal only",
}
COLORS = ["#4C78A8", "#B07AA1", "#F28E2B", "#59A14F", "#E15759"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-summary", type=Path, required=True, help="Capacity-controlled core summary containing all five head variants.")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--classes", type=Path, default=PROJECT_ROOT / "data" / "classes.txt")
    parser.add_argument("--include-supplementary", action="store_true", help="Also create confusion and failure figures when per-image records are available.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.core_summary.read_text(encoding="utf-8"))
    variants = ["gap", "wide_gap", "gap_gmp", "fractal", "fractal_only"]
    means = [100 * summary["variants"][name]["mean"]["accuracy"] for name in variants]
    stds = [100 * summary["variants"][name]["sample_std"]["accuracy"] for name in variants]
    params = [summary["variants"][name]["parameter_count"] / 1e6 for name in variants]

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.45), constrained_layout=True)
    x = np.arange(len(variants))
    # A horizontal accuracy panel keeps long head names readable and leaves
    # enough clearance for the value labels and error bars.
    y = np.arange(len(variants))
    axes[0].barh(y, means, xerr=stds, capsize=3, color=COLORS, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Top-1 accuracy (%)")
    axes[0].set_yticks(y, [VARIANT_LABELS[name] for name in variants], fontsize=7)
    axes[0].invert_yaxis()
    axes[0].text(0.02, 0.98, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].set_xlim(60, 100.5)
    for index, value in enumerate(means):
        if value >= 90:
            axes[0].text(value - 0.18, index, f"{value:.2f}", ha="right", va="center", fontsize=7.5, color="white", fontweight="bold")
        else:
            axes[0].text(value + stds[index] + 0.45, index, f"{value:.2f}", ha="left", va="center", fontsize=7.5)

    axes[1].bar(x, params, color=COLORS, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Parameters (M)")
    axes[1].set_xticks(x, [VARIANT_LABELS[name] for name in variants], rotation=22, ha="right", fontsize=7)
    axes[1].text(0.02, 0.98, "(b)", transform=axes[1].transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_ylim(0, max(params) * 1.22)
    for index, value in enumerate(params):
        axes[1].text(index, value + 0.03, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(args.output_dir / "fig_rebuttal_core_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if not args.include_supplementary:
        print(json.dumps({"figures": ["fig_rebuttal_core_ablation.png"]}, indent=2))
        return
    if args.results_root is None:
        raise SystemExit("--results-root is required with --include-supplementary")

    selected_seed = summary.get("representative_fractal_seed", {"seed": 43})["seed"]
    result_dir = args.results_root / f"fractal_seed_{selected_seed}"
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    classes = [line.strip() for line in args.classes.read_text(encoding="utf-8").splitlines() if line.strip()]
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) != 0)
    short_labels = [name.replace("_Leaf", "").replace("_", " ") for name in classes]
    fig, ax = plt.subplots(figsize=(6.3, 5.4), constrained_layout=True)
    image = ax.imshow(normalized, cmap="YlGn", vmin=0, vmax=1)
    for row in range(len(classes)):
        for col in range(len(classes)):
            if matrix[row, col] > 0:
                ax.text(col, row, f"{int(matrix[row, col])}\n{normalized[row, col]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if normalized[row, col] > 0.55 else "black")
    ax.set_xticks(range(len(classes)), short_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), short_labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized proportion")
    fig.savefig(args.output_dir / "fig_rebuttal_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with (result_dir / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        errors = [row for row in csv.DictReader(handle) if row["status"] == "Incorrect"]
    errors.sort(key=lambda row: float(row["confidence"]), reverse=True)
    selected_errors = errors[:12]
    fig, axes = plt.subplots(3, 4, figsize=(7.1, 5.5), constrained_layout=True)
    for ax, row in zip(axes.flat, selected_errors):
        ax.imshow(Image.open(row["image_path"]).convert("RGB"))
        true_label = row["true_label"].replace("_Leaf", "").replace("_", " ")
        predicted_label = row["predicted_label"].replace("_Leaf", "").replace("_", " ")
        ax.set_title(f"{true_label} -> {predicted_label}\np={float(row['confidence']):.3f}", fontsize=7)
        ax.axis("off")
    for ax in axes.flat[len(selected_errors):]:
        ax.axis("off")
    fig.savefig(args.output_dir / "fig_rebuttal_failure_examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({
        "representative_seed": selected_seed,
        "figures": [
            "fig_rebuttal_core_ablation.png",
            "fig_rebuttal_confusion_matrix.png",
            "fig_rebuttal_failure_examples.png",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
