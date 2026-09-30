"""Generate Fig. 9 from the public secondary-domain summary JSON."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=root / "results" / "secondary_transfer" / "secondary_1409_final_summary.json")
    parser.add_argument("--output", type=Path, default=root / "figures" / "fig_rebuttal_secondary_1409.png")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    models = [("gap", "GAP", "#4C78A8"), ("gap_gmp", "GAP+GMP", "#F28E2B"), ("fractal", "GAP+Fractal", "#59A14F")]
    scopes = [("source_group", "Source-group\nrepresentatives"), ("all_1409", "All 1,409\nmerged rows")]
    x = np.arange(len(scopes))
    width = 0.23
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    for index, (key, label, color) in enumerate(models):
        means = [100 * summary[scope]["variants"][key]["accuracy_mean"] for scope, _ in scopes]
        stds = [100 * summary[scope]["variants"][key]["accuracy_std"] for scope, _ in scopes]
        bars = ax.bar(x + (index - 1) * width, means, width, yerr=stds, capsize=3, label=label, color=color, edgecolor="black", linewidth=0.4)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [label for _, label in scopes])
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(90, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.01, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.text(0.51, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), borderaxespad=0.0, fontsize=8)
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.12, right=0.98)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
