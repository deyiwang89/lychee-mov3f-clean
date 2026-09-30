"""Run the pre-specified fractal sensitivity configurations sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "scales_12": {"variant": "fractal", "box_sizes": [1, 2], "threshold": 0.5},
    "threshold_03": {"variant": "fractal", "box_sizes": [1, 2, 4], "threshold": 0.3},
    "threshold_07": {"variant": "fractal", "box_sizes": [1, 2, 4], "threshold": 0.7},
    "gated": {"variant": "gated", "box_sizes": [1, 2, 4], "threshold": 0.5},
    "penultimate": {"variant": "penultimate", "box_sizes": [1, 2, 4], "threshold": 0.5},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--settings", nargs="+", choices=sorted(SETTINGS), default=list(SETTINGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    experiment_root = root 
    runner = experiment_root / "scripts" / "run_training_matrix.py"
    manifest = []
    for name in args.settings:
        setting = SETTINGS[name]
        command = [
            sys.executable, str(runner),
            "--project-root", str(root),
            "--seeds", *map(str, args.seeds),
            "--variants", setting["variant"],
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--val-batch-size", str(args.val_batch_size),
            "--box-sizes", *map(str, setting["box_sizes"]),
            "--threshold", str(setting["threshold"]),
            "--output-root", str(experiment_root / "weights" / "sensitivity" / name),
            "--results-root", str(experiment_root / "results" / "sensitivity" / name),
        ]
        if args.force:
            command.append("--force")
        manifest.append({"name": name, "setting": setting, "seeds": args.seeds, "command": command})
        print(f"SENSITIVITY RUN: {name}", flush=True)
        subprocess.run(command, cwd=root, check=True)
    manifest_path = experiment_root / "weights" / "sensitivity" / "sensitivity_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sensitivity manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
