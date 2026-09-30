"""Launch the reproducible rebuttal training matrix.

The script uses the active Python interpreter rather than a machine-specific
Anaconda path. It is intentionally explicit about split files and variants.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "gap": "mobilenetv3_gap",
    "gap_matched": "mobilenetv3_gap_matched",
    "wide_gap": "mobilenetv3_gap_wide",
    "gap_gmp": "mobilenetv3_gap_gmp",
    "fractal": "mobilenetv3_fractal",
    "fractal_only": "mobilenetv3_fractal_only",
    "gated": "mobilenetv3_gated_fractal",
    "penultimate": "mobilenetv3_fractal_penultimate",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=["gap", "fractal", "fractal_only", "wide_gap", "gap_gmp"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--box-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    split_root = root / "data" / "manifests"
    experiment_root = root 
    output_root = args.output_root or experiment_root / "weights"
    results_root = args.results_root or experiment_root / "results" / "core"
    output_root = output_root if output_root.is_absolute() else root / output_root
    results_root = results_root if results_root.is_absolute() else root / results_root
    output_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    manifest = []

    for variant in args.variants:
        for seed in args.seeds:
            split_dir = split_root / f"seed_{seed}"
            train_file = split_dir / "train.txt"
            val_file = split_dir / "val.txt"
            test_file = split_dir / "test.txt"
            if not train_file.exists() or not val_file.exists() or not test_file.exists():
                raise SystemExit(f"Missing split files for seed {seed}; run make_splits.py first")
            save_dir = output_root / f"{variant}_seed_{seed}"
            result_dir = results_root / f"{variant}_seed_{seed}"
            status_path = save_dir / "run_status.json"
            command = [
                sys.executable,
                str(root / "train.py"),
                "--seed", str(seed),
                "--save_dir", str(save_dir),
                "--backbone", VARIANTS[variant],
                "--train-annotation", str(train_file),
                "--val-annotation", str(val_file),
                "--unfreeze-epoch", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--val-batch-size", str(args.val_batch_size),
                "--no-freeze",
                "--fractal-box-sizes", *map(str, args.box_sizes),
                "--fractal-threshold", str(args.threshold),
            ]
            if args.cpu:
                command.append("--cpu")
            if args.no_pretrained:
                command.append("--no-pretrained")
            entry = {"variant": variant, "seed": seed, "command": command, "output": str(save_dir), "results": str(result_dir)}
            manifest.append(entry)
            if status_path.exists() and not args.force:
                previous_status = json.loads(status_path.read_text(encoding="utf-8"))
                if previous_status.get("status") == "completed":
                    print(f"SKIP completed: {save_dir}")
                    continue
            print("RUN:", " ".join(f'"{part}"' if " " in part else part for part in command))
            if not args.dry_run:
                save_dir.mkdir(parents=True, exist_ok=True)
                status_path.write_text(json.dumps({
                    "status": "running",
                    "started_at": dt.datetime.now().astimezone().isoformat(),
                    "variant": variant,
                    "seed": seed,
                }, indent=2), encoding="utf-8")
                subprocess.run(command, cwd=root, check=True)
                checkpoint = save_dir / "best_epoch_weights.pth"
                if not checkpoint.exists():
                    raise RuntimeError(f"Training completed without best checkpoint: {checkpoint}")
                if not args.skip_evaluation:
                    evaluation_command = [
                        sys.executable,
                        str(root  / "scripts" / "evaluate_model.py"),
                        "--project-root", str(root),
                        "--weights", str(checkpoint),
                        "--backbone", VARIANTS[variant],
                        "--seed", str(seed),
                        "--test-annotation", str(test_file),
                        "--output-dir", str(result_dir),
                        "--fractal-box-sizes", *map(str, args.box_sizes),
                        "--fractal-threshold", str(args.threshold),
                    ]
                    subprocess.run(evaluation_command, cwd=root, check=True)
                artifacts = {"best_checkpoint": str(checkpoint)}
                for label, path in (
                    ("best_checkpoint", checkpoint),
                    ("run_config", save_dir / "run_config.json"),
                    ("metrics", result_dir / "metrics.json"),
                    ("predictions", result_dir / "predictions.csv"),
                ):
                    if path.exists():
                        artifacts[label] = {
                            "path": str(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                status_path.write_text(json.dumps({
                    "status": "completed",
                    "completed_at": dt.datetime.now().astimezone().isoformat(),
                    "variant": variant,
                    "seed": seed,
                    "artifacts": artifacts,
                }, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = output_root / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Training manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
