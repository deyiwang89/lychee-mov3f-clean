"""Rebuild a complete, hash-verified manifest from all experiment run records."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def prediction_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-completed", action="store_true")
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    output = args.output or root / "results" / "complete_artifact_manifest.json"
    runs = []
    failures = []
    for status_path in sorted((root / "weights").glob("**/run_status.json")):
        weight_dir = status_path.parent
        config_path = weight_dir / "run_config.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not config_path.exists():
            failures.append(f"Missing run_config.json: {relative(weight_dir, root)}")
            continue
        if args.require_completed and status.get("status") != "completed":
            failures.append(f"Run is not completed: {relative(weight_dir, root)}")
            continue

        artifacts = []
        for name, registered in sorted(status.get("artifacts", {}).items()):
            artifact_path = Path(registered["path"])
            if not artifact_path.is_absolute():
                artifact_path = root / artifact_path
            if not artifact_path.exists():
                failures.append(f"Missing {name}: {relative(artifact_path, root)}")
                continue
            actual_hash = sha256(artifact_path)
            recorded_hash = registered.get("sha256", "").lower()
            matches = actual_hash.lower() == recorded_hash
            if not matches:
                failures.append(f"SHA-256 mismatch for {name}: {relative(artifact_path, root)}")
            artifact = {
                "name": name,
                "path": relative(artifact_path, root),
                "sha256": actual_hash,
                "recorded_sha256": recorded_hash,
                "hash_matches": matches,
                "size_bytes": artifact_path.stat().st_size,
            }
            if name == "predictions" and artifact_path.suffix.lower() == ".csv":
                artifact["rows"] = prediction_rows(artifact_path)
            artifacts.append(artifact)

        config = json.loads(config_path.read_text(encoding="utf-8"))
        runs.append({
            "run_directory": relative(weight_dir, root),
            "variant": status.get("variant"),
            "backbone": config.get("backbone"),
            "seed": status.get("seed"),
            "status": status.get("status"),
            "completed_at": status.get("completed_at"),
            "run_config_path": relative(config_path, root),
            "run_config_sha256": sha256(config_path),
            "run_status_path": relative(status_path, root),
            "run_status_sha256": sha256(status_path),
            "train_samples": config.get("num_train"),
            "validation_samples": config.get("num_val"),
            "artifacts": artifacts,
        })

    report = {
        "purpose": "Complete manifest rebuilt from per-run records after parallel runners overwrote partial launcher manifests.",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "experiment_root": str(root),
        "run_count": len(runs),
        "all_checks_passed": not failures,
        "failures": failures,
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "run_count": len(runs),
        "all_checks_passed": not failures,
        "failures": failures,
    }, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
