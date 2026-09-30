"""Repair seed metadata omitted by an already-running core matrix process.

The core matrix was launched before ``--seed`` was added to its evaluation
command.  Run this only after the matrix finishes.  It changes metadata, not
predictions or measured metrics, and records every before/after file hash.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


VARIANTS = ("gap", "fractal", "fractal_only", "gap_matched")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()

    repaired_at = dt.datetime.now().astimezone().isoformat()
    report = {
        "purpose": "Backfill omitted seed metadata; metric values and predictions are unchanged.",
        "repaired_at": repaired_at,
        "runs": [],
    }
    for variant in args.variants:
        for seed in args.seeds:
            run_name = f"{variant}_seed_{seed}"
            metrics_path = args.results_root / run_name / "metrics.json"
            config_path = args.weights_root / run_name / "run_config.json"
            status_path = args.weights_root / run_name / "run_status.json"
            for path in (metrics_path, config_path, status_path):
                if not path.exists():
                    raise SystemExit(f"Missing completed artifact: {path}")

            config = json.loads(config_path.read_text(encoding="utf-8"))
            status = json.loads(status_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if status.get("status") != "completed":
                raise SystemExit(f"Run is not completed: {run_name}")
            if config.get("seed") != seed or status.get("seed") != seed:
                raise SystemExit(f"Seed mismatch outside metrics.json: {run_name}")
            old_seed = metrics.get("seed")
            if old_seed not in (None, seed):
                raise SystemExit(f"Unexpected metrics seed for {run_name}: {old_seed}")

            metrics_hash_before = sha256(metrics_path)
            if old_seed is None:
                metrics["seed"] = seed
                write_json(metrics_path, metrics)
            metrics_hash_after = sha256(metrics_path)

            artifacts = status.setdefault("artifacts", {})
            metrics_artifact = artifacts.get("metrics")
            if not isinstance(metrics_artifact, dict):
                raise SystemExit(f"Missing metrics artifact registration: {run_name}")
            metrics_artifact["sha256"] = metrics_hash_after
            status["metadata_repaired_at"] = repaired_at
            status["metadata_repair"] = "Backfilled metrics.seed from the verified run_config and directory name."
            write_json(status_path, status)

            report["runs"].append({
                "run": run_name,
                "seed_before": old_seed,
                "seed_after": seed,
                "metrics_sha256_before": metrics_hash_before,
                "metrics_sha256_after": metrics_hash_after,
                "status_sha256_after": sha256(status_path),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({"repaired_runs": len(report["runs"]), "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
