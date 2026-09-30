"""Create a deterministic flat calibration set from seed-43 training images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "seed_43" / "train.txt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT  / "results" / "real_scene_23_sliding_window" / "k230_calibration")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in args.split.read_text(encoding="utf-8").splitlines():
        label, relative_path = line.split(";", 1)
        source = (PROJECT_ROOT / relative_path.strip().split()[0]).resolve()
        if "datasets\\train" not in str(source) and "datasets/train" not in str(source):
            continue
        rows.append((int(label), source))
        if len(rows) >= args.count:
            break
    if len(rows) < args.count:
        raise SystemExit(f"Only {len(rows)} training images found, need {args.count}")
    manifest = []
    for index, (label, source) in enumerate(rows):
        target = output / f"calib_{index:02d}_{source.name}"
        shutil.copy2(source, target)
        manifest.append({"index": index, "label_index": label, "source": str(source), "target": str(target), "sha256": sha256(target)})
    (output / "calibration_manifest.csv").open("w", encoding="utf-8", newline="").close()
    with (output / "calibration_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"prepared={len(manifest)} output={output}")


if __name__ == "__main__":
    main()
