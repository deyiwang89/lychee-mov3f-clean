"""Copy every image in a specified audited split into a flat calibration dir."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, line in enumerate(args.split.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        label, relative_path = line.split(";", 1)
        source = (PROJECT_ROOT / relative_path.strip().split()[0]).resolve()
        target = args.output / f"calib_{index:04d}_class{int(label)}_{source.name}"
        shutil.copy2(source, target)
        rows.append({
            "index": index,
            "label_index": int(label),
            "source": str(source),
            "target": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        })
    with (args.output / "calibration_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"prepared={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
