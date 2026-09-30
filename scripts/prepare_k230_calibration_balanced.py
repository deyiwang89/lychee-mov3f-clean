"""Create a deterministic class-balanced K230 calibration set."""

from __future__ import annotations

import csv
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT = PROJECT_ROOT / "data" / "manifests" / "seed_43" / "train.txt"
OUTPUT = PROJECT_ROOT  / "results" / "real_scene_23_sliding_window" / "k230_calibration_balanced100"
PER_CLASS = 10


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for line in SPLIT.read_text(encoding="utf-8").splitlines():
        label, relative_path = line.split(";", 1)
        source = (PROJECT_ROOT / relative_path.strip().split()[0]).resolve()
        grouped[int(label)].append(source)
    if sorted(grouped) != list(range(10)):
        raise SystemExit(f"Expected labels 0-9, got {sorted(grouped)}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    index = 0
    for label in range(10):
        sources = sorted(grouped[label], key=lambda path: str(path).lower())[:PER_CLASS]
        if len(sources) != PER_CLASS:
            raise SystemExit(f"Label {label} has only {len(sources)} calibration candidates")
        for source in sources:
            target = OUTPUT / f"calib_{index:03d}_class{label}_{source.name}"
            shutil.copy2(source, target)
            rows.append({
                "index": index,
                "label_index": label,
                "source": str(source),
                "target": str(target),
                "sha256": digest(target),
            })
            index += 1

    with (OUTPUT / "calibration_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"prepared={len(rows)} output={OUTPUT}")


if __name__ == "__main__":
    main()
