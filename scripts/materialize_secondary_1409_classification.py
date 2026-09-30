"""Materialize a derived classification copy from the audited 1409 manifest."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image


def crop_normalized(image: Image.Image, union: tuple[float, float, float, float], margin: float) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = union
    box_width, box_height = right - left, bottom - top
    left = max(0.0, left - margin * box_width)
    top = max(0.0, top - margin * box_height)
    right = min(1.0, right + margin * box_width)
    bottom = min(1.0, bottom + margin * box_height)
    return image.crop((round(left * width), round(top * height), round(right * width), round(bottom * height)))


def materialize(manifest_path: Path, output_dir: Path, margin: float) -> None:
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for index, row in enumerate(rows):
        class_dir = output_dir / row["label"]
        class_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(row["image_path"]).convert("RGB")
        crop = crop_normalized(image, tuple(json.loads(row["bbox_union"])), margin)
        filename = f"{index:04d}__{row['label']}__{row['source_id']}__{row['sha256'][:12]}.jpg"
        target = class_dir / filename
        crop.save(target, quality=95)
        output_rows.append({**row, "derived_image_path": str(target.resolve()), "crop_margin": str(margin)})
    with (output_dir / "classification_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    (output_dir / "README.txt").write_text(
        "Derived classification copy. Raw YOLO files are preserved elsewhere. "
        "Each image is cropped from the union of its YOLO boxes with a fixed margin. "
        "The original train/valid/test split is metadata only; this directory is a merged secondary-domain archive.\n",
        encoding="utf-8",
    )
    print(f"materialized={len(rows)} output={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.10)
    args = parser.parse_args()
    materialize(args.manifest, args.output_dir, args.margin)


if __name__ == "__main__":
    main()
