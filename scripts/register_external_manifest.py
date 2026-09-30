"""Register independent external images without modifying training data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=[
        "Felt_Leaf", "Deficiency_Leaf", "Leaf_Gall", "Fungal_Leaf_Spot", "Curl_Leaf"
    ])
    args = parser.parse_args()
    external_root = args.external_root.resolve()
    project_root = args.project_root.resolve()
    records = []
    for image in sorted(external_root.rglob("*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = image.parent.name
        if label not in args.classes:
            raise SystemExit(f"Unexpected external class directory: {label}")
        records.append({
            "image_path": str(image),
            "relative_path": image.relative_to(external_root).as_posix(),
            "label": label,
            "sha256": sha256(image),
            "collection_date": "",
            "location": "",
            "device": "",
            "lighting": "",
            "background": "",
            "occlusion": "",
            "annotator": "",
            "reviewer": "",
        })
    if not records:
        raise SystemExit("No external images found")
    hashes = [record["sha256"] for record in records]
    duplicate_hashes = sorted({value for value in hashes if hashes.count(value) > 1})
    dataset_hashes = set()
    for split in ("train", "test"):
        for image in (project_root / "datasets" / split).rglob("*.jpg"):
            dataset_hashes.add(sha256(image))
    overlap = sorted(set(hashes) & dataset_hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "external_root": str(external_root),
        "total_images": len(records),
        "class_counts": {label: sum(record["label"] == label for record in records) for label in args.classes},
        "duplicate_hashes": duplicate_hashes,
        "overlap_with_training_pool": overlap,
        "metadata_fields_requiring_completion": ["collection_date", "location", "device", "lighting", "background", "annotator", "reviewer"],
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
