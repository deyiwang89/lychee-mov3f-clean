"""Create reproducible stratified development/test splits from archived images.

The existing datasets/train and datasets/test folders are treated as one source
pool. Twenty percent is reserved as the development pool and split again into
training and validation subsets; the 80% test set remains untouched until final
evaluation. The original annotation files are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(root: Path, classes: list[str]) -> tuple[dict[str, list[str]], dict]:
    candidates = []
    for split_priority, split in enumerate(("train", "test")):
        for path in sorted((root / "datasets" / split).rglob("*.jpg")):
            if path.parent.name in classes:
                candidates.append({
                    "path": path,
                    "relative_path": path.relative_to(root).as_posix(),
                    "class_name": path.parent.name,
                    "split_priority": split_priority,
                    "sha256": sha256(path),
                })

    by_hash = defaultdict(list)
    for item in candidates:
        by_hash[item["sha256"]].append(item)

    images: dict[str, list[str]] = defaultdict(list)
    duplicate_groups = []
    for digest, group in sorted(by_hash.items()):
        labels = {item["class_name"] for item in group}
        if len(labels) != 1:
            raise SystemExit(f"Identical image bytes have conflicting labels: {group}")
        canonical = min(group, key=lambda item: (item["split_priority"], item["relative_path"]))
        images[canonical["class_name"]].append(canonical["relative_path"])
        if len(group) > 1:
            duplicate_groups.append({
                "sha256": digest,
                "class_name": canonical["class_name"],
                "canonical_path": canonical["relative_path"],
                "excluded_paths": [
                    item["relative_path"] for item in group if item is not canonical
                ],
            })
    audit = {
        "source_file_count": len(candidates),
        "unique_image_count": len(by_hash),
        "excluded_duplicate_count": len(candidates) - len(by_hash),
        "duplicate_groups": duplicate_groups,
    }
    return {key: sorted(value) for key, value in images.items()}, audit


def annotation_line(label: int, image_path: str) -> str:
    return f"{label};{image_path}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--classes", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--train-ratio", type=float, default=0.20)
    parser.add_argument("--val-ratio-within-development", type=float, default=0.20)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    classes_path = args.classes or root / "data" / "classes.txt"
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    images, duplicate_audit = collect_images(root, classes)
    missing = [name for name in classes if not images.get(name)]
    if missing:
        raise SystemExit(f"Missing image classes: {missing}")
    if not 0 < args.train_ratio < 1:
        raise SystemExit("--train-ratio must be between 0 and 1")
    if not 0 < args.val_ratio_within_development < 1:
        raise SystemExit("--val-ratio-within-development must be between 0 and 1")

    output_root = args.output_root or root / "data" / "manifests"
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        **duplicate_audit,
        "development_ratio": args.train_ratio,
        "val_ratio_within_development": args.val_ratio_within_development,
        "classes": classes,
        "seeds": {},
    }

    for seed in args.seeds:
        rng = random.Random(seed)
        development, train, val, test = [], [], [], []
        class_summary = {}
        for label, class_name in enumerate(classes):
            items = list(images[class_name])
            rng.shuffle(items)
            development_count = max(2, round(len(items) * args.train_ratio))
            development_items, test_items = items[:development_count], items[development_count:]
            val_count = max(1, round(development_count * args.val_ratio_within_development))
            val_items = development_items[:val_count]
            train_items = development_items[val_count:]
            development.extend(annotation_line(label, item) for item in development_items)
            train.extend(annotation_line(label, item) for item in train_items)
            val.extend(annotation_line(label, item) for item in val_items)
            test.extend(annotation_line(label, item) for item in test_items)
            class_summary[class_name] = {
                "total": len(items),
                "development": len(development_items),
                "train": len(train_items),
                "val": len(val_items),
                "test": len(test_items),
            }
        rng.shuffle(development)
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "development.txt").write_text("".join(development), encoding="utf-8")
        (seed_dir / "train.txt").write_text("".join(train), encoding="utf-8")
        (seed_dir / "val.txt").write_text("".join(val), encoding="utf-8")
        (seed_dir / "test.txt").write_text("".join(test), encoding="utf-8")
        metadata = {
            "seed": seed,
            "development_ratio": args.train_ratio,
            "val_ratio_within_development": args.val_ratio_within_development,
            "source_file_count": duplicate_audit["source_file_count"],
            "unique_image_count": len(development) + len(test),
            "excluded_duplicate_count": duplicate_audit["excluded_duplicate_count"],
            "development_count": len(development),
            "train_count": len(train),
            "val_count": len(val),
            "test_count": len(test),
            "class_counts": class_summary,
            "source_folders": ["datasets/train", "datasets/test"],
        }
        (seed_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["seeds"][str(seed)] = metadata
        print(
            f"seed={seed}: development={len(development)}, train={len(train)}, "
            f"val={len(val)}, test={len(test)}"
        )

    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Split files written to: {output_root}")


if __name__ == "__main__":
    main()
