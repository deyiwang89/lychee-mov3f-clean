"""Audit exact and perceptually near-duplicate images without modifying data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
from skimage.metrics import structural_similarity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def ssim(root: Path, left_path: str, right_path: str) -> float:
    left = cv2.imread(str(root / left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(root / right_path), cv2.IMREAD_GRAYSCALE)
    left = cv2.resize(left, (256, 256), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (256, 256), interpolation=cv2.INTER_AREA)
    return float(structural_similarity(left, right, data_range=255))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-hamming", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    paths = sorted([
        *root.joinpath("datasets", "train").rglob("*.jpg"),
        *root.joinpath("datasets", "test").rglob("*.jpg"),
    ])
    records = [{
        "path": path.relative_to(root).as_posix(),
        "class_name": path.parent.name,
        "sha256": sha256(path),
        "dhash": dhash(path),
    } for path in paths]
    exact_pairs = []
    near_pairs = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1:]:
            distance = bin(left["dhash"] ^ right["dhash"]).count("1")
            if left["sha256"] == right["sha256"]:
                exact_pairs.append({"left": left["path"], "right": right["path"], "class_name": left["class_name"]})
            elif distance <= args.max_hamming:
                near_pairs.append({
                    "left": left["path"], "right": right["path"],
                    "left_class": left["class_name"], "right_class": right["class_name"],
                    "hamming_distance": distance,
                    "ssim_256_gray": ssim(root, left["path"], right["path"]),
                })
    report = {
        "source_file_count": len(records),
        "unique_sha256_count": len({record["sha256"] for record in records}),
        "dhash_max_hamming": args.max_hamming,
        "exact_duplicate_pairs": exact_pairs,
        "near_duplicate_pairs": near_pairs,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "source_file_count": report["source_file_count"],
        "unique_sha256_count": report["unique_sha256_count"],
        "exact_duplicate_pairs": len(exact_pairs),
        "near_duplicate_pairs": len(near_pairs),
    }, indent=2))


if __name__ == "__main__":
    main()
