"""Evaluate the original-view or lossless-crop release using the frozen protocol."""

import argparse
import csv
import json
from pathlib import Path

from evaluate_secondary_1409 import evaluate_checkpoint


def package_rows(root, kind, all_rows=False):
    name = "manifest_all.csv" if all_rows else "representatives_951.csv"
    with (root / name).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = (root / row[f"{kind}_path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Invalid package image: {path}")
        row["image_path"] = str(path)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("original", "crop"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--backbone", default="mobilenetv3_fractal")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--box-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    rows = package_rows(args.data_root, args.kind, args.all)
    # "full" here means already-cropped PNG input, not a different evaluation protocol.
    mode = "bbox" if args.kind == "original" else "full"
    evaluate_checkpoint(args.weights, args.backbone, rows, args.output_dir, mode,
                        args.box_sizes, args.threshold)
    (args.output_dir / "input_protocol.json").write_text(
        json.dumps(dict(package_kind=args.kind, crop_margin=0.1,
                        already_cropped=args.kind == "crop", samples=len(rows)), indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
