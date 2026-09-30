"""Pack the audited seed-43 test set into one RGB224 binary for K230."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def resize_center_crop(image: Image.Image, size: int = 224) -> Image.Image:
    width, height = image.size
    if not ((width <= height and width == size) or (height <= width and height == size)):
        if width < height:
            new_width = size
            new_height = int(size * height / width)
        else:
            new_height = size
            new_width = int(size * width / height)
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    width, height = image.size
    top = int(round((height - size) / 2.0))
    left = int(round((width - size) / 2.0))
    return image.crop((left, top, left + size, top + size))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-count", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with args.predictions.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise SystemExit("Prediction CSV is empty")

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    bundle_path = args.output / "seed43_test_rgb224.bin"
    manifest_path = args.output / "manifest.csv"
    digest = hashlib.sha256()
    manifest_rows = []
    verified = 0
    verify_matches = 0

    with bundle_path.open("wb") as bundle:
        for index, row in enumerate(source_rows):
            path = Path(row["image_path"])
            if not path.exists():
                raise FileNotFoundError(path)
            image = resize_center_crop(Image.open(path).convert("RGB"))
            array = np.asarray(image, dtype=np.uint8)
            if array.shape != (224, 224, 3):
                raise RuntimeError(f"Unexpected shape {array.shape} for {path}")
            payload = array.tobytes(order="C")
            bundle.write(payload)
            digest.update(payload)

            onnx_label = ""
            onnx_match = ""
            if index < args.verify_count:
                tensor = array.astype(np.float32) / 255.0
                tensor = ((tensor - MEAN) / STD).transpose(2, 0, 1)[None, ...]
                logits = session.run(None, {input_name: tensor.astype(np.float32)})[0]
                onnx_index = int(np.argmax(logits, axis=1)[0])
                labels = [
                    "Anthrax_Leaf", "Bituminous_Leaf", "Curl_Leaf", "Deficiency_Leaf",
                    "Dry_Leaf", "Felt_Leaf", "Fungal_Leaf_Spot", "Healthy_Leaf",
                    "Leaf_Gall", "Leaf_Blight",
                ]
                onnx_label = labels[onnx_index]
                onnx_match = str(onnx_label == row["predicted_label"]).lower()
                verified += 1
                verify_matches += int(onnx_match == "true")

            manifest_rows.append({
                "index": index,
                "image_path": str(path),
                "true_label": row["true_label"],
                "pc_label": row["predicted_label"],
                "pc_confidence": row["confidence"],
                "pc_correct": str(row["true_label"] == row["predicted_label"]).lower(),
                "onnx_check_label": onnx_label,
                "onnx_pc_match": onnx_match,
                "byte_offset": index * 224 * 224 * 3,
                "byte_length": 224 * 224 * 3,
            })

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "samples": len(manifest_rows),
        "item_shape": [224, 224, 3],
        "item_layout": "HWC",
        "item_color": "RGB",
        "item_dtype": "uint8",
        "bytes_per_item": 224 * 224 * 3,
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": digest.hexdigest(),
        "source_predictions": str(args.predictions.resolve()),
        "onnx": str(args.onnx.resolve()),
        "onnx_verification_samples": verified,
        "onnx_pc_top1_matches": verify_matches,
        "preprocessing": "short-edge resize to 224, then 224x224 center crop",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if verify_matches != verified:
        raise SystemExit(
            f"ONNX preprocessing audit failed: {verify_matches}/{verified} Top-1 matches"
        )


if __name__ == "__main__":
    main()
