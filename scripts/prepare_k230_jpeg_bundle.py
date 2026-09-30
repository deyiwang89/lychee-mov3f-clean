"""Pack center-cropped test images as length-prefixed JPEG records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
LABELS = [
    "Anthrax_Leaf", "Bituminous_Leaf", "Curl_Leaf", "Deficiency_Leaf",
    "Dry_Leaf", "Felt_Leaf", "Fungal_Leaf_Spot", "Healthy_Leaf",
    "Leaf_Gall", "Leaf_Blight",
]


def resize_center_crop(image: Image.Image, size: int = 224) -> Image.Image:
    width, height = image.size
    if not ((width <= height and width == size) or (height <= width and height == size)):
        if width < height:
            width, height = size, int(size * height / width)
        else:
            height, width = size, int(size * width / height)
        image = image.resize((width, height), Image.Resampling.BILINEAR)
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
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with args.predictions.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    bundle_path = args.output / "seed43_test_jpeg_records.bin"
    manifest_path = args.output / "manifest.csv"
    digest = hashlib.sha256()
    manifest_rows = []
    verify_matches = 0
    labels_by_index = {label: index for index, label in enumerate(LABELS)}

    with bundle_path.open("wb") as bundle:
        for index, row in enumerate(source_rows):
            image = resize_center_crop(Image.open(Path(row["image_path"])).convert("RGB"))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=args.quality, subsampling=0, optimize=False)
            payload = buffer.getvalue()
            bundle.write(struct.pack("<I", len(payload)))
            bundle.write(payload)
            digest.update(struct.pack("<I", len(payload)))
            digest.update(payload)

            bundle_pc_label = row["predicted_label"]
            bundle_pc_confidence = row["confidence"]
            bundle_pc_match = "not_checked"
            if index < args.verify_count:
                decoded = Image.open(io.BytesIO(payload)).convert("RGB")
                array = np.asarray(decoded, dtype=np.float32) / 255.0
                tensor = ((array - MEAN) / STD).transpose(2, 0, 1)[None, ...]
                logits = session.run(None, {input_name: tensor.astype(np.float32)})[0][0]
                probabilities = np.exp(logits - np.max(logits))
                probabilities /= probabilities.sum()
                predicted_index = int(np.argmax(probabilities))
                bundle_pc_label = LABELS[predicted_index]
                bundle_pc_confidence = f"{float(probabilities[predicted_index]):.6f}"
                bundle_pc_match = str(bundle_pc_label == row["predicted_label"]).lower()
                verify_matches += int(bundle_pc_match == "true")

            manifest_rows.append({
                "index": index,
                "image_path": row["image_path"],
                "true_label": row["true_label"],
                "pc_original_label": row["predicted_label"],
                "pc_bundle_label": bundle_pc_label,
                "pc_bundle_confidence": bundle_pc_confidence,
                "pc_original_correct": str(row["true_label"] == row["predicted_label"]).lower(),
                "pc_bundle_correct": str(bundle_pc_label == row["true_label"]).lower() if bundle_pc_label else "",
                "pc_bundle_matches_original": bundle_pc_match,
                "byte_offset": bundle.tell() - len(payload) - 4,
                "jpeg_length": len(payload),
            })

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    metadata = {
        "samples": len(manifest_rows),
        "record_format": "uint32_le_jpeg_length followed by JPEG bytes",
        "jpeg_quality": args.quality,
        "jpeg_subsampling": 0,
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": digest.hexdigest(),
        "onnx_verification_samples": args.verify_count,
        "onnx_bundle_vs_original_top1_matches": verify_matches,
        "preprocessing": "short-edge resize to 224, then 224x224 center crop, JPEG encode",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if verify_matches != min(args.verify_count, len(source_rows)):
        raise SystemExit(f"JPEG audit failed: {verify_matches}/{min(args.verify_count, len(source_rows))}")


if __name__ == "__main__":
    main()
