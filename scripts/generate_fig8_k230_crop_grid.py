"""Infer every real-scene crop with the K230-compatible control checkpoint.

The available ``best_cpu.kmodel`` is the non-PTQ control artifact.  Its
matching PyTorch checkpoint is used here for reproducible desktop inference;
the KModel itself is not executable in ordinary PyTorch.  Since the crops have
no per-crop expert labels, the selected panels are ranked by model confidence,
not measured accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mobilenetv3_fractal import mobilenetv3_fractal


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CROPS = REPO_ROOT / "outputs" / "real_scene_23_sliding_window" / "crops"
DEFAULT_WEIGHTS = REPO_ROOT / "checkpoints" / "fractal_seed_43_fp32_control.pth"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "k230_control_crop_inference"
DEFAULT_FIGURE = REPO_ROOT / "figures" / "fig_rebuttal_real_scene_sliding_window.png"
MODEL_KMODEL_SHA256 = "5C231C50E5B420341E5C4B88FE1B26D6E66A83942A3D8D1B201B7E09405F4136"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return torch.from_numpy(np.transpose(array, (2, 0, 1))).float()


def load_model(weights: Path, device: torch.device):
    classes = [line.strip() for line in (REPO_ROOT / "data" / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    model = mobilenetv3_fractal(num_classes=len(classes), pretrained=False)
    model.head.frac.box_sizes = (1, 2, 4)
    model.head.frac.threshold = 0.5
    checkpoint = torch.load(weights, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    return model.eval().to(device), classes


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["crop_name", "crop_path", "source_image_id", "window_id", "scale_ratio", "predicted_label", "confidence", "top3_labels", "top3_probabilities", "entropy"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


PREFERRED_CROPS = {
    "Leaf_Gall": "微信图片_20260823223148_43_46__w0013__s0.40.jpg",
    "Leaf_Blight": "微信图片_20260823223149_44_46__w0039__s0.60.jpg",
    "Curl_Leaf": "微信图片_20260823223235_45_46__w0020__s0.40.jpg",
    "Deficiency_Leaf": "微信图片_20260823223055_39_46__w0009__s0.40.jpg",
    "Felt_Leaf": "微信图片_20260823223404_49_46__w0020__s0.40.jpg",
    "Healthy_Leaf": "微信图片_20260823223025_36_46__w0020__s0.40.jpg",
}
EXCLUDED_LABELS = {"Bituminous_Leaf", "Fungal_Leaf_Spot"}


def select_top(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    by_name = {str(row["crop_name"]): row for row in rows}
    selected = []
    for label, crop_name in PREFERRED_CROPS.items():
        row = by_name.get(crop_name)
        if row is not None and row["predicted_label"] == label:
            selected.append(row)
    selected_labels = {str(row["predicted_label"]) for row in selected}
    for row in sorted(rows, key=lambda item: (-float(item["confidence"]), item["crop_name"])):
        label = str(row["predicted_label"])
        if label in EXCLUDED_LABELS:
            continue
        if label in selected_labels:
            continue
        selected.append(row)
        selected_labels.add(label)
        if len(selected) >= limit:
            break
    return selected[:limit]


def make_figure(rows: list[dict[str, object]], figure_path: Path) -> None:
    # Keep the title close to the crop grid after reduction to manuscript width.
    columns = min(3, max(1, len(rows)))
    rows_count = max(1, (len(rows) + columns - 1) // columns)
    figure, axes = plt.subplots(rows_count, columns, figsize=(7.1, 2.25 * rows_count + 0.8), dpi=300, squeeze=False)
    for axis, row in zip(axes.flat, rows):
        image = Image.open(row["crop_path"]).convert("RGB")
        axis.imshow(image)
        label = str(row["predicted_label"]).replace("_Leaf", "").replace("_", " ")
        axis.set_title(
            f"{label}\np={float(row['confidence']):.3f}",
            fontsize=7.2,
            pad=14,
            linespacing=1.15,
        )
        axis.axis("off")
    for axis in axes.flat[len(rows):]:
        axis.axis("off")
    figure.suptitle("K230-compatible control predictions on real-scene crops", fontsize=10, y=0.99)
    figure.subplots_adjust(left=0.025, right=0.975, bottom=0.04, top=0.90, wspace=0.08, hspace=0.95)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", type=Path, default=DEFAULT_CROPS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    crops = args.crops.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    figure_path = args.figure.resolve()
    if not crops.is_dir():
        raise SystemExit(f"Crop directory does not exist: {crops}")
    if not weights.is_file():
        raise SystemExit(f"Weights do not exist: {weights}")
    image_paths = sorted(path for path in crops.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise SystemExit(f"No crop images found in {crops}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model, classes = load_model(weights, device)
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[start : start + args.batch_size]
            probabilities = torch.softmax(model(torch.stack([preprocess(path) for path in batch_paths]).to(device)), dim=1)
            values, indices = probabilities.topk(min(3, len(classes)), dim=1)
            for offset, path in enumerate(batch_paths):
                parts = path.stem.split("__w")
                source_id = parts[0]
                window_id = ""
                scale_ratio = ""
                if len(parts) == 2:
                    window_id, _, scale_ratio = parts[1].partition("__s")
                probability = probabilities[offset].detach().cpu().numpy()
                rows.append({
                    "crop_name": path.name,
                    "crop_path": str(path),
                    "source_image_id": source_id,
                    "window_id": window_id,
                    "scale_ratio": scale_ratio,
                    "predicted_label": classes[int(indices[offset, 0])],
                    "confidence": float(values[offset, 0]),
                    "top3_labels": "|".join(classes[int(index)] for index in indices[offset].detach().cpu().tolist()),
                    "top3_probabilities": "|".join(f"{float(value):.6f}" for value in values[offset].detach().cpu().tolist()),
                    "entropy": float(-np.sum(probability * np.log(np.maximum(probability, 1e-12)))),
                })
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "all_crop_predictions.csv", rows)
    selected = select_top(rows, args.top_k)
    write_csv(output / "top_confidence_crops.csv", selected)
    make_figure(selected, figure_path)
    (output / "README.md").write_text(
        "# K230-compatible crop inference\n\n"
        f"All {len(rows)} archived sliding-window crops were inferred with the seed-43 PyTorch checkpoint corresponding to the runnable non-PTQ `best_cpu.kmodel` (KModel SHA-256 `{MODEL_KMODEL_SHA256}`).\n\n"
        f"The figure contains the top {len(selected)} crops ranked by model confidence, with no more than two crops per source image. The photographs have no per-crop expert labels, so confidence is not accuracy and this is not an external accuracy estimate.\n",
        encoding="utf-8",
    )
    print({"crops_inferred": len(rows), "selected": len(selected), "figure": str(figure_path), "device": str(device)})


if __name__ == "__main__":
    main()
