"""Controlled image-perturbation evaluation for the rebuttal."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nets import get_model_from_name
from utils.utils import cvtColor, get_classes, preprocess_input
from utils.utils_aug import CenterCrop, Resize


def perturb(image: np.ndarray, kind: str, level: float) -> np.ndarray:
    if kind == "brightness":
        return np.clip(image.astype(np.float32) * level, 0, 255).astype(np.uint8)
    if kind == "gaussian_blur":
        kernel = int(level)
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    if kind == "motion_blur":
        kernel = int(level)
        matrix = np.zeros((kernel, kernel), dtype=np.float32)
        matrix[kernel // 2, :] = 1.0 / kernel
        return cv2.filter2D(image, -1, matrix)
    if kind == "occlusion":
        result = image.copy()
        height, width = result.shape[:2]
        size = max(1, int(np.sqrt(height * width * level)))
        y0, x0 = (height - size) // 2, (width - size) // 2
        result[y0:y0 + size, x0:x0 + size] = 128
        return result
    if kind == "scale_crop":
        height, width = image.shape[:2]
        crop_h, crop_w = int(height / level), int(width / level)
        y0, x0 = (height - crop_h) // 2, (width - crop_w) // 2
        crop = image[y0:y0 + crop_h, x0:x0 + crop_w]
        return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    if kind == "rotation":
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), level, 1.0)
        return cv2.warpAffine(image, matrix, (width, height), borderValue=(128, 128, 128))
    if kind == "background_clutter":
        height, width = image.shape[:2]
        canvas = np.full_like(image, 176)
        tile = max(8, min(height, width) // 8)
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                canvas[y:y + tile, x:x + tile] = (96, 144, 96) if (x // tile + y // tile) % 2 else (184, 176, 128)
        scaled = cv2.resize(image, (int(width * 0.72), int(height * 0.72)), interpolation=cv2.INTER_AREA)
        y0, x0 = (height - scaled.shape[0]) // 2, (width - scaled.shape[1]) // 2
        canvas[y0:y0 + scaled.shape[0], x0:x0 + scaled.shape[1]] = scaled
        return canvas
    if kind == "noncentered":
        height, width = image.shape[:2]
        canvas = np.full_like(image, 128)
        scaled = cv2.resize(image, (int(width * 0.75), int(height * 0.75)), interpolation=cv2.INTER_AREA)
        y0, x0 = int(height * 0.05), int(width * 0.20)
        canvas[y0:y0 + scaled.shape[0], x0:x0 + scaled.shape[1]] = scaled
        return canvas
    raise ValueError(f"Unknown perturbation: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--backbone", required=True, choices=sorted(get_model_from_name))
    parser.add_argument("--test-annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fractal-box-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--fractal-threshold", type=float, default=0.5)
    args = parser.parse_args()
    root = args.project_root.resolve()
    classes, num_classes = get_classes(str(root / "data" / "classes.txt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_from_name[args.backbone](num_classes=num_classes, pretrained=False)
    if hasattr(model, "head") and hasattr(model.head, "frac"):
        model.head.frac.box_sizes = tuple(args.fractal_box_sizes)
        model.head.frac.threshold = args.fractal_threshold
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval().to(device)
    records = []
    for line in args.test_annotation.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label, path = line.split(";", 1)
        image_path = Path(path.strip())
        records.append((int(label), image_path if image_path.is_absolute() else root / image_path))
    perturbations = {
        "brightness_low": ("brightness", 0.7),
        "brightness_high": ("brightness", 1.3),
        "gaussian_blur": ("gaussian_blur", 5),
        "motion_blur": ("motion_blur", 9),
        "occlusion_10": ("occlusion", 0.10),
        "occlusion_25": ("occlusion", 0.25),
        "scale_crop_1.2": ("scale_crop", 1.2),
        "rotation_10": ("rotation", 10),
        "background_clutter": ("background_clutter", 1.0),
        "noncentered": ("noncentered", 1.0),
    }
    conditions = [("clean", (None, None)), *perturbations.items()]
    results = {name: [] for name, _ in conditions}
    pending_tensors, pending_labels, pending_conditions = [], [], []
    resize = Resize(224)
    center_crop = CenterCrop([224, 224])

    def flush() -> None:
        if not pending_tensors:
            return
        tensors = torch.stack(pending_tensors).to(device)
        labels = torch.tensor(pending_labels, device=device)
        try:
            logits = model(tensors, fractal_features=torch.zeros((len(tensors), 4), device=device))
        except TypeError:
            logits = model(tensors)
        correct = (logits.argmax(dim=1) == labels).cpu().tolist()
        for condition, is_correct in zip(pending_conditions, correct):
            results[condition].append(bool(is_correct))
        pending_tensors.clear()
        pending_labels.clear()
        pending_conditions.clear()

    with torch.no_grad():
        for true_label, path in records:
            # Match DataGenerator(..., random=False): resize the shorter edge
            # to 224 and then take a 224x224 center crop before perturbation.
            source_image = cvtColor(Image.open(path).convert("RGB"))
            source = np.asarray(center_crop(resize(source_image)))
            for name, (kind, level) in conditions:
                image = source if kind is None else perturb(source, kind, level)
                tensor = torch.from_numpy(np.transpose(preprocess_input(image.astype(np.float32)), (2, 0, 1))).float()
                pending_tensors.append(tensor)
                pending_labels.append(true_label)
                pending_conditions.append(name)
                if len(pending_tensors) >= args.batch_size:
                    flush()
        flush()
    clean_accuracy = float(np.mean(results["clean"]))
    summary = {
        "weights": str(args.weights),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "test_annotation": str(args.test_annotation),
        "test_annotation_sha256": hashlib.sha256(args.test_annotation.read_bytes()).hexdigest(),
        "backbone": args.backbone,
        "fractal_box_sizes": args.fractal_box_sizes,
        "fractal_threshold": args.fractal_threshold,
        "clean_preprocessing": "shorter-edge resize to 224 followed by 224x224 center crop; identical to the formal test loader",
        "occlusion_definition": "central square covering the stated fraction of total image area",
        "samples": len(records),
        "conditions": {
            name: {
                "accuracy": float(np.mean(values)),
                "accuracy_change_from_clean": float(np.mean(values) - clean_accuracy),
                "errors": int(len(values) - np.sum(values)),
            }
            for name, values in results.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
