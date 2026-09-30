"""Rebuild Fig. 8 from the archived real-scene photographs.

The procedure is intentionally qualitative: it crops fixed multi-scale windows,
classifies each crop with the frozen MobileNetV3-Fractal seed-43 checkpoint,
and visualizes high-confidence, non-overlapping candidates.  The photographs
have no per-leaf expert labels, so this script never calculates an accuracy.

It writes a new result directory by default and leaves the earlier Fig. 8
artifacts untouched.  Pass ``--overwrite`` only to replace a prior run using
the same output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mobilenetv3_fractal import mobilenetv3_fractal


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
WINDOW_SCALES = (0.40, 0.60)
WINDOW_STRIDE_RATIO = 0.50
DEFAULT_CONFIDENCE = 0.80
DEFAULT_NMS_IOU = 0.30
MAX_CANDIDATES = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positions(length: int, window: int, stride: int) -> list[int]:
    """Return start positions, always including the right/bottom boundary."""
    if window >= length:
        return [0]
    values = list(range(0, length - window + 1, stride))
    boundary = length - window
    if values[-1] != boundary:
        values.append(boundary)
    return values


def overlap_iou(left: dict[str, object], right: dict[str, object]) -> float:
    x1 = max(int(left["x1"]), int(right["x1"]))
    y1 = max(int(left["y1"]), int(right["y1"]))
    x2 = min(int(left["x2"]), int(right["x2"]))
    y2 = min(int(left["y2"]), int(right["y2"]))
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = (int(left["x2"]) - int(left["x1"])) * (int(left["y2"]) - int(left["y1"]))
    right_area = (int(right["x2"]) - int(right["x1"])) * (int(right["y2"]) - int(right["y1"]))
    union = left_area + right_area - intersection
    return float(intersection / union) if union else 0.0


def nms(rows: list[dict[str, object]], threshold: float) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: (-float(item["confidence"]), int(item["window_id"]))):
        if all(overlap_iou(row, previous) <= threshold for previous in selected):
            selected.append(row)
        if len(selected) == MAX_CANDIDATES:
            break
    return selected


def preprocess(image: Image.Image) -> torch.Tensor:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(rgb.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return torch.from_numpy(np.transpose(array, (2, 0, 1))).float()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_windows(image_path: Path, crops_dir: Path) -> tuple[Image.Image, list[dict[str, object]]]:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    short_edge = min(width, height)
    image_id = image_path.stem
    rows: list[dict[str, object]] = []
    window_id = 0
    for scale in WINDOW_SCALES:
        window = max(1, int(round(short_edge * scale)))
        stride = max(1, int(round(window * WINDOW_STRIDE_RATIO)))
        for top in positions(height, window, stride):
            for left in positions(width, window, stride):
                crop = image.crop((left, top, left + window, top + window))
                crop_name = f"{image_id}__w{window_id:04d}__s{scale:.2f}.jpg"
                crop_path = crops_dir / crop_name
                crop.save(crop_path, quality=95, subsampling=0)
                rows.append(
                    {
                        "image_id": image_id,
                        "window_id": window_id,
                        "source_filename": image_path.name,
                        "source_image": image_path.name,
                        "scale_ratio": f"{scale:.2f}",
                        "window_size": window,
                        "stride": stride,
                        "x1": left,
                        "y1": top,
                        "x2": left + window,
                        "y2": top + window,
                        "crop_path": f"crops/{crop_name}",
                    }
                )
                window_id += 1
    return image, rows


def diagnostic_figure(image: Image.Image, summary: dict[str, object], candidates: list[dict[str, object]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 7.6), dpi=180)
    axis.imshow(image)
    axis.axis("off")
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(candidates))))
    for index, row in enumerate(candidates):
        color = colors[index]
        x1, y1 = int(row["x1"]), int(row["y1"])
        width = int(row["x2"]) - x1
        height = int(row["y2"]) - y1
        axis.add_patch(Rectangle((x1, y1), width, height, fill=False, linewidth=2.4, edgecolor=color))
        text_y = y1 - 4 if y1 > 20 else y1 + 12
        axis.text(
            x1,
            text_y,
            f"{row['predicted_label']}  {float(row['confidence']):.2f}",
            color="white",
            fontsize=7,
            va="bottom" if y1 > 20 else "top",
            bbox={"facecolor": color, "alpha": 0.86, "pad": 1.5, "edgecolor": "none"},
        )
    # Do not render source filenames here: several are Chinese WeChat names and
    # a missing CJK font can turn them into unreadable glyph boxes in Fig. 8.
    title = f"{summary['review_status']} | retained windows: {summary['candidate_count']}"
    axis.set_title(title, fontsize=8.5)
    figure.tight_layout(pad=0.25)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def select_fig8_cases(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    """Show both unambiguous and conflicting results without claiming correctness."""
    selected: list[dict[str, object]] = []
    for status in ("high_confidence_candidate", "ambiguous_multi_window_prediction"):
        selected.extend([row for row in summaries if row["review_status"] == status][:2])
    if len(selected) < 4:
        fallback = [row for row in summaries if row not in selected]
        selected.extend(fallback[: 4 - len(selected)])
    if len(selected) < 4:
        raise RuntimeError("At least four readable source images are required to construct Fig. 8.")
    return selected


def build_fig8(selection: list[dict[str, object]], diagnostics_dir: Path, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 10.0), dpi=240)
    for index, (axis, row) in enumerate(zip(axes.flat, selection)):
        diagnostic_path = diagnostics_dir / f"{row['image_id']}.png"
        axis.imshow(Image.open(diagnostic_path).convert("RGB"))
        axis.axis("off")
        status = str(row["review_status"]).replace("_", " ")
        axis.set_title(f"({chr(97 + index)}) {status}", fontsize=8.5)
    figure.tight_layout(pad=0.55)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def load_model(weights: Path, classes_path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    model = mobilenetv3_fractal(num_classes=len(classes), pretrained=False)
    # These are the fixed descriptor settings of the seed-43 main model.
    model.head.frac.box_sizes = (1, 2, 4)
    model.head.frac.threshold = 0.5
    checkpoint = torch.load(weights, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model.eval().to(device), classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild Fig. 8 using the frozen MobileNetV3-Fractal seed-43 checkpoint.")
    parser.add_argument(
        "--images",
        type=Path,
        default=REPO_ROOT / "data" / "real_scene_23",
        help="Directory containing original whole-scene photos.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Frozen main-experiment checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "real_scene_23_sliding_window",
        help="New output directory. Existing output is rejected unless --overwrite is supplied.",
    )
    parser.add_argument(
        "--fig8-output",
        type=Path,
        default=REPO_ROOT / "figures" / "fig_rebuttal_real_scene_sliding_window.png",
        help="Four-panel Fig. 8 output.",
    )
    parser.add_argument("--classes", type=Path, default=REPO_ROOT / "data" / "classes.txt")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Delete only the specified output directory before regenerating it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = args.images.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    fig8_output = args.fig8_output.resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"Input image directory does not exist: {images_dir}")
    if not weights.is_file():
        raise SystemExit(f"Checkpoint does not exist: {weights}")
    if not 0.0 < args.confidence <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise SystemExit("--confidence must be in (0, 1] and --iou must be in [0, 1].")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output}. Use --overwrite to replace this run only.")
        if output.parent != (REPO_ROOT / "outputs").resolve():
            raise SystemExit("For safety, --overwrite is permitted only for directories immediately under outputs/.")
        shutil.rmtree(output)
    image_paths = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise SystemExit(f"No JPG, JPEG, or PNG files found in {images_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    output.mkdir(parents=True)
    crops_dir = output / "crops"
    diagnostics_dir = output / "diagnostic_figures"
    crops_dir.mkdir()
    diagnostics_dir.mkdir()

    images: dict[str, Image.Image] = {}
    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        image, image_rows = create_windows(image_path, crops_dir)
        if image_path.stem in images:
            raise SystemExit(f"Duplicate filename stem after extension removal: {image_path.stem}")
        images[image_path.stem] = image
        rows.extend(image_rows)

    model, classes = load_model(weights, args.classes.resolve(), device)
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            batch = torch.stack([preprocess(Image.open(output / Path(str(row["crop_path"])))) for row in batch_rows]).to(device)
            probabilities = torch.softmax(model(batch), dim=1)
            top_values, top_indices = probabilities.topk(k=min(3, len(classes)), dim=1)
            confidence, predicted = probabilities.max(dim=1)
            for index, row in enumerate(batch_rows):
                probability = probabilities[index].detach().cpu().numpy()
                row["predicted_label"] = classes[int(predicted[index])]
                row["confidence"] = float(confidence[index])
                row["top3_labels"] = "|".join(classes[int(value)] for value in top_indices[index].detach().cpu().tolist())
                row["top3_probabilities"] = "|".join(f"{float(value):.6f}" for value in top_values[index].detach().cpu().tolist())
                row["entropy"] = float(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))
                row["above_threshold"] = str(float(row["confidence"]) >= args.confidence).lower()

    candidates: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for image_id, image in images.items():
        image_rows = [row for row in rows if row["image_id"] == image_id]
        selected = nms([row for row in image_rows if float(row["confidence"]) >= args.confidence], args.iou)
        candidates.extend(selected)
        labels = {str(row["predicted_label"]) for row in selected}
        best = selected[0] if selected else None
        status = "low_confidence" if not selected else ("ambiguous_multi_window_prediction" if len(labels) > 1 else "high_confidence_candidate")
        summary: dict[str, object] = {
            "image_id": image_id,
            "source_filename": image_rows[0]["source_filename"],
            "source_image": image_rows[0]["source_image"],
            "window_count": len(image_rows),
            "candidate_count": len(selected),
            "best_window_id": "" if best is None else best["window_id"],
            "best_predicted_label": "" if best is None else best["predicted_label"],
            "best_confidence": "" if best is None else f"{float(best['confidence']):.6f}",
            "top3_labels": "" if best is None else best["top3_labels"],
            "prediction_agreement": "" if not selected else ("agree" if len(labels) == 1 else "disagree"),
            "review_status": status,
        }
        summaries.append(summary)
        diagnostic_figure(image, summary, selected, diagnostics_dir / f"{image_id}.png")

    window_fields = [
        "image_id", "window_id", "source_filename", "source_image", "scale_ratio", "window_size", "stride", "x1", "y1", "x2", "y2",
        "crop_path", "predicted_label", "confidence", "top3_labels", "top3_probabilities", "entropy", "above_threshold",
    ]
    summary_fields = [
        "image_id", "source_filename", "source_image", "window_count", "candidate_count", "best_window_id", "best_predicted_label",
        "best_confidence", "top3_labels", "prediction_agreement", "review_status",
    ]
    write_csv(output / "all_window_predictions.csv", rows, window_fields)
    write_csv(output / "high_confidence_candidates.csv", candidates, window_fields)
    write_csv(output / "per_image_summary.csv", summaries, summary_fields)
    selection = select_fig8_cases(summaries)
    write_csv(output / "fig8_selection.csv", selection, summary_fields)
    build_fig8(selection, diagnostics_dir, fig8_output)

    manifest = {
        "purpose": "Qualitative real-scene sliding-window diagnosis; no accuracy is reported.",
        "script": "scripts/run_real_scene_sliding_window.py",
        "source_images": "data/real_scene_23",
        "source_image_sha256": {path.name: sha256(path) for path in image_paths},
        "weights_filename": weights.name,
        "weights_sha256": sha256(weights),
        "model": "MobileNetV3-Fractal",
        "seed": 43,
        "classes": classes,
        "input_size": [224, 224],
        "window_scales": list(WINDOW_SCALES),
        "stride_ratio": WINDOW_STRIDE_RATIO,
        "confidence_threshold": args.confidence,
        "nms_iou_threshold": args.iou,
        "max_candidates_per_image": MAX_CANDIDATES,
        "device": str(device),
        "images": len(image_paths),
        "windows": len(rows),
        "retained_candidates": len(candidates),
        "fig8_output": "figures/fig_rebuttal_real_scene_sliding_window.png",
        "fig8_source_images": [str(row["source_filename"]) for row in selection],
    }
    (output / "manifest_sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Rebuilt Fig. 8: real-scene sliding-window diagnosis\n\n"
        "This run uses frozen MobileNetV3-Fractal seed-43 on the archived original whole-scene photographs. "
        "It generates 40% and 60% short-edge square windows with a 50% window-size stride, then retains high-confidence "
        "(>= 0.80) windows after IoU NMS (0.30). The images do not have per-leaf expert ground truth; these results are "
        "qualitative diagnostic demonstrations and do not report an external-test accuracy. `fig8_selection.csv` records "
        "the four panels used in the manuscript figure.\n",
        encoding="utf-8",
    )
    print(json.dumps({"images": len(image_paths), "windows": len(rows), "retained_candidates": len(candidates), "fig8": str(fig8_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
