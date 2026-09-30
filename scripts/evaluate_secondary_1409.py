"""Convert the YOLO-style 1409-image archive to a source-grouped classification evaluation.

The raw archive is never modified. One deterministic representative is selected per
filename-derived source group to prevent augmented variants from inflating the score.
Frozen checkpoints trained on the primary 3,765-image protocol are evaluated directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MAIN_CLASSES = [
    "Anthrax_Leaf", "Bituminous_Leaf", "Curl_Leaf", "Deficiency_Leaf", "Dry_Leaf",
    "Felt_Leaf", "Fungal_Leaf_Spot", "Healthy_Leaf", "Leaf_Gall", "Leaf_Blight",
]
SECONDARY_CLASSES = [
    "Curl_Leaf", "Deficiency_Leaf", "Felt_Leaf", "Fungal_Leaf_Spot",
    "Healthy_Leaf", "Leaf_Blight", "Leaf_Gall",
]
YOLO_TO_CLASS = {index: name for index, name in enumerate(SECONDARY_CLASSES)}
NAME_PATTERN = re.compile(
    r"(?:aug\d+_|orig_)?(Curl-Leaf|Deficiency-Leaf|Felt-Leaf|Fungal-Leaf-Spot|"
    r"Healthy-Leaf|Leaf-Blight|Leaf-Gall)(\d+)", re.IGNORECASE
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_source_key(stem: str) -> tuple[str, str]:
    match = NAME_PATTERN.search(stem)
    if not match:
        return "", ""
    class_name = match.group(1).replace("-", "_")
    source_id = f"{class_name}{int(match.group(2)):05d}"
    return source_id, class_name


def read_yolo_boxes(label_path: Path) -> tuple[str, list[tuple[float, float, float, float]]]:
    labels = []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO label: {label_path}: {line}")
        class_id = int(parts[0])
        if class_id not in YOLO_TO_CLASS:
            raise ValueError(f"Unknown class id {class_id}: {label_path}")
        labels.append(YOLO_TO_CLASS[class_id])
        boxes.append(tuple(float(value) for value in parts[1:]))
    if not boxes:
        raise ValueError(f"Empty label file: {label_path}")
    if len(set(labels)) != 1:
        raise ValueError(f"Multiple classes in one image: {label_path}: {labels}")
    return labels[0], boxes


def representative_rank(path: Path, split: str) -> tuple[int, int, str]:
    stem = path.stem.lower()
    kind = 0 if stem.startswith("orig_") else 1 if stem.startswith("aug2_") else 2 if stem.startswith("aug1_") else 3
    split_rank = {"test": 0, "valid": 1, "train": 2}.get(split, 3)
    return kind, split_rank, path.name


def build_manifest(raw_root: Path, output_dir: Path) -> list[dict[str, str]]:
    primary_hashes = {sha256(path) for path in (PROJECT_ROOT / "data" / "primary_external").rglob("*.jpg")}
    rows = []
    groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for split in ("train", "valid", "test"):
        image_dir = raw_root / split / "images"
        label_dir = raw_root / split / "labels"
        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = label_dir / f"{image_path.stem}.txt"
            label, boxes = read_yolo_boxes(label_path)
            source_id, parsed_class = parse_source_key(image_path.stem)
            if not source_id:
                source_id = f"unparsed::{sha256(image_path)}"
            row = {
                "image_path": str(image_path.resolve()),
                "label": label,
                "source_id": source_id,
                "parsed_class": parsed_class,
                "original_split": split,
                "augmentation_kind": "orig" if image_path.stem.lower().startswith("orig_") else "aug2" if image_path.stem.lower().startswith("aug2_") else "aug1" if image_path.stem.lower().startswith("aug1_") else "other",
                "bbox_count": str(len(boxes)),
                "bbox_union": json.dumps(box_union(boxes)),
                "sha256": sha256(image_path),
                "exact_overlap_with_primary": str(sha256(image_path) in primary_hashes).lower(),
            }
            groups[source_id].append(row)
            rows.append(row)
    selected = set()
    for source_id, candidates in groups.items():
        chosen = min(candidates, key=lambda row: representative_rank(Path(row["image_path"]), row["original_split"]))
        selected.add(chosen["image_path"])
    for row in rows:
        row["source_group_size"] = str(len(groups[row["source_id"]]))
        row["source_group_representative"] = str(row["image_path"] in selected).lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) + ["source_group_size", "source_group_representative"]
    for filename, subset in (("secondary_1409_manifest_all.csv", rows), ("secondary_1409_manifest_representative.csv", [r for r in rows if r["source_group_representative"] == "true"])):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)
    summary = {
        "raw_images": len(rows),
        "source_groups": len(groups),
        "representative_images": len(selected),
        "classes": dict(Counter(row["label"] for row in rows)),
        "representative_classes": dict(Counter(row["label"] for row in rows if row["source_group_representative"] == "true")),
        "original_split_counts": dict(Counter(row["original_split"] for row in rows)),
        "augmentation_kind_counts": dict(Counter(row["augmentation_kind"] for row in rows)),
        "exact_overlap_with_primary": sum(row["exact_overlap_with_primary"] == "true" for row in rows),
        "statement_limit": "This archive is augmentation-aware and lacks complete collection metadata; source-group evaluation is secondary-domain evidence, not automatically verified independent external validation.",
    }
    (output_dir / "secondary_1409_manifest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return [row for row in rows if row["source_group_representative"] == "true"]


def box_union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    corners = []
    for cx, cy, width, height in boxes:
        corners.append((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2))
    return (min(box[0] for box in corners), min(box[1] for box in corners), max(box[2] for box in corners), max(box[3] for box in corners))


def crop_normalized(image: Image.Image, union: tuple[float, float, float, float], margin: float) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = union
    box_width, box_height = right - left, bottom - top
    left = max(0.0, left - margin * box_width)
    top = max(0.0, top - margin * box_height)
    right = min(1.0, right + margin * box_width)
    bottom = min(1.0, bottom + margin * box_height)
    return image.crop((round(left * width), round(top * height), round(right * width), round(bottom * height)))


class SecondaryDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], mode: str, crop_margin: float = 0.10):
        self.rows = rows
        self.mode = mode
        self.crop_margin = crop_margin

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.mode == "bbox":
            image = crop_normalized(image, tuple(json.loads(row["bbox_union"])), self.crop_margin)
        short_edge = min(image.size)
        scale = 224 / short_edge
        resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR)
        left = max(0, (resized.width - 224) // 2)
        top = max(0, (resized.height - 224) // 2)
        image = resized.crop((left, top, left + 224, top + 224))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return torch.from_numpy(array.transpose(2, 0, 1)).float(), MAIN_CLASSES.index(row["label"]), row


def secondary_collate(batch):
    images, targets, rows = zip(*batch)
    return torch.stack(images), torch.tensor(targets, dtype=torch.long), list(rows)


def evaluate_checkpoint(weights: Path, backbone: str, rows: list[dict[str, str]], output_dir: Path, mode: str, box_sizes: list[int], threshold: float) -> dict:
    from nets import get_model_from_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_from_name[backbone](num_classes=len(MAIN_CLASSES), pretrained=False)
    if hasattr(model, "head") and hasattr(model.head, "frac"):
        model.head.frac.box_sizes = tuple(box_sizes)
        model.head.frac.threshold = threshold
    checkpoint = torch.load(weights, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval().to(device)
    loader = DataLoader(SecondaryDataset(rows, mode), batch_size=32, shuffle=False, num_workers=0, collate_fn=secondary_collate)
    y_true, y_pred, confidences, metadata = [], [], [], []
    with torch.no_grad():
        for images, targets, batch_rows in loader:
            try:
                logits = model(images.to(device), fractal_features=torch.zeros((images.size(0), 4), device=device))
            except TypeError:
                logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
            confidences.extend(confidence.cpu().numpy().tolist())
            metadata.extend(batch_rows)
    y_true_np, y_pred_np = np.asarray(y_true), np.asarray(y_pred)
    present = sorted(set(y_true_np.tolist()))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true_np, y_pred_np, labels=present, target_names=[MAIN_CLASSES[i] for i in present], output_dict=True, zero_division=0)
    metrics = {
        "scope": "all 1,409 merged rows (augmentation-aware diagnostic)" if len(rows) == 1409 else "source-group deduplicated secondary-domain evaluation",
        "mode": mode,
        "samples": len(rows),
        "covered_classes": [MAIN_CLASSES[i] for i in present],
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "macro_precision": float(precision_score(y_true_np, y_pred_np, labels=present, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_np, y_pred_np, labels=present, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, labels=present, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix_10_classes": confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(MAIN_CLASSES)))).tolist(),
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "backbone": backbone,
        "fractal_box_sizes": box_sizes,
        "fractal_threshold": threshold,
        "statement_limit": "The seven-class archive is not treated as full-class or automatically verified independent external validation.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [*metadata[0].keys(), "predicted_label", "correct", "confidence"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, pred, confidence in zip(metadata, y_pred, confidences):
            writer.writerow({**row, "predicted_label": MAIN_CLASSES[pred], "correct": str(row["label"] == MAIN_CLASSES[pred]).lower(), "confidence": f"{confidence:.6f}"})
    print(json.dumps({key: metrics[key] for key in ("scope", "mode", "samples", "accuracy", "macro_f1")}, ensure_ascii=False))
    return metrics


def run_matrix(rows: list[dict[str, str]], weights_root: Path, output_dir: Path, mode: str, box_sizes: list[int], threshold: float) -> dict:
    results = {}
    variants = {
        "gap": ("mobilenetv3_gap", "gap_seed_"),
        "gap_gmp": ("mobilenetv3_gap_gmp", "gap_gmp_seed_"),
        "fractal": ("mobilenetv3_fractal", "fractal_seed_"),
    }
    for variant, (backbone, prefix) in variants.items():
        results[variant] = {}
        for seed in (42, 43, 44):
            weights = weights_root / f"{prefix}{seed}" / "best_epoch_weights.pth"
            results[variant][str(seed)] = evaluate_checkpoint(weights, backbone, rows, output_dir / variant / f"seed_{seed}", mode, box_sizes, threshold)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("bbox", "full"), default="bbox")
    parser.add_argument("--include-all", action="store_true", help="Also evaluate all 1,409 rows after merging the original splits.")
    parser.add_argument("--box-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    rows = build_manifest(args.raw_root, args.output_dir / "manifest")
    results = {"source_group_representative": run_matrix(rows, args.weights_root, args.output_dir / "source_group_representative", args.mode, args.box_sizes, args.threshold)}
    if args.include_all:
        with (args.output_dir / "manifest" / "secondary_1409_manifest_all.csv").open(encoding="utf-8", newline="") as handle:
            all_rows = list(csv.DictReader(handle))
        results["all_1409_merged_splits"] = run_matrix(all_rows, args.weights_root, args.output_dir / "all_1409_merged_splits", args.mode, args.box_sizes, args.threshold)
    (args.output_dir / f"summary_{args.mode}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
