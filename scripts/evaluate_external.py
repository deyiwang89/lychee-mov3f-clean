"""Evaluate a frozen checkpoint once on a validated partial external set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nets import get_model_from_name
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import get_classes


REQUIRED_METADATA = ("collection_date", "location", "device", "lighting", "background", "annotator", "reviewer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--backbone", required=True, choices=sorted(get_model_from_name))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fractal-box-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--fractal-threshold", type=float, default=0.5)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    classes, num_classes = get_classes(str(root / "data" / "classes.txt"))
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("External manifest is empty")
    missing_fields = [field for field in ("image_path", "label", "sha256", *REQUIRED_METADATA) if field not in rows[0]]
    if missing_fields:
        raise SystemExit(f"External manifest is missing columns: {missing_fields}")
    incomplete = [index + 2 for index, row in enumerate(rows) if any(not row[field].strip() for field in REQUIRED_METADATA)]
    if incomplete:
        raise SystemExit(f"External metadata is incomplete in CSV rows: {incomplete[:20]}")
    unknown_labels = sorted({row["label"] for row in rows} - set(classes))
    if unknown_labels:
        raise SystemExit(f"Unknown external labels: {unknown_labels}")
    for row in rows:
        path = Path(row["image_path"])
        if not path.exists():
            raise SystemExit(f"Missing external image: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for external image: {path}")

    annotations = [f"{classes.index(row['label'])};{Path(row['image_path']).resolve()}" for row in rows]
    loader = DataLoader(
        DataGenerator(annotations, [224, 224], False, extract_external_fractal=False),
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=detection_collate, num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_from_name[args.backbone](num_classes=num_classes, pretrained=False)
    if hasattr(model, "head") and hasattr(model.head, "frac"):
        model.head.frac.box_sizes = tuple(args.fractal_box_sizes)
        model.head.frac.threshold = args.fractal_threshold
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict.get("model", state_dict) if isinstance(state_dict, dict) else state_dict)
    model.eval().to(device)
    y_true, y_pred, confidences = [], [], []
    with torch.no_grad():
        for images, targets, fractal_features in loader:
            images = images.to(device)
            try:
                logits = model(images, fractal_features=fractal_features.to(device))
            except TypeError:
                logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            y_true.extend(targets.tolist())
            y_pred.extend(predicted.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    present_ids = sorted(set(y_true))
    y_true_np, y_pred_np = np.asarray(y_true), np.asarray(y_pred)
    report = classification_report(
        y_true_np, y_pred_np, labels=present_ids,
        target_names=[classes[index] for index in present_ids], output_dict=True, zero_division=0,
    )
    metrics = {
        "scope": "partial external validation",
        "covered_classes": [classes[index] for index in present_ids],
        "samples": len(y_true),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "classification_report": report,
        "confusion_matrix_over_all_model_classes": confusion_matrix(y_true_np, y_pred_np, labels=list(range(num_classes))).tolist(),
        "weights": str(args.weights),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "statement_limit": "This partial external set does not establish full-class, cross-season, or cross-location generalization.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [*rows[0].keys(), "predicted_label", "correct", "confidence"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, predicted, confidence in zip(rows, y_pred, confidences):
            writer.writerow({
                **row,
                "predicted_label": classes[predicted],
                "correct": row["label"] == classes[predicted],
                "confidence": f"{confidence:.6f}",
            })
    print(json.dumps({key: metrics[key] for key in ("scope", "covered_classes", "samples", "accuracy")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
