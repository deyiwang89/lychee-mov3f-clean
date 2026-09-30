"""Evaluate one rebuttal checkpoint with reproducible per-image outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nets import get_model_from_name
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import get_classes


def model_forward(model, images, fractal_features):
    try:
        return model(images, fractal_features=fractal_features)
    except TypeError:
        return model(images)


def paired_bootstrap(y_true, pred_a, pred_b, iterations=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        deltas[i] = np.mean(pred_b[idx] == y_true[idx]) - np.mean(pred_a[idx] == y_true[idx])
    return {
        "accuracy_delta": float(np.mean(pred_b == y_true) - np.mean(pred_a == y_true)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def mcnemar_exact(y_true, pred_a, pred_b):
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b01 = int(np.sum(a_correct & ~b_correct))
    b10 = int(np.sum(~a_correct & b_correct))
    n = b01 + b10
    if n == 0:
        return {"a_wrong_b_correct": b10, "a_correct_b_wrong": b01, "p_value": 1.0}
    k = min(b01, b10)
    p_value = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    return {"a_wrong_b_correct": b10, "a_correct_b_wrong": b01, "p_value": float(p_value)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--backbone", required=True, choices=sorted(get_model_from_name))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--test-annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fractal-box-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--fractal-threshold", type=float, default=0.5)
    parser.add_argument("--compare-csv", type=Path, help="Optional second prediction CSV for paired tests")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
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
    lines = [line.strip() for line in args.test_annotation.read_text(encoding="utf-8").splitlines() if line.strip()]
    # DataGenerator opens annotation paths directly; normalize them against the
    # project root so the evaluator is independent of the caller's cwd.
    lines = [
        f"{label};{(root / image_path).resolve()}"
        for label, image_path in (line.split(";", 1) for line in lines)
    ]
    loader = DataLoader(
        DataGenerator(lines, [224, 224], False, extract_external_fractal=False),
        batch_size=32, shuffle=False, collate_fn=detection_collate, num_workers=0,
    )
    y_true, y_pred, confidences, paths = [], [], [], []
    top5_correct = 0
    with torch.no_grad():
        for offset, batch in enumerate(loader):
            images, targets, fractal_features = batch
            logits = model_forward(model, images.to(device), fractal_features.to(device))
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            top5 = probabilities.topk(k=min(5, num_classes), dim=1).indices
            top5_correct += int((top5 == targets.to(device).unsqueeze(1)).any(dim=1).sum().item())
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
            confidences.extend(confidence.cpu().numpy().tolist())
    for line in lines:
        paths.append(line.split(";", 1)[1].split()[0])
    y_true_np, y_pred_np = np.array(y_true), np.array(y_pred)
    report = classification_report(y_true_np, y_pred_np, target_names=classes, output_dict=True, zero_division=0)
    metrics = {
        "weights": str(args.weights),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "weights_size_bytes": args.weights.stat().st_size,
        "backbone": args.backbone,
        "seed": args.seed,
        "test_annotation": str(args.test_annotation),
        "test_annotation_sha256": hashlib.sha256(args.test_annotation.read_bytes()).hexdigest(),
        "fractal_box_sizes": args.fractal_box_sizes,
        "fractal_threshold": args.fractal_threshold,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "samples": len(y_true),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "top5_accuracy": float(top5_correct / len(y_true)),
        "macro_precision": float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true_np, y_pred_np).tolist(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "true_label", "predicted_label", "status", "confidence"])
        for path, true, pred, confidence in zip(paths, y_true, y_pred, confidences):
            writer.writerow([path, classes[true], classes[pred], "Correct" if true == pred else "Incorrect", f"{confidence:.6f}"])
    print(json.dumps({key: metrics[key] for key in ("samples", "accuracy", "top5_accuracy", "macro_precision", "macro_recall", "macro_f1")}, indent=2))

    if args.compare_csv:
        with args.compare_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        compare_pred = np.array([classes.index(row["predicted_label"]) for row in rows])
        if len(compare_pred) != len(y_true_np):
            raise SystemExit("--compare-csv has a different number of predictions")
        comparison = {
            "paired_bootstrap": paired_bootstrap(y_true_np, compare_pred, y_pred_np),
            "mcnemar_exact": mcnemar_exact(y_true_np, compare_pred, y_pred_np),
        }
        (output_dir / "paired_tests.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
