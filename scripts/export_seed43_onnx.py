"""Export the audited seed-43 GAP+Fractal checkpoint and verify ONNX parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nets import get_model_from_name
from utils.utils import get_classes


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preprocess(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT  / "weights" / "fractal_seed_43" / "best_epoch_weights.pth")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT  / "results" / "real_scene_23_sliding_window" / "model")
    parser.add_argument("--regression-csv", type=Path, default=PROJECT_ROOT  / "results" / "real_scene_23_sliding_window" / "all_window_predictions.csv")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    classes, num_classes = get_classes(str(PROJECT_ROOT / "data" / "classes.txt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_from_name["mobilenetv3_fractal"](num_classes=num_classes, pretrained=False)
    model.head.frac.box_sizes = (1, 2, 4)
    model.head.frac.threshold = 0.5
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval().to(device)
    dummy = torch.randn(1, 3, 224, 224, device=device)
    onnx_path = output / "mobilenetv3_fractal_seed43_op13.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=13,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes=None,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    regression_paths = []
    if args.regression_csv.exists():
        import csv
        with args.regression_csv.open(encoding="utf-8", newline="") as handle:
            regression_paths = [Path(row["crop_path"]) for row in list(csv.DictReader(handle))[:20]]
    if not regression_paths:
        raise SystemExit(f"No regression crops found in {args.regression_csv}")
    max_abs = 0.0
    top1_matches = 0
    with torch.inference_mode():
        for path in regression_paths:
            array = preprocess(path)
            torch_output = model(torch.from_numpy(array).to(device)).detach().cpu().numpy()
            onnx_output = session.run(["logits"], {"images": array})[0]
            max_abs = max(max_abs, float(np.max(np.abs(torch_output - onnx_output))))
            top1_matches += int(np.argmax(torch_output, axis=1)[0] == np.argmax(onnx_output, axis=1)[0])
    metadata = {
        "weights": str(args.weights.resolve()),
        "weights_sha256": digest(args.weights.resolve()),
        "onnx": str(onnx_path),
        "onnx_sha256": digest(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "opset": 13,
        "input_shape": [1, 3, 224, 224],
        "input_layout": "NCHW",
        "input_color": "RGB",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "classes": classes,
        "regression_samples": len(regression_paths),
        "torch_onnx_top1_matches": top1_matches,
        "torch_onnx_max_abs_logit_difference": max_abs,
        "device_for_export": str(device),
    }
    (output / "onnx_export_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
