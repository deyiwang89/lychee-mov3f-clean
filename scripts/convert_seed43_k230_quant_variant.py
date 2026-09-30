"""Compile one K230 PTQ variant for the seed-43 model."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nncase
import numpy as np
import onnx
import onnxsim
from PIL import Image


def calibration_samples(directory: Path) -> list[list[np.ndarray]]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    samples = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in suffixes:
            continue
        image = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)[None, ...]
        samples.append([array])
    if not samples:
        raise SystemExit(f"No calibration images found in {directory}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("NoClip", "Kld"), required=True)
    parser.add_argument("--weight-type", choices=("uint8", "int16"), required=True)
    parser.add_argument("--activation-type", choices=("uint8", "int16"), required=True)
    parser.add_argument("--mix-quant", action="store_true")
    args = parser.parse_args()

    model = onnx.load(args.onnx)
    simplified, check = onnxsim.simplify(
        model, overwrite_input_shapes={"images": [1, 3, 224, 224]}
    )
    if not check:
        raise SystemExit("ONNX simplification validation failed")
    temporary = args.output.with_suffix(".simplified.onnx")
    onnx.save(simplified, temporary)

    options = nncase.CompileOptions()
    options.target = "k230"
    options.preprocess = True
    options.swapRB = False
    options.input_shape = [1, 3, 224, 224]
    options.input_type = "uint8"
    options.input_range = [0, 1]
    options.mean = [0.485, 0.456, 0.406]
    options.std = [0.229, 0.224, 0.225]
    options.input_layout = "NCHW"
    options.quant_type = args.activation_type

    compiler = nncase.Compiler(options)
    compiler.import_onnx(temporary.read_bytes(), nncase.ImportOptions())
    samples = calibration_samples(args.calibration)
    ptq = nncase.PTQTensorOptions()
    ptq.samples_count = len(samples)
    ptq.calibrate_method = args.method
    ptq.w_quant_type = args.weight_type
    ptq.quant_type = args.activation_type
    ptq.use_mix_quant = args.mix_quant
    ptq.set_tensor_data(samples)
    compiler.use_ptq(ptq)
    compiler.compile()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(compiler.gencode_tobytes())
    temporary.unlink(missing_ok=True)
    shutil.rmtree("gmodel_dump_dir", ignore_errors=True)
    print(
        f"output={args.output} samples={len(samples)} method={args.method} "
        f"weights={args.weight_type} activations={args.activation_type}"
    )


if __name__ == "__main__":
    main()
