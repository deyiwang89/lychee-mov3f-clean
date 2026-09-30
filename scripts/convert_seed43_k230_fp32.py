"""Compile a non-PTQ K230 control model with embedded uint8 preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

import nncase
import onnx
import onnxsim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(args.onnx)
    simplified, check = onnxsim.simplify(model, input_shapes={"images": [1, 3, 224, 224]})
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

    compiler = nncase.Compiler(options)
    compiler.import_onnx(temporary.read_bytes(), nncase.ImportOptions())
    compiler.compile()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(compiler.gencode_tobytes())
    temporary.unlink(missing_ok=True)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
