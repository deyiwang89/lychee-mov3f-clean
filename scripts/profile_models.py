"""Profile rebuttal model parameters and conventional MACs reproducibly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import thop
import torch
from thop import profile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nets import get_model_from_name


MODELS = {
    "gap": "mobilenetv3_gap",
    "gap_matched": "mobilenetv3_gap_matched",
    "wide_gap": "mobilenetv3_gap_wide",
    "gap_gmp": "mobilenetv3_gap_gmp",
    "fractal": "mobilenetv3_fractal",
    "fractal_only": "mobilenetv3_fractal_only",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=224)
    args = parser.parse_args()
    dummy = torch.zeros(1, 3, args.input_size, args.input_size)
    results = {}
    for label, backbone in MODELS.items():
        model = get_model_from_name[backbone](num_classes=10, pretrained=False).eval()
        macs, thop_params = profile(model, inputs=(dummy,), verbose=False)
        exact_params = sum(parameter.numel() for parameter in model.parameters())
        results[label] = {
            "backbone": backbone,
            "parameters": exact_params,
            "thop_parameters": int(thop_params),
            "macs": int(macs),
            "gmacs": macs / 1e9,
            "gflops_assuming_two_flops_per_mac": 2.0 * macs / 1e9,
            "contains_unprofiled_threshold_or_reduction_ops": "fractal" in label,
        }
    report = {
        "tool": "thop.profile",
        "thop_version": getattr(thop, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "input": [1, 3, args.input_size, args.input_size],
        "mac_to_flop_convention": "1 MAC = 2 FLOPs",
        "limitation": "THOP counts registered tensor modules but not all threshold, comparison, logarithm, and reduction operations in channel-wise box counting. Runtime measurements therefore remain necessary.",
        "models": results,
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
