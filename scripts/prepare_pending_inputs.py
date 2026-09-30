"""Create reproducible intake templates for external images and K230 records."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PENDING = ROOT / "pending_inputs"
EXTERNAL_CLASSES = (
    "Felt_Leaf", "Deficiency_Leaf", "Leaf_Gall",
    "Fungal_Leaf_Spot", "Curl_Leaf",
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    external_root = PENDING / "external_images"
    for class_name in EXTERNAL_CLASSES:
        class_dir = external_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / "PLACE_IMAGES_HERE.txt").write_text(
            f"Place only independently collected {class_name} images in this folder.\n",
            encoding="utf-8",
        )

    write_csv(
        PENDING / "external_collection_sessions_template.csv",
        ["session_id", "collection_date", "location", "device", "collector",
         "lighting", "background", "weather", "cultivar", "notes"],
        [{"session_id": "session_001"}],
    )

    measurement_fields = [
        "timestamp", "run_id", "model", "firmware_version", "kmodel_sha256",
        "state", "minute", "power_w",
        "temperature_c", "ambient_temperature_c", "peak_memory_mb", "fps",
        "latency_ms", "cpu_util_pct", "kpu_util_pct", "error_count",
        "restart_count", "display_enabled", "measurement_device", "notes",
    ]
    rows = [
        {"run_id": "k230_run_001", "model": "mobilenetv3_fractal", "state": state, "minute": 0}
        for state in ("idle", "capture_preprocess", "core_inference", "display_enabled")
    ]
    rows.extend({
        "run_id": "k230_run_001", "model": "mobilenetv3_fractal",
        "state": "continuous_60min", "minute": minute,
    } for minute in range(1, 61))
    write_csv(PENDING / "k230" / "k230_measurements_template.csv", measurement_fields, rows)

    comparison_fields = [
        "model", "input_size", "firmware_version", "nncase_version",
        "calibration_manifest", "calibration_manifest_sha256",
        "pth_sha256", "onnx_sha256", "kmodel_sha256", "fp32_accuracy",
        "int8_accuracy", "accuracy_change_pp", "pth_size_mb", "onnx_size_mb",
        "kmodel_size_mb", "core_latency_ms", "display_fps", "power_w",
        "temperature_c", "peak_memory_mb", "cpu_execution", "kpu_execution",
        "postprocessing", "display_condition", "measurement_device",
        "measurement_date", "notes",
    ]
    write_csv(
        PENDING / "k230" / "same_device_model_comparison_template.csv",
        comparison_fields,
        [
            {"model": "mobilenetv3_gap", "input_size": "224x224"},
            {"model": "mobilenetv3_fractal", "input_size": "224x224"},
        ],
    )
    print(f"Prepared pending-input templates under: {PENDING}")


if __name__ == "__main__":
    main()
