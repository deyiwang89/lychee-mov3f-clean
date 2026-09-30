"""Validate K230 measurement records before they are cited in the manuscript."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STATES = {"idle", "capture_preprocess", "core_inference", "display_enabled", "continuous_60min"}
REQUIRED = {
    "timestamp", "run_id", "model", "firmware_version", "kmodel_sha256",
    "state", "minute", "power_w",
    "temperature_c", "peak_memory_mb", "fps", "latency_ms",
    "error_count", "restart_count", "measurement_device",
}
REQUIRED_VALUES = {
    "timestamp", "run_id", "model", "firmware_version", "kmodel_sha256",
    "power_w", "temperature_c",
    "peak_memory_mb", "fps", "latency_ms", "error_count",
    "restart_count", "measurement_device",
}
NUMERIC_NONNEGATIVE = {
    "power_w", "temperature_c", "peak_memory_mb", "fps", "latency_ms",
    "error_count", "restart_count",
}


def present(value: str) -> bool:
    return bool(value.strip()) and value.strip().upper() != "NA"


def parse_nonnegative(value: str, field: str, row_number: int, errors: list[str]) -> None:
    try:
        number = float(value)
    except ValueError:
        errors.append(f"row {row_number}: {field} is not numeric")
        return
    if number < 0:
        errors.append(f"row {row_number}: {field} must be nonnegative")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    errors = []
    errors.extend(f"missing column: {field}" for field in sorted(REQUIRED - fields))
    states = {row.get("state", "").strip() for row in rows if row.get("state", "").strip()}
    errors.extend(f"missing state: {state}" for state in sorted(STATES - states))
    continuous = [row for row in rows if row.get("state", "").strip() == "continuous_60min"]
    if continuous and len(continuous) != 60:
        errors.append(f"continuous_60min has {len(continuous)} rows; expected exactly 60")
    if continuous:
        try:
            minutes = sorted(int(row.get("minute", "")) for row in continuous)
        except ValueError:
            errors.append("continuous_60min contains a non-integer minute value")
        else:
            if minutes != list(range(1, 61)):
                errors.append("continuous_60min minute values must be exactly 1 through 60")
    for index, row in enumerate(rows, start=2):
        state = row.get("state", "").strip()
        if state and state not in STATES:
            errors.append(f"row {index}: unknown state: {state}")
            continue
        if state not in STATES:
            continue
        for field in sorted(REQUIRED_VALUES):
            if not present(row.get(field, "")):
                errors.append(f"row {index}: missing required value: {field}")
        for field in sorted(NUMERIC_NONNEGATIVE):
            value = row.get(field, "").strip()
            if present(value):
                parse_nonnegative(value, field, index, errors)
        kmodel_hash = row.get("kmodel_sha256", "").strip()
        if present(kmodel_hash) and (
            len(kmodel_hash) != 64 or any(character not in "0123456789abcdefABCDEF" for character in kmodel_hash)
        ):
            errors.append(f"row {index}: kmodel_sha256 must contain exactly 64 hexadecimal characters")
    report = {"csv": str(args.csv_path), "rows": len(rows), "states": sorted(states), "valid": not errors, "errors": errors}
    output = args.output or args.csv_path.with_suffix(".validation.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
