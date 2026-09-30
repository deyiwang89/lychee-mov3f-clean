"""Split a length-prefixed JPEG bundle into transfer-sized parts."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-per-part", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parts = []
    with args.bundle.open("rb") as source:
        index = 0
        part_index = 0
        while True:
            payloads = []
            for _ in range(args.records_per_part):
                header = source.read(4)
                if not header:
                    break
                if len(header) != 4:
                    raise RuntimeError("truncated record header")
                length = struct.unpack("<I", header)[0]
                payload = source.read(length)
                if len(payload) != length:
                    raise RuntimeError("truncated JPEG payload")
                payloads.append((header, payload))
            if not payloads:
                break
            path = args.output / ("seed43_jpeg_part_%02d.bin" % part_index)
            with path.open("wb") as target:
                for header, payload in payloads:
                    target.write(header)
                    target.write(payload)
            parts.append({"file": path.name, "start_index": index, "count": len(payloads), "bytes": path.stat().st_size})
            index += len(payloads)
            part_index += 1
    metadata = {"samples": index, "records_per_part": args.records_per_part, "parts": parts}
    (args.output / "parts.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
