#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_file(src: Path, dst: Path) -> tuple[int, int]:
    data = json.loads(src.read_text(encoding="utf-8"))
    converted = 0
    for item in data:
        label = item.get("label")
        if isinstance(label, list):
            if not label:
                raise ValueError(f"Empty label list in item {item.get('id')}")
            item["label"] = int(label[0])
            converted += 1
        elif isinstance(label, bool):
            item["label"] = int(label)
            converted += 1
        elif isinstance(label, int):
            item["label"] = int(label)
        else:
            raise ValueError(f"Unsupported label {label!r} in item {item.get('id')}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(data), converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = [
        "sharegpt_mami_train_instructblip.json",
        "sharegpt_mami_test_instructblip.json",
    ]
    for name in files:
        total, converted = convert_file(input_dir / name, output_dir / name)
        print(f"{name}: total={total} converted={converted}")


if __name__ == "__main__":
    main()
