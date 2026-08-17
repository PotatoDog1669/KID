#!/usr/bin/env python3
from __future__ import annotations

"""Export samples where one prediction file regresses against another."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CLASSES = [
    "Targeted Harmful",
    "Sexual Innuendo",
    "General Offense",
    "Dispirited Culture",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline/current prediction CSV files and export selected "
            "ShareGPT samples for prompt debugging."
        )
    )
    parser.add_argument(
        "--baseline-preds",
        default="results/data/toxicn_mm_taskb/qwen2.5vl_oursft.csv",
        help="Baseline prediction CSV with label and pred_label columns.",
    )
    parser.add_argument(
        "--current-preds",
        default="results/data/toxicn_mm_taskb/predictions_20260703_151627.csv",
        help="Current prediction CSV with label and pred_label columns.",
    )
    parser.add_argument(
        "--source-json",
        default="data/gt/ToxiCN_MM_taskB/sharegpt_toxicn_test_multiclass.json",
        help="Source ShareGPT JSON to subset, usually the original non-KG test file.",
    )
    parser.add_argument(
        "--current-kg-json",
        default="data/gt/ToxiCN_MM_taskB/kg_qwen36_en/sharegpt_toxicn_test_multiclass.json",
        help="Optional current KG JSON used only to include descriptions in the report.",
    )
    parser.add_argument(
        "--baseline-kg-json",
        default="data/gt/ToxiCN_MM_taskB/kg/sharegpt_toxicn_test_multiclass.json",
        help="Optional baseline KG JSON used only to include descriptions in the report.",
    )
    parser.add_argument(
        "--output-json",
        default="data/gt/ToxiCN_MM_taskB/regression_debug/sharegpt_toxicn_test_regressions.json",
        help="Output ShareGPT JSON subset.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/gt/ToxiCN_MM_taskB/regression_debug/prediction_regressions.csv",
        help="Output CSV report.",
    )
    parser.add_argument(
        "--mode",
        choices=["regressions", "fixes", "changed", "all_disagreements"],
        default="regressions",
        help=(
            "regressions: baseline correct, current wrong; "
            "fixes: baseline wrong, current correct; "
            "changed: different predictions; "
            "all_disagreements: either model is wrong or predictions differ."
        ),
    )
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class names in label-id order.",
    )
    return parser.parse_args()


def read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def maybe_read_json_list(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    return read_json_list(path)


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def extract_description(item: dict[str, Any] | None) -> str:
    if not item:
        return ""
    messages = item.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    content = str(messages[0].get("content", ""))
    marker = "Classify this meme into one of the following categories:"
    desc = content.split(marker, 1)[0]
    desc = re.sub(r"^\s*<image>\s*", "", desc)
    desc = re.sub(r"\[Text Description of the Meme\]\s*", "", desc)
    return " ".join(desc.split())


def class_name(label_id: int, classes: list[str]) -> str:
    if 0 <= label_id < len(classes):
        return classes[label_id]
    return str(label_id)


def selected_mask(args: argparse.Namespace, base: pd.DataFrame, cur: pd.DataFrame) -> pd.Series:
    base_ok = base["label"].eq(base["pred_label"])
    cur_ok = cur["label"].eq(cur["pred_label"])
    changed = ~base["pred_label"].eq(cur["pred_label"])

    if args.mode == "regressions":
        return base_ok & ~cur_ok
    if args.mode == "fixes":
        return ~base_ok & cur_ok
    if args.mode == "changed":
        return changed
    return changed | ~base_ok | ~cur_ok


def validate_predictions(base: pd.DataFrame, cur: pd.DataFrame, source_len: int) -> None:
    required = {"label", "pred_label"}
    for name, df in [("baseline", base), ("current", cur)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} predictions missing required columns: {sorted(missing)}")
    if len(base) != len(cur):
        raise ValueError(f"Prediction lengths differ: baseline={len(base)}, current={len(cur)}")
    if len(base) != source_len:
        raise ValueError(f"Prediction/source lengths differ: predictions={len(base)}, source={source_len}")
    if not base["label"].equals(cur["label"]):
        raise ValueError("Baseline/current label columns are not identical by row.")


def main() -> int:
    args = parse_args()
    classes = [part.strip() for part in args.classes.split(",") if part.strip()]

    baseline_preds = pd.read_csv(args.baseline_preds)
    current_preds = pd.read_csv(args.current_preds)
    source = read_json_list(Path(args.source_json))
    current_kg = maybe_read_json_list(Path(args.current_kg_json))
    baseline_kg = maybe_read_json_list(Path(args.baseline_kg_json))

    validate_predictions(baseline_preds, current_preds, len(source))

    mask = selected_mask(args, baseline_preds, current_preds)
    indices = [int(i) for i, selected in enumerate(mask.tolist()) if selected]
    subset = [source[i] for i in indices]

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    write_json(out_json, subset)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "id",
                "images",
                "label",
                "label_name",
                "baseline_pred",
                "baseline_pred_name",
                "current_pred",
                "current_pred_name",
                "baseline_description",
                "current_description",
            ],
        )
        writer.writeheader()
        for i in indices:
            label = int(current_preds.loc[i, "label"])
            base_pred = int(baseline_preds.loc[i, "pred_label"])
            cur_pred = int(current_preds.loc[i, "pred_label"])
            item = source[i]
            writer.writerow(
                {
                    "index": i,
                    "id": item.get("id", ""),
                    "images": " ".join(str(x) for x in item.get("images", [])),
                    "label": label,
                    "label_name": class_name(label, classes),
                    "baseline_pred": base_pred,
                    "baseline_pred_name": class_name(base_pred, classes),
                    "current_pred": cur_pred,
                    "current_pred_name": class_name(cur_pred, classes),
                    "baseline_description": extract_description(baseline_kg[i] if baseline_kg else None),
                    "current_description": extract_description(current_kg[i] if current_kg else None),
                }
            )

    print(f"mode={args.mode}")
    print(f"selected={len(indices)}")
    print(f"json={out_json}")
    print(f"csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
