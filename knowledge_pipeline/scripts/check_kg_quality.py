#!/usr/bin/env python3
from __future__ import annotations

"""Lightweight quality checks for KG-enhanced ShareGPT meme data."""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CLASSES = [
    "Targeted Harmful",
    "Sexual Innuendo",
    "General Offense",
    "Dispirited Culture",
]

CLASSIFICATION_PREFIX = "Classify this meme into one of the following categories:"
DEFAULT_CLASSIFICATION_PATTERNS = [
    r"This meme is confirmed harmful\.\s*Classify this meme into one of the following categories:.*",
    r"Identify if this meme is Harmful or Non-harmful\.",
    r"Please determine whether this meme is abusive or not\.\s*Answer with either 'Abusive' or 'Non-abusive'\.",
    r"Classify this meme into one of the following categories:.*",
    r"This meme is confirmed harmful\.\s*What is its target:.*",
    r"Is it hateful\?",
    r"Is it abusive\?",
    r"Analyze this meme with text:.*?(?=\n\s*\[Text Description of the Meme\]|\Z)",
    r"1\.\s*Is it misogynous\?.*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report format and information-density stats for KG JSON files.")
    parser.add_argument("paths", nargs="+", help="ShareGPT JSON files to inspect.")
    parser.add_argument("--min-desc-len", type=int, default=300, help="Warn when description is shorter than this.")
    parser.add_argument("--min-entities", type=int, default=3, help="Warn when entity tag count is below this.")
    parser.add_argument("--min-knowledge", type=int, default=2, help="Warn when knowledge supplement count is below this.")
    parser.add_argument("--max-knowledge", type=int, default=4, help="Warn when knowledge supplement count is above this.")
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class names treated as category-label leakage before the classification instruction.",
    )
    parser.add_argument(
        "--task-profile",
        default=None,
        help=(
            "Optional task-profile JSON. Its forbidden_labels are added to leakage checks, "
            "which is useful for non-ToxiCN benchmarks."
        ),
    )
    parser.add_argument(
        "--classification-instruction",
        default=None,
        help=(
            "Optional exact final classification instruction. Text after this instruction is excluded "
            "from the KG description before leakage checks."
        ),
    )
    parser.add_argument(
        "--only-generated-field",
        action="store_true",
        help="Only inspect items that contain a non-empty generated_kg field.",
    )
    parser.add_argument("--show-examples", type=int, default=8, help="Show up to this many warning examples per file.")
    return parser.parse_args()


def read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def first_user_content(item: dict[str, Any]) -> str:
    messages = item.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    return str(messages[0].get("content", ""))


def read_task_profile(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in task profile {path}")
    return data


def normalize_terms(values: list[str]) -> list[str]:
    seen = set()
    terms = []
    for value in values:
        term = str(value).strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def compile_leakage_pattern(leakage_terms: list[str]) -> re.Pattern[str] | None:
    if not leakage_terms:
        return None
    patterns = []
    for term in leakage_terms:
        escaped = re.escape(term)
        if re.search(r"[A-Za-z0-9]", term):
            escaped = rf"(?<![A-Za-z0-9_-]){escaped}(?![A-Za-z0-9_-])"
        patterns.append(escaped)
    return re.compile("|".join(patterns), flags=re.IGNORECASE)


def extract_description(user_content: str, classification_instruction: str | None) -> str:
    desc = user_content
    if classification_instruction and classification_instruction in desc:
        desc = desc.split(classification_instruction, 1)[0]
    else:
        for pattern in DEFAULT_CLASSIFICATION_PATTERNS:
            match = re.search(pattern, desc, flags=re.DOTALL | re.IGNORECASE)
            if match:
                desc = desc[: match.start()]
                break
    desc = re.sub(r"^\s*<image>\s*", "", desc)
    desc = re.sub(r"\[Text Description of the Meme\]\s*", "", desc)
    return desc.strip()


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{100 * numerator / denominator:.1f}%"


def quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, max(0, int(q * len(values)) - 1))
    return sorted(values)[index]


def has_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def check_file(path: Path, args: argparse.Namespace, leakage_terms: list[str]) -> None:
    raw_data = read_json_list(path)
    data = [
        item
        for item in raw_data
        if not args.only_generated_field or str(item.get("generated_kg", "")).strip()
    ]
    leakage_pattern = compile_leakage_pattern(leakage_terms)

    descs: list[str] = []
    lengths: list[int] = []
    entity_counts: list[int] = []
    knowledge_counts: list[int] = []
    labels: Counter[int] = Counter()
    warnings: Counter[str] = Counter()
    warnings_by_label: dict[int, Counter[str]] = defaultdict(Counter)
    examples: list[tuple[str, str, str]] = []

    for idx, item in enumerate(data):
        label = item.get("label")
        if isinstance(label, int):
            labels[label] += 1

        desc = extract_description(first_user_content(item), args.classification_instruction)
        descs.append(desc)
        desc_len = len(desc)
        entity_count = len(re.findall(r"<[^>\n]+>", desc))
        knowledge_count = len(re.findall(r"\[[^\]\n]+\]", desc))

        lengths.append(desc_len)
        entity_counts.append(entity_count)
        knowledge_counts.append(knowledge_count)

        item_warnings: list[str] = []
        if desc_len < args.min_desc_len:
            item_warnings.append("short_desc")
        if entity_count < args.min_entities:
            item_warnings.append("few_entities")
        if knowledge_count < args.min_knowledge and has_chinese(desc):
            item_warnings.append("few_knowledge_for_chinese")
        if knowledge_count > args.max_knowledge:
            item_warnings.append("too_many_knowledge")
        if leakage_pattern and leakage_pattern.search(desc):
            item_warnings.append("category_label_leakage")
        if re.search(r"</[^>]+>", desc):
            item_warnings.append("xml_closing_tag")
        if re.search(r"\b(Step\s*[12]|Reasoning|Analysis|JSON|category:)\b", desc, flags=re.IGNORECASE):
            item_warnings.append("meta_output")
        if not desc:
            item_warnings.append("empty_description")

        for warning in item_warnings:
            warnings[warning] += 1
            if isinstance(label, int):
                warnings_by_label[label][warning] += 1
        if item_warnings and len(examples) < args.show_examples:
            examples.append((str(item.get("id", idx)), ",".join(item_warnings), " ".join(desc.split())[:260]))

    print(f"\n== {path} ==")
    print(f"items: {len(data)}")
    if args.only_generated_field:
        print(f"source_items: {len(raw_data)}")
    print(f"labels: {dict(sorted(labels.items()))}")
    if lengths:
        print(
            "desc_len: "
            f"mean={statistics.mean(lengths):.1f} "
            f"median={statistics.median(lengths):.1f} "
            f"p95={quantile(lengths, 0.95)} "
            f"min={min(lengths)} max={max(lengths)}"
        )
        print(
            "entities: "
            f"mean={statistics.mean(entity_counts):.2f} "
            f"median={statistics.median(entity_counts):.1f} "
            f"min={min(entity_counts)} max={max(entity_counts)}"
        )
        print(
            "knowledge: "
            f"mean={statistics.mean(knowledge_counts):.2f} "
            f"median={statistics.median(knowledge_counts):.1f} "
            f"min={min(knowledge_counts)} max={max(knowledge_counts)}"
        )

    print("warnings:")
    if warnings:
        for name, count in warnings.most_common():
            print(f"  {name}: {count} ({pct(count, len(data))})")
    else:
        print("  none")

    if warnings_by_label:
        print("warnings_by_label:")
        for label in sorted(warnings_by_label):
            print(f"  {label}: {dict(warnings_by_label[label])}")

    if examples:
        print("examples:")
        for item_id, warning_names, snippet in examples:
            print(f"  {item_id} [{warning_names}] {snippet}")


def main() -> int:
    args = parse_args()
    task_profile = read_task_profile(Path(args.task_profile) if args.task_profile else None)
    profile_forbidden_labels = task_profile.get("forbidden_labels", [])
    if profile_forbidden_labels and not isinstance(profile_forbidden_labels, list):
        raise ValueError("Task profile field 'forbidden_labels' must be a list when provided.")
    leakage_terms = normalize_terms(
        [part.strip() for part in args.classes.split(",") if part.strip()]
        + [str(label) for label in profile_forbidden_labels]
    )
    print(f"leakage_terms: {leakage_terms}")
    for raw_path in args.paths:
        check_file(Path(raw_path), args, leakage_terms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
