#!/usr/bin/env python3
from __future__ import annotations

"""Build KG-enhanced ShareGPT data with a local OpenAI-compatible VLM."""

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "your-vision-language-model"
DEFAULT_INPUT = "data/gt/ToxiCN_MM_taskB/sharegpt_toxicn_test_multiclass.json"
DEFAULT_IMAGE_ROOT = "data/image/ToxiCN_MM"
DEFAULT_PROMPT = "knowledge_pipeline/prompts/kg_test_zh.md"
DEFAULT_CATEGORY_MAP = "knowledge_pipeline/prompts/category_definitions/toxicn_taskb_zh.json"
CLASSIFICATION_PREFIX = "Classify this meme into one of the following categories:"

OUTPUT_HINT = (
    "\n\nOutput only the final Step 2 factual description in 1-3 sentences. "
    "Do not output Step 1, reasoning, headings, numbering, JSON, category labels, or extra explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use a vision-language model to generate entity-marked knowledge "
            "descriptions, then inject them into ShareGPT classification data."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input ShareGPT JSON file.")
    parser.add_argument("--output", required=True, help="Output ShareGPT JSON file.")
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT, help="Directory containing meme images.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="KG prompt markdown file.")
    parser.add_argument(
        "--task-profile",
        default=None,
        help=(
            "Optional task-profile JSON used by universal prompts. "
            "Supports task_context, evidence_focus, and forbidden_labels fields."
        ),
    )
    parser.add_argument("--category-map", default=DEFAULT_CATEGORY_MAP, help="Category definition JSON file.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--knowledge-num", type=int, default=2, help="Value used to replace {N} in the prompt.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--start", type=int, default=0, help="Start index in the input file.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of examples to process.")
    parser.add_argument("--save-every", type=int, default=1, help="Save after this many new examples.")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent model requests.")
    parser.add_argument("--resume", action="store_true", help="Reuse already generated items from --output.")
    parser.add_argument(
        "--category-preset",
        default="original",
        help="Classification category list to write into the new user prompt.",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help=(
            "Custom category list, e.g. 'A, B, or C.'. "
            "Overrides --category-preset when provided."
        ),
    )
    parser.add_argument(
        "--classification-instruction",
        default=None,
        help=(
            "Custom final classification instruction to append after the generated KG. "
            "Use this for benchmarks whose prompt is not 'Classify this meme into...'."
        ),
    )
    parser.add_argument(
        "--injection-mode",
        choices=["replace", "append"],
        default="replace",
        help=(
            "How to inject generated evidence into the ShareGPT user message. "
            "'replace' rebuilds the text description from generated output; "
            "'append' preserves the original user description and inserts generated evidence "
            "before the final classification instruction."
        ),
    )
    parser.add_argument(
        "--additional-evidence-heading",
        default="[Additional Objective Evidence]",
        help="Section heading used when --injection-mode append is selected.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the model; write placeholder descriptions for quick format checks.",
    )
    parser.add_argument(
        "--include-original-user",
        action="store_true",
        help=(
            "Append the original user message to the model request as extra context. "
            "By default the model sees only the KG prompt and image."
        ),
    )
    parser.add_argument(
        "--keep-generated-field",
        action="store_true",
        help="Also store the generated description in a top-level generated_kg field.",
    )
    parser.add_argument(
        "--disable-output-hint",
        action="store_true",
        help="Do not append the short instruction that asks the model to output only Step 2.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Set chat_template_kwargs.enable_thinking=true. Default is false.",
    )
    parser.add_argument(
        "--teacher-use-gold-label",
        action="store_true",
        help=(
            "Expose the gold label from the original assistant answer to the teacher model. "
            "Use this for training-set KG generation, not inference-set generation."
        ),
    )
    parser.add_argument(
        "--on-error",
        choices=["fail", "keep-original", "placeholder"],
        default="fail",
        help=(
            "How to handle an item when model generation still fails after retries. "
            "'keep-original' reuses the original text description; 'placeholder' writes a short placeholder."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def read_category_map(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def read_task_profile(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in task profile {path}")
    return data


def format_profile_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(f"- {str(item).strip()}" for item in value if str(item).strip())
    raise ValueError("Task profile list fields must be strings or arrays.")


def apply_task_profile(prompt: str, task_profile: dict[str, Any]) -> str:
    if not task_profile:
        return prompt
    replacements = {
        "{TASK_NAME}": str(task_profile.get("task_name", "")).strip(),
        "{TASK_CONTEXT}": str(task_profile.get("task_context", "")).strip(),
        "{EVIDENCE_FOCUS}": format_profile_list(task_profile.get("evidence_focus")),
        "{FORBIDDEN_LABELS}": format_profile_list(task_profile.get("forbidden_labels")),
    }
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def write_json_atomic(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_classification_instruction(
    user_content: str,
    categories: str | None,
    classification_instruction: str | None = None,
) -> str:
    if classification_instruction:
        return classification_instruction.strip()

    if categories:
        return f"{CLASSIFICATION_PREFIX} {categories.strip()}"

    generic_patterns = [
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
    for pattern in generic_patterns:
        match = re.search(pattern, user_content, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0).strip()

    raise ValueError("Could not find the classification instruction in the original user message.")


def find_classification_instruction_span(
    user_content: str,
    categories: str | None,
    classification_instruction: str | None = None,
) -> tuple[str, int, int]:
    if classification_instruction:
        instruction = classification_instruction.strip()
        start = user_content.find(instruction)
        if start >= 0:
            return instruction, start, start + len(instruction)
        return instruction, len(user_content), len(user_content)

    if categories:
        instruction = f"{CLASSIFICATION_PREFIX} {categories.strip()}"
        start = user_content.find(instruction)
        if start >= 0:
            return instruction, start, start + len(instruction)
        return instruction, len(user_content), len(user_content)

    generic_patterns = [
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
    for pattern in generic_patterns:
        match = re.search(pattern, user_content, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0).strip(), match.start(), match.end()

    raise ValueError("Could not find the classification instruction in the original user message.")


def extract_gold_label(item: dict[str, Any]) -> str:
    messages = item.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Item {item.get('id')} does not contain messages.")
    assistant_content = messages[-1].get("content", "")
    match = re.search(r"category:\s*([^.]+)", assistant_content)
    if not match:
        raise ValueError(f"Could not extract gold label from item {item.get('id')}: {assistant_content}")
    return match.group(1).strip()


def get_preset_config(category_map: dict[str, Any], category_preset: str) -> dict[str, Any] | None:
    if category_preset == "original":
        return None
    preset_config = category_map.get(category_preset)
    if not isinstance(preset_config, dict):
        available = ", ".join(sorted(category_map))
        raise ValueError(f"Unknown --category-preset {category_preset!r}. Available presets: {available}")
    return preset_config


def get_definitions(preset_config: dict[str, Any] | None) -> dict[str, str]:
    if not preset_config:
        return {}
    definitions = preset_config.get("definitions", {})
    if not isinstance(definitions, dict):
        raise ValueError("Category preset field 'definitions' must be an object.")
    return {str(key): str(value) for key, value in definitions.items()}


def format_category_definitions(preset_config: dict[str, Any] | None) -> str:
    definitions = get_definitions(preset_config)
    if not definitions:
        return ""
    names = preset_config.get("category_order", list(definitions)) if preset_config else list(definitions)
    return "\n".join(f"- {name}: {definitions[name]}" for name in names if name in definitions)


def build_kg_prompt(
    prompt_template: str,
    knowledge_num: int,
    append_output_hint: bool,
    preset_config: dict[str, Any] | None,
    gold_label: str | None,
    task_profile: dict[str, Any] | None = None,
) -> str:
    prompt = prompt_template.replace("{N}", str(knowledge_num)).strip()
    prompt = apply_task_profile(prompt, task_profile or {})
    prompt = prompt.replace("{CATEGORY_DEFINITIONS}", format_category_definitions(preset_config))
    if gold_label:
        definitions = get_definitions(preset_config)
        prompt = prompt.replace("{GOLD_LABEL}", gold_label)
        prompt = prompt.replace("{GOLD_LABEL_DESCRIPTION}", definitions.get(gold_label, ""))
    if append_output_hint:
        prompt += OUTPUT_HINT
    return prompt


def make_model_content(
    prompt: str,
    image_paths: list[Path],
    original_user_content: str | None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})

    if original_user_content:
        content.append(
            {
                "type": "text",
                "text": (
                    "\n\n以下是原始样本中的用户输入，仅作为 OCR/上下文参考；"
                    "不要复制其中的分类问题，也不要输出分类标签：\n"
                    f"{original_user_content}"
                ),
            }
        )
    return content


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    image_paths: list[Path],
    original_user_content: str | None,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    timeout: float,
) -> tuple[str, dict[str, int | None]]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": make_model_content(prompt, image_paths, original_user_content),
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            }
        },
    )
    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("Model returned an empty response.")

    usage = response.usage
    usage_stats = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "image_tokens": None,
        "finish_reason": response.choices[0].finish_reason,
    }
    prompt_details = getattr(usage, "prompt_tokens_details", None) if usage else None
    if prompt_details is not None:
        usage_stats["image_tokens"] = getattr(prompt_details, "image_tokens", None)

    return clean_generated_description(text), usage_stats


def clean_generated_description(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:text|markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # If the model still emits sections, keep the final factual-description part.
    step2_match = re.search(r"(?:Step\s*2|事实描述)[^\n]*\n+(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if step2_match:
        text = step2_match.group(1).strip()

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^#{1,6}\s+", stripped):
            continue
        if re.match(r"^(?:Step\s*1|深度解析|元素拆解|文本提取|知识检索)\b", stripped, flags=re.IGNORECASE):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def resolve_categories(args: argparse.Namespace, preset_config: dict[str, Any] | None) -> str | None:
    if args.categories:
        return args.categories
    if args.category_preset == "original":
        return None
    if not preset_config:
        return None
    categories = preset_config.get("classification_categories")
    if not isinstance(categories, str) or not categories.strip():
        raise ValueError("Category preset field 'classification_categories' must be a non-empty string.")
    return categories


def build_new_user_content(
    generated_description: str,
    original_user_content: str,
    categories: str | None,
    classification_instruction: str | None,
    injection_mode: str = "replace",
    additional_evidence_heading: str = "[Additional Objective Evidence]",
) -> str:
    classification_instruction, instruction_start, instruction_end = find_classification_instruction_span(
        original_user_content, categories, classification_instruction
    )
    if injection_mode == "append":
        original_prefix = original_user_content[:instruction_start].rstrip()
        original_suffix = original_user_content[instruction_end:].strip()
        sections = [
            original_prefix,
            additional_evidence_heading.strip(),
            generated_description.strip(),
            classification_instruction.strip(),
        ]
        if original_suffix:
            sections.append(original_suffix)
        return "\n\n".join(section for section in sections if section)

    return (
        "<image>\n\n"
        "[Text Description of the Meme]\n"
        f"{generated_description.strip()}\n\n"
        f"{classification_instruction}"
    )


def extract_original_description(user_content: str) -> str:
    match = re.search(
        r"\[Text Description of the Meme\]\s*(.*?)(?:\n\s*Classify this meme into one of the following categories:|\Z)",
        user_content,
        flags=re.DOTALL,
    )
    if not match:
        return ""
    return match.group(1).strip()


def fallback_generated_description(on_error: str, original_user_content: str) -> str:
    if on_error == "keep-original":
        original_description = extract_original_description(original_user_content)
        if original_description:
            return original_description
    return "<KG_DESCRIPTION_UNAVAILABLE>[teacher model generation failed for this item]"


def collect_image_paths(item: dict[str, Any], image_root: Path) -> list[Path]:
    images = item.get("images")
    if not images:
        raise ValueError(f"Item {item.get('id')} does not contain images.")

    paths = []
    for image in images:
        image_path = Path(image)
        if not image_path.is_absolute() and not image_path.exists():
            image_path = image_root / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for item {item.get('id')}: {image_path}")
        paths.append(image_path)
    return paths


def transform_item(
    item: dict[str, Any],
    generated_description: str,
    keep_generated_field: bool,
    categories: str | None,
    classification_instruction: str | None,
    injection_mode: str,
    additional_evidence_heading: str,
) -> dict[str, Any]:
    messages = item.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Item {item.get('id')} does not contain messages.")
    if messages[0].get("role") != "user":
        raise ValueError(f"First message of item {item.get('id')} is not a user message.")

    new_item = dict(item)
    new_messages = [dict(message) for message in messages]
    new_messages[0]["content"] = build_new_user_content(
        generated_description,
        messages[0]["content"],
        categories,
        classification_instruction,
        injection_mode,
        additional_evidence_heading,
    )
    new_item["messages"] = new_messages

    if keep_generated_field:
        new_item["generated_kg"] = generated_description
    return new_item


def selected_indices(total: int, start: int, limit: int | None) -> set[int]:
    if start < 0:
        raise ValueError("--start must be non-negative.")
    end = total if limit is None else min(total, start + limit)
    return set(range(start, end))


def can_resume_existing_item(item: dict[str, Any], keep_generated_field: bool) -> bool:
    if keep_generated_field:
        return bool(str(item.get("generated_kg", "")).strip())
    user_content = first_message_content(item)
    return bool(user_content and "[Text Description of the Meme]" in user_content)


def first_message_content(item: dict[str, Any]) -> str:
    messages = item.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    return str(messages[0].get("content", ""))


def empty_usage_stats() -> dict[str, int | str | None]:
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "image_tokens": None,
        "finish_reason": None,
    }


def format_done_log(
    elapsed: float,
    generated_description: str,
    usage_stats: dict[str, int | str | None],
) -> str:
    knowledge_count = len(re.findall(r"\[[^\]]+\]", generated_description))
    completion_tokens = usage_stats.get("completion_tokens")
    total_tokens = usage_stats.get("total_tokens")
    completion_tps = completion_tokens / elapsed if isinstance(completion_tokens, int) else None
    total_tps = total_tokens / elapsed if isinstance(total_tokens, int) else None
    token_log = ""
    if isinstance(total_tokens, int):
        token_log = (
            f"; prompt_tokens={usage_stats.get('prompt_tokens')}"
            f"; image_tokens={usage_stats.get('image_tokens')}"
            f"; completion_tokens={completion_tokens}"
            f"; total_tokens={total_tokens}"
            f"; finish_reason={usage_stats.get('finish_reason')}"
            f"; completion_tps_e2e={completion_tps:.2f}"
            f"; total_tps_e2e={total_tps:.2f}"
        )
    elif usage_stats.get("finish_reason"):
        token_log = f"; finish_reason={usage_stats.get('finish_reason')}"
    return f"done in {elapsed:.2f}s; chars={len(generated_description)}; knowledge={knowledge_count}{token_log}"


def generate_one_item(
    idx: int,
    item: dict[str, Any],
    args: argparse.Namespace,
    image_root: Path,
    prompt_template: str,
    preset_config: dict[str, Any] | None,
    categories: str | None,
    task_profile: dict[str, Any] | None,
) -> tuple[int, dict[str, Any], str, float, dict[str, int | str | None]]:
    image_paths = collect_image_paths(item, image_root)
    original_user_content = item["messages"][0]["content"]
    gold_label = extract_gold_label(item) if args.teacher_use_gold_label else None
    prompt = build_kg_prompt(
        prompt_template=prompt_template,
        knowledge_num=args.knowledge_num,
        append_output_hint=not args.disable_output_hint,
        preset_config=preset_config,
        gold_label=gold_label,
        task_profile=task_profile,
    )
    item_started_at = time.monotonic()

    if args.dry_run:
        generated_description = "<KG_DESCRIPTION_PLACEHOLDER>[dry-run placeholder]"
        usage_stats = empty_usage_stats()
    else:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        last_error: Exception | None = None
        for attempt in range(1, args.retries + 1):
            try:
                generated_description, usage_stats = call_model(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    image_paths=image_paths,
                    original_user_content=original_user_content if args.include_original_user else None,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    enable_thinking=args.enable_thinking,
                    timeout=args.timeout,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= args.retries:
                    if args.on_error == "fail":
                        raise
                    generated_description = fallback_generated_description(args.on_error, original_user_content)
                    usage_stats = empty_usage_stats()
                    usage_stats["finish_reason"] = f"fallback:{type(exc).__name__}"
                    print(
                        f"[{idx + 1}] fallback for {item.get('id')} after {attempt} attempts: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                sleep_time = args.retry_sleep * attempt
                print(
                    f"[{idx + 1}] attempt {attempt} failed for {item.get('id')}: {exc}; "
                    f"retrying in {sleep_time:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(sleep_time)
        else:
            raise RuntimeError(f"Generation failed for {item.get('id')}: {last_error}")

    elapsed = time.monotonic() - item_started_at
    new_item = transform_item(
        item,
        generated_description,
        args.keep_generated_field,
        categories,
        args.classification_instruction,
        args.injection_mode,
        args.additional_evidence_heading,
    )
    return idx, new_item, generated_description, elapsed, usage_stats


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    image_root = Path(args.image_root)
    prompt_path = Path(args.prompt)
    task_profile_path = Path(args.task_profile) if args.task_profile else None
    category_map_path = Path(args.category_map)

    source_data = read_json(input_path)
    category_map = read_category_map(category_map_path)
    task_profile = read_task_profile(task_profile_path)
    preset_config = get_preset_config(category_map, args.category_preset)
    prompt_template = prompt_path.read_text(encoding="utf-8")
    categories = resolve_categories(args, preset_config)

    indices_to_process = selected_indices(len(source_data), args.start, args.limit)
    output_data: list[dict[str, Any] | None] = [None] * len(source_data)

    if args.resume and output_path.exists():
        existing_data = read_json(output_path)
        existing_by_id = {item.get("id"): item for item in existing_data}
        for idx, item in enumerate(source_data):
            existing_item = existing_by_id.get(item.get("id"))
            if existing_item is not None and can_resume_existing_item(existing_item, args.keep_generated_field):
                output_data[idx] = existing_item

    processed_since_save = 0
    total_selected = len(indices_to_process)
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1.")
    print(f"Loaded {len(source_data)} examples; selected {total_selected} for KG generation.", flush=True)

    tasks: list[tuple[int, dict[str, Any]]] = []
    for idx, item in enumerate(source_data):
        if idx not in indices_to_process:
            output_data[idx] = output_data[idx] or item
            continue

        if output_data[idx] is not None:
            print(f"[{idx + 1}/{len(source_data)}] skip existing {item.get('id')}", flush=True)
            continue

        tasks.append((idx, item))

    def store_result(
        result_idx: int,
        new_item: dict[str, Any],
        generated_description: str,
        elapsed: float,
        usage_stats: dict[str, int | str | None],
    ) -> None:
        nonlocal processed_since_save
        output_data[result_idx] = new_item
        print(f"  {format_done_log(elapsed, generated_description, usage_stats)}", flush=True)
        processed_since_save += 1
        if processed_since_save >= args.save_every:
            write_json_atomic(output_path, [entry for entry in output_data if entry is not None])
            processed_since_save = 0

    if args.concurrency == 1:
        for idx, item in tasks:
            print(f"[{idx + 1}/{len(source_data)}] generating {item.get('id')}", flush=True)
            result = generate_one_item(
                idx=idx,
                item=item,
                args=args,
                image_root=image_root,
                prompt_template=prompt_template,
                preset_config=preset_config,
                categories=categories,
                task_profile=task_profile,
            )
            store_result(*result)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_task = {}
            for idx, item in tasks:
                print(f"[{idx + 1}/{len(source_data)}] submit {item.get('id')}", flush=True)
                future = executor.submit(
                    generate_one_item,
                    idx,
                    item,
                    args,
                    image_root,
                    prompt_template,
                    preset_config,
                    categories,
                    task_profile,
                )
                future_to_task[future] = (idx, item)

            for future in as_completed(future_to_task):
                idx, item = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[{idx + 1}/{len(source_data)}] failed {item.get('id')}: {exc}", file=sys.stderr)
                    raise
                print(f"[{idx + 1}/{len(source_data)}] completed {item.get('id')}", flush=True)
                store_result(*result)

    final_data = [entry if entry is not None else item for entry, item in zip(output_data, source_data)]
    write_json_atomic(output_path, final_data)
    print(f"Saved {len(final_data)} examples to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
