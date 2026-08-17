# KID Knowledge Pipeline

This directory constructs and checks entity-anchored auxiliary knowledge for KID. It is executed before the single student fine-tuning stage.

`scripts/build_kg_sharegpt.py` converts raw multimodal samples to ShareGPT-style examples with teacher knowledge. `scripts/check_kg_quality.py` checks for empty outputs, malformed records, label leakage, and train/test mixing. Prompt templates and task profiles are under `prompts/`.

Keep provider credentials in a local `.env` file only. Teacher outputs and source datasets are experiment artifacts and must remain untracked.
