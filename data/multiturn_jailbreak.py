from __future__ import annotations

import re
from typing import Any, Dict, List

from datasets import load_dataset

from data.loaders import ConversationExample, conversation_to_prompt, dedupe_examples, normalize_text
from utils.hf_auth import hf_auth_kwargs


MHJ_ID = "ScaleAI/mhj"
MHJ_CONVERSATION_FILES = [
    "harmbench_behaviors.csv",
]


def _extract_turns(row: Dict[str, Any]) -> List[Dict[str, str]]:
    for field in ["messages", "conversation", "conversations", "turns"]:
        val = row.get(field)
        if isinstance(val, list) and val:
            turns = []
            for msg in val:
                if not isinstance(msg, dict):
                    continue
                role = normalize_text(msg.get("role") or msg.get("from") or "user").lower()
                content = normalize_text(msg.get("content") or msg.get("value") or msg.get("text"))
                if content:
                    turns.append({"role": "assistant" if role == "assistant" else "user", "content": content})
            if turns:
                return turns
    prompts = []
    for field in ["prompt_sequence", "prompts", "jailbreak_turns"]:
        val = row.get(field)
        if isinstance(val, list):
            prompts = [normalize_text(x) for x in val if normalize_text(x)]
            break
    if not prompts:
        for field in ["prompt", "jailbreak_prompt", "single_turn_prompt", "goal"]:
            text = normalize_text(row.get(field))
            if text:
                prompts = [text]
                break
    if prompts:
        return [{"role": "user", "content": text} for text in prompts]

    message_fields = []
    for key, value in row.items():
        match = re.fullmatch(r"message_(\d+)", str(key))
        if not match:
            continue
        text = normalize_text(value)
        if text:
            message_fields.append((int(match.group(1)), text))
    message_fields.sort(key=lambda item: item[0])
    if message_fields:
        return [
            {"role": "user" if idx % 2 == 0 else "assistant", "content": text}
            for idx, text in message_fields
        ]
    return []


def _load_scaleai_mhj_rows(dataset_id: str) -> tuple[Dict[str, Any], str]:
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
    except Exception as e:
        raise RuntimeError("ScaleAI/mhj direct CSV loading requires `huggingface_hub` and `pandas`. Install requirements.txt.") from e

    errors = []
    for filename in MHJ_CONVERSATION_FILES:
        try:
            path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=filename, **hf_auth_kwargs())
            df = pd.read_csv(path, keep_default_na=False)
            return {"train": df.to_dict(orient="records")}, filename
        except Exception as e:
            errors.append(f"{filename}: {e}")
    raise RuntimeError(
        "Failed to load ScaleAI/mhj conversation CSV directly. Tried "
        f"{MHJ_CONVERSATION_FILES}. Errors: {' | '.join(errors)}"
    )


def load_multiturn_jailbreak(cfg: Dict[str, Any]) -> tuple[List[ConversationExample], str]:
    dataset_id = cfg.get("echo_chamber_dataset") or cfg.get("multiturn_dataset", "ScaleAI/mhj")
    note = "Echo Chamber was not configured as a ready-to-load dataset; using ScaleAI/mhj as public multi-turn substitute."
    if cfg.get("echo_chamber_dataset"):
        note = f"Using configured Echo Chamber-compatible dataset `{dataset_id}`."
    if dataset_id == MHJ_ID:
        ds_dict, loaded_file = _load_scaleai_mhj_rows(dataset_id)
        note = f"{note} Loaded `{loaded_file}` directly because the HF auto-builder mixes incompatible repo files."
    else:
        try:
            ds_dict = load_dataset(dataset_id, **hf_auth_kwargs())
        except Exception as e:
            raise RuntimeError(f"Failed to load multi-turn jailbreak dataset `{dataset_id}`: {e}") from e

    examples: List[ConversationExample] = []
    for split_name, ds in ds_dict.items():
        for row_id, row in enumerate(ds):
            turns = _extract_turns(row)
            if not turns:
                continue
            family = normalize_text(row.get("attack_family") or row.get("tactic") or row.get("category") or row.get("goal_id")) or "unknown"
            examples.append(
                ConversationExample(
                    id=f"multiturn:{dataset_id}:{split_name}:{row_id}",
                    group="ordinary_multiturn_jailbreak",
                    source=dataset_id,
                    turns=turns,
                    prompt_text=conversation_to_prompt(turns),
                    target=1,
                    family=family,
                    mode="ordinary_multiturn",
                    is_attack=True,
                    source_is_benign=False,
                    metadata={
                        "source_split": split_name,
                        "row_id": row_id,
                        "goal": row.get("goal", row.get("behavior", "")),
                        "goal_id": row.get("goal_id", row.get("behavior_id", "")),
                        "comments": row.get("comments", row.get("comment", "")),
                        "single_turn_prompt": row.get("single_turn_prompt", row.get("prompt", "")),
                        "adapter_note": note,
                    },
                )
            )
    return dedupe_examples(examples), note
