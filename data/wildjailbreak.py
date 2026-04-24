from __future__ import annotations

from typing import Any, Dict, List

from datasets import load_dataset

from data.loaders import ConversationExample, conversation_to_prompt, dedupe_examples, normalize_text
from utils.hf_auth import hf_auth_kwargs


WILDJAILBREAK_ID = "allenai/wildjailbreak"


def _load_config(config_name: str):
    try:
        return load_dataset(WILDJAILBREAK_ID, config_name, delimiter="\t", keep_default_na=False, **hf_auth_kwargs())
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "401" in msg or "403" in msg or "authorized" in msg.lower():
            raise RuntimeError(
                "WildJailbreak is gated. Accept the AI2 Responsible Use Guidelines "
                "for allenai/wildjailbreak on Hugging Face, authenticate with `huggingface-cli login`, "
                "then rerun."
            ) from e
        raise RuntimeError(f"Failed to load allenai/wildjailbreak config `{config_name}`: {e}") from e


def _row_text(row: Dict[str, Any]) -> tuple[str, str]:
    for field in ["adversarial", "vanilla", "prompt", "instruction"]:
        text = normalize_text(row.get(field))
        if text:
            return text, field
    return "", ""


def load_wildjailbreak() -> List[ConversationExample]:
    examples: List[ConversationExample] = []
    for config_name in ["train", "eval"]:
        ds_dict = _load_config(config_name)
        for split_name, ds in ds_dict.items():
            for row_id, row in enumerate(ds):
                text, field = _row_text(row)
                if not text:
                    continue
                data_type = normalize_text(row.get("data_type"))
                if data_type in {"vanilla_harmful", "adversarial_harmful"}:
                    target = 1
                    source_is_benign = False
                    mode = "jailbreak" if "adversarial" in data_type else "harmful_direct"
                elif data_type in {"vanilla_benign", "adversarial_benign"}:
                    target = 0
                    source_is_benign = True
                    mode = "benign_contrastive"
                else:
                    continue
                turns = [{"role": "user", "content": text}]
                examples.append(
                    ConversationExample(
                        id=f"wildjailbreak:{config_name}:{split_name}:{row_id}",
                        group="ordinary_harmful_benign",
                        source=WILDJAILBREAK_ID,
                        turns=turns,
                        prompt_text=conversation_to_prompt(turns),
                        target=target,
                        mode=mode,
                        is_attack=target == 1,
                        source_is_benign=source_is_benign,
                        metadata={
                            "source_config": config_name,
                            "source_split": split_name,
                            "row_id": row_id,
                            "data_type": data_type,
                            "tactics": row.get("tactics", ""),
                            "completion": row.get("completion", ""),
                            "source_text_field": field,
                        },
                    )
                )
    return dedupe_examples(examples)
