from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ConversationExample:
    id: str
    group: str
    source: str
    turns: List[Dict[str, str]]
    prompt_text: str
    target: int
    split: str = ""
    family: str = ""
    mode: str = ""
    is_attack: bool = False
    source_is_benign: bool = False
    in_family: bool = True
    family_holdout: bool = False
    is_ood: bool = False
    condition_a: bool = False
    condition_b: bool = False
    shuffled: bool = False
    compressed_single_turn: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["metadata"] = out["metadata"] or {}
        return out


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\x00", " ").split()).strip()


def conversation_to_prompt(turns: List[Dict[str, str]]) -> str:
    chunks = []
    for turn in turns:
        role = turn.get("role", "user").strip().title()
        content = normalize_text(turn.get("content", ""))
        if content:
            chunks.append(f"{role}: {content}")
    chunks.append("Assistant:")
    return "\n\n".join(chunks)


def dedupe_examples(examples: List[ConversationExample]) -> List[ConversationExample]:
    seen = set()
    out: List[ConversationExample] = []
    for ex in examples:
        key = normalize_text(ex.prompt_text).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def subsample_examples(examples: List[ConversationExample], n: int) -> List[ConversationExample]:
    if n <= 0 or len(examples) <= n:
        return list(examples)
    return list(examples[:n])
