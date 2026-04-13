from __future__ import annotations

import ast
from collections import defaultdict
from typing import Any, Dict, List

from datasets import load_dataset

from data.loaders import normalize_text


DEFAULT_REAL_ATTACK_CORPUS_ID = "mvrcii/safety-harmful"


def _messages_from_text(value: Any) -> List[Dict[str, str]]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                raw = ast.literal_eval(text)
            except Exception:
                return [{"role": "user", "content": normalize_text(text)}]
        else:
            return [{"role": "user", "content": normalize_text(text)}]
    else:
        return []
    turns = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = normalize_text(item.get("role") or item.get("from") or "user").lower()
        content = normalize_text(item.get("content") or item.get("value") or item.get("text"))
        if content:
            turns.append({"role": "assistant" if role == "assistant" else "user", "content": content})
    return turns


def _user_prompt_from_row(row: Dict[str, Any]) -> str:
    for key in ["prompt", "instruction", "goal", "query"]:
        text = normalize_text(row.get(key))
        if text:
            return text
    turns = _messages_from_text(row.get("text"))
    for turn in turns:
        if turn["role"] == "user":
            return turn["content"]
    return normalize_text(row.get("text"))


def _stratified_cap(rows: List[Dict[str, Any]], max_records: int) -> List[Dict[str, Any]]:
    if max_records <= 0 or len(rows) <= max_records:
        return rows
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('topic', 'unknown')}::{row.get('original_source', 'unknown')}"
        buckets[key].append(row)
    selected: List[Dict[str, Any]] = []
    per_bucket = max(1, max_records // max(len(buckets), 1))
    leftovers: List[Dict[str, Any]] = []
    for bucket_rows in buckets.values():
        selected.extend(bucket_rows[:per_bucket])
        leftovers.extend(bucket_rows[per_bucket:])
    remaining = max_records - len(selected)
    if remaining > 0:
        selected.extend(leftovers[:remaining])
    return selected[:max_records]


def load_real_attack_corpus(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    corpus_cfg = cfg.get("real_attack_corpus", {})
    if not bool(corpus_cfg.get("enabled", True)):
        return []
    dataset_id = str(corpus_cfg.get("dataset_id", DEFAULT_REAL_ATTACK_CORPUS_ID))
    split = str(corpus_cfg.get("split", "train"))
    max_records = int(corpus_cfg.get("max_records", 0))
    min_chars = int(corpus_cfg.get("min_chars", 12))
    try:
        ds = load_dataset(dataset_id, split=split)
    except Exception as e:
        if bool(corpus_cfg.get("required", True)):
            raise RuntimeError(
                f"Failed to load real attack corpus `{dataset_id}` split `{split}`. "
                "Install/upgrade datasets, check network access, and verify the dataset is available."
            ) from e
        return []

    rows: List[Dict[str, Any]] = []
    seen = set()
    for row_id, row in enumerate(ds):
        prompt = _user_prompt_from_row(row)
        if len(prompt) < min_chars:
            continue
        key = prompt.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "text": prompt,
            "base_source": dataset_id,
            "base_example_id": f"{dataset_id}:{split}:{row_id}",
            "base_source_split": split,
            "base_row_id": row_id,
            "base_topic": normalize_text(row.get("topic")),
            "base_subtopic": normalize_text(row.get("subtopic")),
            "base_original_source": normalize_text(row.get("source")),
            "base_label": normalize_text(row.get("label")),
            "base_response_refusal_label": normalize_text(row.get("response_refusal_label")),
            "base_turn_type": normalize_text(row.get("turn_type")),
            "base_final_turn_role": normalize_text(row.get("final_turn_role")),
        })
    return _stratified_cap(rows, max_records)
