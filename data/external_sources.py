from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from datasets import load_dataset

from data.loaders import normalize_text
from utils.hf_auth import hf_auth_kwargs


DEFAULT_EXTERNAL_DATASETS: Dict[str, Dict[str, Any]] = {
    "jailbreakbench": {
        "dataset_id": "JailbreakBench/JBB-Behaviors",
        "configs": ["behaviors", None],
        "splits": ["harmful", "train"],
        "kind": "harmful_behavior",
        "required": False,
        "max_records": 200,
    },
    "advbench": {
        "dataset_id": "S3IC/advbench",
        "configs": [None],
        "splits": ["train"],
        "kind": "harmful_behavior",
        "required": False,
        "max_records": 520,
    },
    "lakera_gandalf": {
        "dataset_id": "Lakera/gandalf_ignore_instructions",
        "configs": [None],
        "splits": ["train", "validation", "test"],
        "kind": "prompt_injection",
        "required": False,
        "max_records": 1000,
    },
    "deepset_prompt_injections": {
        "dataset_id": "deepset/prompt-injections",
        "configs": [None],
        "splits": ["train", "test"],
        "kind": "prompt_injection_binary",
        "required": False,
        "max_records": 662,
    },
}


def _first_text(row: Dict[str, Any], fields: Iterable[str]) -> Tuple[str, str]:
    for field in fields:
        text = normalize_text(row.get(field))
        if text:
            return text, field
    return "", ""


def _load_dataset_variant(dataset_id: str, config_name: str | None, split: str):
    if config_name:
        return load_dataset(dataset_id, config_name, split=split, **hf_auth_kwargs())
    return load_dataset(dataset_id, split=split, **hf_auth_kwargs())


def _record_from_row(name: str, dataset_id: str, split: str, kind: str, row_id: int, row: Dict[str, Any]) -> Dict[str, Any] | None:
    text, field = _first_text(row, ["Goal", "goal", "prompt", "instruction", "text", "Behavior", "behavior", "query"])
    if not text:
        return None
    label = row.get("label", "")
    if kind == "prompt_injection_binary":
        try:
            is_attack = int(label) == 1
        except Exception:
            is_attack = normalize_text(label).lower() in {"1", "true", "attack", "injection", "malicious"}
    else:
        is_attack = True
    return {
        "text": text,
        "base_source": dataset_id,
        "base_dataset_key": name,
        "base_example_id": f"{dataset_id}:{split}:{row_id}",
        "base_source_split": split,
        "base_row_id": row_id,
        "base_kind": kind,
        "base_is_attack": is_attack,
        "base_text_field": field,
        "base_topic": normalize_text(row.get("Category") or row.get("category") or row.get("topic")),
        "base_subtopic": normalize_text(row.get("Behavior") or row.get("behavior") or row.get("subtopic")),
        "base_original_source": normalize_text(row.get("Source") or row.get("source")),
        "base_label": normalize_text(label),
        "base_target": normalize_text(row.get("Target") or row.get("target") or row.get("output")),
        "base_similarity": row.get("similarity", ""),
    }


def _cap_by_dataset(rows: List[Dict[str, Any]], max_records: int) -> List[Dict[str, Any]]:
    if max_records <= 0 or len(rows) <= max_records:
        return rows
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('base_source_split', '')}::{row.get('base_topic', '')}::{row.get('base_original_source', '')}"
        buckets[key].append(row)
    selected: List[Dict[str, Any]] = []
    per_bucket = max(1, max_records // max(len(buckets), 1))
    leftovers: List[Dict[str, Any]] = []
    for bucket_rows in buckets.values():
        selected.extend(bucket_rows[:per_bucket])
        leftovers.extend(bucket_rows[per_bucket:])
    selected.extend(leftovers[: max(0, max_records - len(selected))])
    return selected[:max_records]


def load_external_source_pools(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    external_cfg = cfg.get("external_datasets", {})
    if not bool(external_cfg.get("enabled", True)):
        return [], [], [], []
    dataset_cfgs = dict(DEFAULT_EXTERNAL_DATASETS)
    dataset_cfgs.update(external_cfg.get("datasets", {}))
    attack_rows: List[Dict[str, Any]] = []
    benign_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    seen = set()
    for name, ds_cfg in dataset_cfgs.items():
        if not bool(ds_cfg.get("enabled", True)):
            continue
        dataset_id = str(ds_cfg["dataset_id"])
        configs = list(ds_cfg.get("configs", [None]))
        splits = list(ds_cfg.get("splits", ["train"]))
        kind = str(ds_cfg.get("kind", "harmful_behavior"))
        required = bool(ds_cfg.get("required", False))
        max_records = int(ds_cfg.get("max_records", 0))
        loaded_rows: List[Dict[str, Any]] = []
        errors = []
        loaded_any = False
        for config_name in configs:
            for split in splits:
                try:
                    ds = _load_dataset_variant(dataset_id, config_name, split)
                    loaded_any = True
                    for row_id, row in enumerate(ds):
                        rec = _record_from_row(name, dataset_id, split, kind, row_id, row)
                        if rec is None:
                            continue
                        key = (rec["base_source"], rec["text"].lower())
                        if key in seen:
                            continue
                        seen.add(key)
                        loaded_rows.append(rec)
                except Exception as e:
                    errors.append({"dataset": name, "dataset_id": dataset_id, "config": config_name or "", "split": split, "error": str(e)})
            if loaded_any:
                break
        loaded_rows = _cap_by_dataset(loaded_rows, max_records)
        if not loaded_rows and required:
            raise RuntimeError(f"Required external dataset `{name}` ({dataset_id}) did not load any rows. Errors: {errors}")
        error_rows.extend(errors)
        for rec in loaded_rows:
            if bool(rec.get("base_is_attack", True)):
                attack_rows.append(rec)
            else:
                benign_rows.append(rec)
        summary_rows.append({
            "dataset": name,
            "dataset_id": dataset_id,
            "kind": kind,
            "n_rows": len(loaded_rows),
            "n_attack_rows": sum(1 for row in loaded_rows if row.get("base_is_attack", True)),
            "n_benign_rows": sum(1 for row in loaded_rows if not row.get("base_is_attack", True)),
            "topic_counts": dict(Counter(str(row.get("base_topic", "")) for row in loaded_rows)),
            "source_counts": dict(Counter(str(row.get("base_original_source", "")) for row in loaded_rows)),
            "errors": len(errors),
        })
    return attack_rows, benign_rows, summary_rows, error_rows
