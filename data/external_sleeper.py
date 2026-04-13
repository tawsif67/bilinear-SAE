from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import load_dataset

from data.loaders import ConversationExample, conversation_to_prompt, normalize_text


TEXT_FIELDS = (
    "prompt",
    "input",
    "question",
    "request",
    "text",
    "user_prompt",
    "deployment_prompt",
    "conversation",
)


def _iter_local_rows(path: Path) -> Iterable[Dict[str, Any]]:
    files: List[Path]
    if path.is_dir():
        files = sorted([p for p in path.rglob("*") if p.suffix.lower() in {".jsonl", ".json", ".csv"}])
    else:
        files = [path]
    for file in files:
        suffix = file.suffix.lower()
        if suffix == ".jsonl":
            with file.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict):
                        row["_source_file"] = str(file)
                        row["_line_no"] = line_no
                        yield row
        elif suffix == ".json":
            data = json.loads(file.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
            for idx, row in enumerate(rows):
                if isinstance(row, dict):
                    row["_source_file"] = str(file)
                    row["_row_id"] = idx
                    yield row
        elif suffix == ".csv":
            with file.open("r", encoding="utf-8", newline="") as f:
                for idx, row in enumerate(csv.DictReader(f)):
                    row["_source_file"] = str(file)
                    row["_row_id"] = idx
                    yield dict(row)


def _turns_from_row(row: Dict[str, Any]) -> List[Dict[str, str]]:
    for key in ("messages", "turns", "conversation"):
        value = row.get(key)
        if isinstance(value, list):
            turns = []
            for item in value:
                if not isinstance(item, dict):
                    continue
                role = normalize_text(item.get("role") or item.get("speaker") or "user").lower() or "user"
                content = normalize_text(item.get("content") or item.get("text") or item.get("message"))
                if content:
                    turns.append({"role": role, "content": content})
            if turns:
                return turns
    for field in TEXT_FIELDS:
        text = normalize_text(row.get(field))
        if text:
            return [{"role": "user", "content": text}]
    return []


def _target_from_row(row: Dict[str, Any]) -> int:
    for field in ("target", "label", "is_attack", "is_backdoor", "is_sleeper", "triggered"):
        if field not in row:
            continue
        value = row.get(field)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and value in {0, 1}:
            return int(value)
        text = normalize_text(value).lower()
        if text in {"1", "true", "attack", "backdoor", "triggered", "harmful", "yes"}:
            return 1
        if text in {"0", "false", "clean", "safe", "benign", "no"}:
            return 0
    return 1


def _example_from_row(row: Dict[str, Any], idx: int, source_name: str) -> ConversationExample | None:
    turns = _turns_from_row(row)
    if not turns:
        return None
    family = normalize_text(row.get("family") or row.get("task") or row.get("trigger_type") or "external_sleeper")
    target = _target_from_row(row)
    split = normalize_text(row.get("split"))
    prompt_text = conversation_to_prompt(turns)
    return ConversationExample(
        id=f"{source_name}:{idx}",
        group="external_sleeper_validation",
        source=source_name,
        turns=turns,
        prompt_text=prompt_text,
        target=target,
        split=split,
        family=family,
        mode=normalize_text(row.get("mode") or row.get("condition") or "external"),
        is_attack=bool(target),
        source_is_benign=not bool(target),
        in_family=False,
        family_holdout=True,
        is_ood=True,
        metadata={
            "external_validation": True,
            "source_file": row.get("_source_file", ""),
            "source_row_id": row.get("_row_id", row.get("_line_no", idx)),
            "original_fields": sorted(k for k in row.keys() if not str(k).startswith("_")),
        },
    )


def load_external_sleeper_benchmark(cfg: Dict[str, Any]) -> Tuple[List[ConversationExample], Dict[str, Any], List[Dict[str, Any]]]:
    sleeper_cfg = cfg.get("external_sleeper", {})
    if not bool(sleeper_cfg.get("enabled", False)):
        return [], {"enabled": False, "n": 0, "note": "external sleeper validation disabled"}, []

    required = bool(sleeper_cfg.get("required", False))
    local_path = sleeper_cfg.get("local_path")
    hf_dataset_id = sleeper_cfg.get("hf_dataset_id")
    source_name = normalize_text(sleeper_cfg.get("source_name") or "external_sleeper")
    max_records = int(sleeper_cfg.get("max_records", 1000))
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    try:
        if local_path:
            rows = list(_iter_local_rows(Path(local_path)))
        elif hf_dataset_id:
            kwargs = {}
            if sleeper_cfg.get("hf_config") is not None:
                kwargs["name"] = sleeper_cfg.get("hf_config")
            split = sleeper_cfg.get("split", "train")
            ds = load_dataset(hf_dataset_id, **kwargs, split=split)
            rows = [dict(row) for row in ds]
            source_name = normalize_text(sleeper_cfg.get("source_name") or hf_dataset_id.replace("/", "_"))
        else:
            msg = (
                "No external sleeper benchmark configured. Set external_sleeper.local_path to a local "
                "Anthropic/Hubinger sleeper-agents export, or external_sleeper.hf_dataset_id if using a HF mirror."
            )
            if required:
                raise RuntimeError(msg)
            return [], {"enabled": True, "n": 0, "note": msg, "blocking_for_iclr_claim": True}, []
    except Exception as e:
        if required:
            raise RuntimeError(f"Failed to load required external sleeper benchmark: {e}") from e
        return [], {"enabled": True, "n": 0, "note": f"external sleeper load failed: {e}", "blocking_for_iclr_claim": True}, [{"error": str(e)}]

    examples: List[ConversationExample] = []
    for idx, row in enumerate(rows[:max_records]):
        ex = _example_from_row(row, idx, source_name)
        if ex is None:
            errors.append({"row": idx, "error": "no prompt/messages field found"})
            continue
        examples.append(ex)

    summary = {
        "enabled": True,
        "source": source_name,
        "n": len(examples),
        "n_errors": len(errors),
        "note": "External sleeper validation holdout. These examples are never used for fitting synthetic CTCR/ARF templates.",
        "blocking_for_iclr_claim": len(examples) == 0,
    }
    return examples, summary, errors
