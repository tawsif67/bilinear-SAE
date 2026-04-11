from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from data.loaders import ConversationExample
from utils.io import write_json, write_jsonl


def _ratio_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    train = int(round(n * ratios[0]))
    val = int(round(n * ratios[1]))
    train = min(max(train, 0), n)
    val = min(max(val, 0), n - train)
    test = n - train - val
    return train, val, test


def deterministic_split(
    examples: Sequence[ConversationExample],
    ratios: Tuple[float, float, float],
    seed: int,
    stratify_key: str | None = None,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        key = "all"
        if stratify_key:
            key = str(getattr(ex, stratify_key, "") or (ex.metadata or {}).get(stratify_key, "missing"))
        buckets[key].append(i)

    splits = {"train": [], "val": [], "test": []}
    for idxs in buckets.values():
        arr = np.array(idxs)
        rng.shuffle(arr)
        n_train, n_val, _ = _ratio_counts(len(arr), ratios)
        splits["train"].extend(arr[:n_train].tolist())
        splits["val"].extend(arr[n_train:n_train + n_val].tolist())
        splits["test"].extend(arr[n_train + n_val:].tolist())

    for values in splits.values():
        values.sort()
    return splits


def apply_split_labels(examples: Sequence[ConversationExample], splits: Dict[str, List[int]]) -> List[ConversationExample]:
    out = list(examples)
    for split_name, idxs in splits.items():
        for idx in idxs:
            out[idx].split = split_name
    return out


def split_hash(splits: Dict[str, List[int]]) -> str:
    raw = repr({k: list(v) for k, v in sorted(splits.items())})
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def save_splits(run_dir: Path, name: str, examples: Sequence[ConversationExample], splits: Dict[str, List[int]]) -> None:
    split_dir = run_dir / "splits"
    write_json(split_dir / f"{name}_indices.json", {"name": name, "hash": split_hash(splits), "splits": splits})
    rows = []
    for split_name, idxs in splits.items():
        for idx in idxs:
            row = examples[idx].to_dict()
            row["split_index"] = idx
            row["split"] = split_name
            rows.append(row)
    write_jsonl(split_dir / f"{name}_examples.jsonl", rows)
