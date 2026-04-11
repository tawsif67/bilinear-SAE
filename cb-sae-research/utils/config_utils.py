from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

from utils.io import load_yaml


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = val
    return out


def parse_thresholds(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    thresholds = []
    for t in out["eval"]["thresholds"]:
        if isinstance(t, str) and t.lower() in {"inf", "+inf", "infinity"}:
            thresholds.append(float("inf"))
        else:
            thresholds.append(float(t))
    out["eval"]["thresholds"] = thresholds
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg = parse_thresholds(load_yaml(Path(path)))
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    if cfg.get("subject_model") != "Qwen/Qwen2.5-3B-Instruct":
        raise ValueError("subject_model must be Qwen/Qwen2.5-3B-Instruct for this experiment.")
    if cfg.get("judge_model") != "google/gemma-3-4b-it":
        raise ValueError("judge_model must be google/gemma-3-4b-it; do not silently swap judges.")
    if "intercept_layers" not in cfg or not cfg["intercept_layers"]:
        raise ValueError("Config must include at least one intercept layer.")
    if cfg["lora"]["batch_size"] <= 0 or cfg["sae"]["batch_size"] <= 0 or cfg["fuser"]["batch_size"] <= 0:
        raise ValueError("Batch sizes must be positive.")
