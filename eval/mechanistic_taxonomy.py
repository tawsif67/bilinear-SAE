from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from data.loaders import ConversationExample


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / max(len(values) - 1, 1)


def mechanistic_taxonomy_rows(examples: List[ConversationExample], scores: Dict[str, np.ndarray], seed: int, eval_slice: str) -> List[Dict[str, Any]]:
    linear = _rank01(np.asarray(scores["linear_sae_only"], dtype=float))
    bilinear = _rank01(np.asarray(scores["bilinear_sae_only"], dtype=float))
    trajectory = _rank01(np.asarray(scores["trajectory_only"], dtype=float))
    full = _rank01(np.asarray(scores["full_fused"], dtype=float))
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples):
        bilinear_gain = float(bilinear[i] - max(linear[i], trajectory[i]))
        trajectory_gain = float(trajectory[i] - max(linear[i], bilinear[i]))
        if bilinear_gain > 0.15 and bilinear[i] >= 0.6:
            category = "conjunction_dominant"
        elif trajectory_gain > 0.15 and trajectory[i] >= 0.6:
            category = "trajectory_drift_dominant"
        elif full[i] >= 0.6 and max(bilinear[i], trajectory[i]) >= 0.5:
            category = "mixed"
        else:
            category = "weak_or_neither"
        rows.append({
            "seed": seed,
            "eval_slice": eval_slice,
            "group": ex.group if eval_slice != "sleeper_ood" else "sleeper_distributed_trigger",
            "example_id": ex.id,
            "target": ex.target,
            "family": ex.family,
            "mode": ex.mode,
            "condition_a": ex.condition_a,
            "condition_b": ex.condition_b,
            "family_holdout": ex.family_holdout,
            "is_ood": ex.is_ood,
            "linear_feature_score": float(linear[i]),
            "bilinear_feature_score": float(bilinear[i]),
            "trajectory_score": float(trajectory[i]),
            "full_fused_score": float(full[i]),
            "bilinear_gain_over_single_and_traj": bilinear_gain,
            "trajectory_gain_over_sparse": trajectory_gain,
            "mechanistic_category": category,
        })
    return rows
