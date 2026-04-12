from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _auc_rank(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def null_control_rows(taxonomy_rows: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    df = pd.DataFrame(taxonomy_rows)
    if df.empty:
        return []
    rng = np.random.default_rng(seed)
    rows = []
    for (eval_slice, score_type), sub in df.groupby(["eval_slice", "mechanistic_category"], dropna=False):
        y = sub["target"].to_numpy(dtype=int)
        score = sub["bilinear_feature_score"].to_numpy(dtype=float)
        shuffled = rng.permutation(y)
        rows.append({
            "seed": seed,
            "eval_slice": eval_slice,
            "category": score_type,
            "true_bilinear_auc": _auc_rank(y, score),
            "shuffled_label_auc": _auc_rank(shuffled, score),
            "null_corrected_bilinear_auc": _auc_rank(y, score) - _auc_rank(shuffled, score),
        })
    return rows
