from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


TRANSFER_SCORES = {
    "trajectory_only": "trajectory_score",
    "linear_sae_only": "linear_feature_score",
    "bilinear_sae_only": "bilinear_feature_score",
    "full_fused": "full_fused_score",
}


def _best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(score)
    if len(candidates) == 0:
        return 0.5
    best_t = float(candidates[0])
    best_bal = -1.0
    for t in candidates:
        pred = score >= t
        pos = y == 1
        neg = y == 0
        tpr = pred[pos].mean() if pos.any() else 0.0
        tnr = (~pred[neg]).mean() if neg.any() else 0.0
        bal = 0.5 * (tpr + tnr)
        if bal > best_bal:
            best_bal = float(bal)
            best_t = float(t)
    return best_t


def transfer_rows(taxonomy_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    df = pd.DataFrame(taxonomy_rows)
    if df.empty:
        return []
    seed = int(df["seed"].iloc[0]) if "seed" in df.columns and len(df) else 0
    rows = []
    slices = sorted(df["eval_slice"].dropna().unique().tolist())
    for method, score_col in TRANSFER_SCORES.items():
        for source in slices:
            src = df[df["eval_slice"] == source]
            y_src = src["target"].to_numpy(dtype=int)
            if len(np.unique(y_src)) < 2:
                continue
            threshold = _best_threshold(y_src, src[score_col].to_numpy(dtype=float))
            for target in slices:
                tgt = df[df["eval_slice"] == target]
                y = tgt["target"].to_numpy(dtype=int)
                score = tgt[score_col].to_numpy(dtype=float)
                pred = score >= threshold
                pos = y == 1
                neg = y == 0
                rows.append({
                    "seed": seed,
                    "method": method,
                    "source_eval_slice": source,
                    "target_eval_slice": target,
                    "threshold": threshold,
                    "balanced_accuracy": float(0.5 * ((pred[pos].mean() if pos.any() else 0.0) + ((~pred[neg]).mean() if neg.any() else 0.0))),
                    "attack_recall": float(pred[pos].mean()) if pos.any() else 0.0,
                    "clean_specificity": float((~pred[neg]).mean()) if neg.any() else 0.0,
                })
    return rows
