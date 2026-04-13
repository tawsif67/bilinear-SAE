from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


SCORE_COLUMNS = ["linear_feature_score", "bilinear_feature_score", "trajectory_score", "ctcr_residual_score", "full_fused_score"]


def conjunction_control_rows(taxonomy_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    df = pd.DataFrame(taxonomy_rows)
    if df.empty or "mode" not in df.columns:
        return []
    df = df[df["group"] == "sleeper_distributed_trigger"].copy()
    if df.empty:
        return []
    rows: List[Dict[str, Any]] = []
    group_cols = ["seed", "family"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        key_data = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        by_mode = sub.groupby("mode")[SCORE_COLUMNS].mean()
        for score_col in SCORE_COLUMNS:
            def val(mode: str) -> float:
                return float(by_mode.loc[mode, score_col]) if mode in by_mode.index else 0.0
            both = val("both_conditions")
            only_a = val("only_condition_a")
            only_b = val("only_condition_b")
            neither = val("neither_condition")
            shuffled = val("shuffled_turns")
            compressed = val("single_turn_compressed")
            rows.append({
                **key_data,
                "score_type": score_col,
                "both_conditions": both,
                "only_condition_a": only_a,
                "only_condition_b": only_b,
                "neither_condition": neither,
                "shuffled_turns": shuffled,
                "single_turn_compressed": compressed,
                "conjunction_selectivity": both - max(only_a, only_b, neither),
                "order_sensitivity": both - shuffled,
                "cross_turn_advantage": both - compressed,
            })
    return rows
