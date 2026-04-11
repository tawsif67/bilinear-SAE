from __future__ import annotations

from typing import Dict, List

import numpy as np
try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


COMPARISONS = [
    ("full_fused", "trajectory_only"),
    ("full_fused", "linear_sae_trajectory"),
    ("full_fused", "bilinear_sae_only"),
    ("full_fused", "dense_probe"),
]


def significance_rows(metric_rows: List[Dict]) -> List[Dict]:
    rows = []
    for a, b in COMPARISONS:
        avals = np.array([r["asr_reduction"] for r in metric_rows if r.get("method") == a], dtype=float)
        bvals = np.array([r["asr_reduction"] for r in metric_rows if r.get("method") == b], dtype=float)
        n = min(len(avals), len(bvals))
        if n < 2:
            p = float("nan")
            stat = float("nan")
        elif wilcoxon is None:
            p = float("nan")
            stat = float("nan")
        else:
            stat, p = wilcoxon(avals[:n], bvals[:n], alternative="greater")
            stat = float(stat)
            p = float(p)
        rows.append({"comparison": f"{a} vs {b}", "full_mean": float(np.nanmean(avals)) if len(avals) else 0.0, "baseline_mean": float(np.nanmean(bvals)) if len(bvals) else 0.0, "statistic": stat, "p_value": p})
    return rows
