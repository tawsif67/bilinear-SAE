from __future__ import annotations

from typing import Any, Dict, List

import torch


@torch.inference_mode()
def layer_summary_rows(h_seq: torch.Tensor, y: torch.Tensor, seed: int, layer: int, split: str) -> List[Dict[str, Any]]:
    if h_seq.numel() == 0:
        return []
    yb = y.bool().to(h_seq.device)
    turn_norms = h_seq.float().norm(dim=-1)
    drift = (h_seq[:, -1, :] - h_seq[:, 0, :]).float().norm(dim=-1)
    rows = []
    for turn in range(h_seq.size(1)):
        vals = turn_norms[:, turn]
        rows.append({
            "seed": seed,
            "layer": layer,
            "split": split,
            "turn": turn + 1,
            "mean_hidden_norm": float(vals.mean().cpu()),
            "attack_hidden_norm": float(vals[yb].mean().cpu()) if yb.any() else 0.0,
            "clean_hidden_norm": float(vals[~yb].mean().cpu()) if (~yb).any() else 0.0,
            "attack_clean_norm_gap": float((vals[yb].mean() - vals[~yb].mean()).cpu()) if yb.any() and (~yb).any() else 0.0,
        })
    rows.append({
        "seed": seed,
        "layer": layer,
        "split": split,
        "turn": 0,
        "mean_hidden_norm": float(drift.mean().cpu()),
        "attack_hidden_norm": float(drift[yb].mean().cpu()) if yb.any() else 0.0,
        "clean_hidden_norm": float(drift[~yb].mean().cpu()) if (~yb).any() else 0.0,
        "attack_clean_norm_gap": float((drift[yb].mean() - drift[~yb].mean()).cpu()) if yb.any() and (~yb).any() else 0.0,
    })
    return rows
