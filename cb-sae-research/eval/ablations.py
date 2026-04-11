from __future__ import annotations

from typing import Dict

import torch


def pair_synergy_score(a_only: float, b_only: float, both: float, neither: float) -> float:
    return float((both - neither) - (a_only - neither) - (b_only - neither))


def causal_pair_ablation_scores(linear_acts: torch.Tensor, bilinear_acts: torch.Tensor, linear_feats: torch.Tensor, bilinear_feats: torch.Tensor) -> Dict[str, float]:
    la = linear_acts[:, linear_feats[:1]].sum().item() if len(linear_feats) else 0.0
    ba = bilinear_acts[:, bilinear_feats[:1]].sum().item() if len(bilinear_feats) else 0.0
    pair = linear_acts[:, linear_feats[:1]].sum().item() + bilinear_acts[:, bilinear_feats[:1]].sum().item() if len(linear_feats) and len(bilinear_feats) else la + ba
    return {"ablate_feature_a_effect": float(la), "ablate_feature_b_effect": float(ba), "ablate_pair_effect": float(pair), "pair_synergy": float(pair - la - ba)}
