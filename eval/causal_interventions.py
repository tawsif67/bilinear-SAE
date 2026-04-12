from __future__ import annotations

from typing import Any, Dict, List

import torch

from data.loaders import ConversationExample
from models.interventions import intervention_locality_ratio, sparse_ablation_delta


@torch.inference_mode()
def causal_intervention_rows(
    examples: List[ConversationExample],
    h_seq: torch.Tensor,
    linear_sae,
    bilinear_sae,
    bad_feats: Dict[str, torch.Tensor],
    seed: int,
    eval_slice: str,
) -> List[Dict[str, Any]]:
    if not examples:
        return []
    h_last = h_seq[:, -1, :]
    y = torch.tensor([ex.target for ex in examples], dtype=torch.bool, device=h_last.device)
    linear_feats = bad_feats["linear"][:2]
    bilinear_feats = bad_feats["bilinear"][:2]
    lin_delta = sparse_ablation_delta(linear_sae, h_last, linear_feats)
    bil_delta = sparse_ablation_delta(bilinear_sae, h_last, bilinear_feats)
    pair_delta = lin_delta + bil_delta
    lin_effect = lin_delta.norm(dim=-1)
    bil_effect = bil_delta.norm(dim=-1)
    pair_effect = pair_delta.norm(dim=-1)
    rows = []
    for name, values in [
        ("linear_top_features", lin_effect),
        ("bilinear_top_features", bil_effect),
        ("linear_plus_bilinear_pair", pair_effect),
    ]:
        rows.append({
            "seed": seed,
            "eval_slice": eval_slice,
            "group": examples[0].group if eval_slice != "sleeper_ood" else "sleeper_distributed_trigger",
            "intervention": name,
            "mean_proxy_effect": float(values.float().mean().cpu()),
            "attack_proxy_effect": float(values[y].float().mean().cpu()) if y.any() else 0.0,
            "clean_proxy_effect": float(values[~y].float().mean().cpu()) if (~y).any() else 0.0,
            "intervention_locality": intervention_locality_ratio(values[y], values[~y]),
        })
    if len(rows) == 3:
        rows[2]["pair_synergy_proxy"] = rows[2]["mean_proxy_effect"] - rows[0]["mean_proxy_effect"] - rows[1]["mean_proxy_effect"]
    return rows
