from __future__ import annotations

from typing import Iterable

import torch


def sparse_ablation_delta(sae, h: torch.Tensor, feature_ids: torch.Tensor) -> torch.Tensor:
    if feature_ids.numel() == 0:
        return torch.zeros_like(h)
    acts = sae.get_sparse_acts(h)
    selected = torch.zeros_like(acts)
    selected[:, feature_ids] = acts[:, feature_ids]
    return sae.dec(selected)


def make_sae_feature_ablation_hook(sae, feature_ids: Iterable[int], position: str = "last"):
    ids = torch.tensor(list(feature_ids), dtype=torch.long)

    def hook(module, args, kwargs):
        h = args[0]
        local_ids = ids.to(h.device)
        if position == "all":
            flat = h.reshape(-1, h.size(-1))
            delta = sparse_ablation_delta(sae, flat, local_ids).reshape_as(h)
            new_h = h - delta
        else:
            new_h = h.clone()
            delta = sparse_ablation_delta(sae, h[:, -1, :], local_ids)
            new_h[:, -1, :] = new_h[:, -1, :] - delta
        return (new_h, *args[1:]), kwargs

    return hook


def intervention_locality_ratio(attack_values: torch.Tensor, clean_values: torch.Tensor) -> float:
    attack = float(attack_values.float().abs().mean().cpu()) if attack_values.numel() else 0.0
    clean = float(clean_values.float().abs().mean().cpu()) if clean_values.numel() else 0.0
    return attack / max(clean, 1e-8)
