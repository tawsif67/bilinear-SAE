from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch

from data.loaders import ConversationExample


REQUIRED_CTCR_MODES = ("both_conditions", "only_condition_a", "only_condition_b", "neither_condition")


def _residual_group(ex: ConversationExample) -> str:
    meta = ex.metadata or {}
    return str(meta.get("residual_group") or "")


def _mode(ex: ConversationExample) -> str:
    return str(ex.mode or (ex.metadata or {}).get("ctcr_role", ""))


def ctcr_residual(h_both: torch.Tensor, h_a: torch.Tensor, h_b: torch.Tensor, h_0: torch.Tensor) -> torch.Tensor:
    return h_both - h_a - h_b + h_0


def build_ctcr_residual_dataset(
    examples: List[ConversationExample],
    h_seq: torch.Tensor,
    turn: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    groups: Dict[str, Dict[str, int]] = defaultdict(dict)
    for idx, ex in enumerate(examples):
        group = _residual_group(ex)
        if group:
            groups[group][_mode(ex)] = idx

    residuals = []
    targets = []
    rows: List[Dict[str, Any]] = []
    for group, by_mode in groups.items():
        if not all(mode in by_mode for mode in REQUIRED_CTCR_MODES):
            continue
        ib = by_mode["both_conditions"]
        ia = by_mode["only_condition_a"]
        ib_only = by_mode["only_condition_b"]
        i0 = by_mode["neither_condition"]
        r = ctcr_residual(h_seq[ib, turn, :].float(), h_seq[ia, turn, :].float(), h_seq[ib_only, turn, :].float(), h_seq[i0, turn, :].float())
        ex = examples[ib]
        residuals.append(r)
        targets.append(int(ex.target))
        rows.append({
            "residual_group": group,
            "both_index": ib,
            "only_a_index": ia,
            "only_b_index": ib_only,
            "neither_index": i0,
            "target": int(ex.target),
            "family": ex.family,
            "source_is_benign": bool(ex.source_is_benign),
            "family_holdout": bool(ex.family_holdout),
            "is_ood": bool(ex.is_ood),
        })
    if not residuals:
        return torch.empty(0, h_seq.size(-1), device=h_seq.device), torch.empty(0, dtype=torch.long, device=h_seq.device), rows
    return torch.stack(residuals, 0), torch.tensor(targets, dtype=torch.long, device=h_seq.device), rows


@torch.inference_mode()
def ctcr_scores_for_examples(
    examples: List[ConversationExample],
    h_seq: torch.Tensor,
    ctcr_sae,
    feature_ids: torch.Tensor,
    turn: int = -1,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    scores = torch.zeros(len(examples), dtype=torch.float32, device=h_seq.device)
    residuals, targets, rows = build_ctcr_residual_dataset(examples, h_seq, turn=turn)
    if residuals.numel() == 0:
        return scores, rows
    feature_ids = feature_ids.to(h_seq.device)
    acts = ctcr_sae.get_sparse_acts(residuals.float())
    residual_scores = acts[:, feature_ids].sum(-1).float() if feature_ids.numel() else acts.sum(-1).float()
    residual_norms = residuals.float().norm(dim=-1)
    for i, row in enumerate(rows):
        scores[int(row["both_index"])] = residual_scores[i]
        row["ctcr_residual_score"] = float(residual_scores[i].detach().cpu())
        row["ctcr_residual_norm"] = float(residual_norms[i].detach().cpu())
        row["ctcr_target"] = int(targets[i].detach().cpu())
    return scores, rows


def ctcr_analysis_rows(
    examples: List[ConversationExample],
    h_seq: torch.Tensor,
    ctcr_sae,
    feature_ids: torch.Tensor,
    seed: int,
    eval_slice: str,
    turn: int = -1,
) -> List[Dict[str, Any]]:
    scores, rows = ctcr_scores_for_examples(examples, h_seq, ctcr_sae, feature_ids, turn=turn)
    out = []
    for row in rows:
        out.append({
            **row,
            "seed": seed,
            "eval_slice": eval_slice,
            "group": "sleeper_distributed_trigger",
            "method": "ctcr_residual_bilinear",
            "assigned_example_score": float(scores[int(row["both_index"])].detach().cpu()),
        })
    return out
