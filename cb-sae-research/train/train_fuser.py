from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from models.fusion import FusionHead
from models.trajectory import TrajectoryEncoder


def _sparse_scores(h_seq: torch.Tensor, linear_sae, bilinear_sae, bad_feats: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    h_last = h_seq[:, -1, :]
    a1 = linear_sae.get_sparse_acts(h_last)
    a2 = bilinear_sae.get_sparse_acts(h_last)
    s1 = a1[:, bad_feats["linear"]].sum(-1).float() if len(bad_feats["linear"]) else torch.zeros(h_seq.size(0), device=h_seq.device)
    s2 = a2[:, bad_feats["bilinear"]].sum(-1).float() if len(bad_feats["bilinear"]) else torch.zeros(h_seq.size(0), device=h_seq.device)
    return s1, s2


def train_fuser(h_seq: torch.Tensor, y: torch.Tensor, linear_sae, bilinear_sae, bad_feats: Dict[str, torch.Tensor], cfg: Dict[str, Any], d_model: int, checkpoint_dir: Path, seed: int, logger):
    traj_cfg = cfg["trajectory_encoder"]
    fuser_cfg = cfg["fuser"]
    encoder = TrajectoryEncoder(d_model, int(traj_cfg["heads"]), int(traj_cfg["layers"]), float(traj_cfg["dropout"])).to(h_seq.device)
    fuser = FusionHead(int(fuser_cfg["hidden_dim"])).to(h_seq.device)
    linear_sae.eval()
    bilinear_sae.eval()
    for p in linear_sae.parameters():
        p.requires_grad = False
    for p in bilinear_sae.parameters():
        p.requires_grad = False
    dl = DataLoader(TensorDataset(h_seq, y.float()), batch_size=int(fuser_cfg["batch_size"]), shuffle=True)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(fuser.parameters()), lr=float(fuser_cfg["lr"]))
    it = iter(dl)
    pbar = tqdm(range(int(fuser_cfg["steps"])), desc=f"Fuser seed {seed}", leave=False)
    last_loss = 0.0
    for _ in pbar:
        try:
            bh, by = next(it)
        except StopIteration:
            it = iter(dl)
            bh, by = next(it)
        opt.zero_grad(set_to_none=True)
        threat, delta = encoder(bh)
        s1, s2 = _sparse_scores(bh, linear_sae, bilinear_sae, bad_feats)
        pred = fuser(s1, s2, threat[:, -1].float())
        target_traj = torch.zeros_like(threat)
        target_traj[by.bool(), 1] = 0.5
        target_traj[by.bool(), 2] = 1.0
        loss = F.binary_cross_entropy(pred, by) + 0.5 * F.mse_loss(threat, target_traj) + 0.05 * delta.abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(fuser.parameters()), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
        pbar.set_postfix(loss=last_loss)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"trajectory": encoder.state_dict(), "fuser": fuser.state_dict()}, checkpoint_dir / f"fuser_seed_{seed}.pt")
    logger.info("Saved fuser checkpoint for seed %d", seed)
    encoder.eval()
    fuser.eval()
    return encoder, fuser, {"fuser_loss": last_loss}
