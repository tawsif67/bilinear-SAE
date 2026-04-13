from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.sae import BilinearSAE, LinearSAE
from utils.gpu_memory import clear_cuda_cache, cuda_memory_snapshot, is_cuda_oom, log_cuda_memory


def train_saes(h_seq: torch.Tensor, cfg: Dict[str, Any], d_model: int, checkpoint_dir: Path, seed: int, logger) -> Tuple[LinearSAE, BilinearSAE, Dict[str, float]]:
    if h_seq.size(0) == 0:
        raise RuntimeError("Cannot train SAEs on an empty hidden-state tensor.")
    h_seq = h_seq.float()
    sae_cfg = cfg["sae"]
    linear = LinearSAE(d_model, int(sae_cfg["expansion_factor"]), int(sae_cfg["k_sparse"]))
    bilinear = BilinearSAE(d_model, int(sae_cfg["expansion_factor"]), int(sae_cfg["k_sparse"]), int(sae_cfg["bilinear_rank"]))
    device = h_seq.device
    linear.to(device)
    bilinear.to(device)
    h_last = h_seq[:, -1, :]
    batch_size = max(1, int(sae_cfg["batch_size"]))
    opt = torch.optim.AdamW(list(linear.parameters()) + list(bilinear.parameters()), lr=float(sae_cfg["lr"]))
    last = {"linear_mse": 0.0, "bilinear_mse": 0.0}
    pbar = tqdm(range(int(sae_cfg["steps"])), desc=f"SAE seed {seed}", leave=False)
    for _ in pbar:
        while True:
            idx = torch.randint(0, h_last.size(0), (min(batch_size, h_last.size(0)),), device=h_last.device)
            h = h_last[idx]
            try:
                opt.zero_grad(set_to_none=True)
                a1, r1 = linear.reconstruct(h)
                a2, r2 = bilinear.reconstruct(h)
                bil2 = bilinear.u1.weight.pow(2).mean() + bilinear.u2.weight.pow(2).mean() + bilinear.vout.weight.pow(2).mean()
                loss1 = F.mse_loss(r1, h) + float(sae_cfg["l1_coeff"]) * a1.mean()
                loss2 = F.mse_loss(r2, h) + float(sae_cfg["l1_coeff"]) * a2.mean() + float(sae_cfg["bilinear_l2_coeff"]) * bil2
                loss = loss1 + loss2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(linear.parameters()) + list(bilinear.parameters()), 1.0)
                opt.step()
                last = {"linear_mse": float(F.mse_loss(r1, h).detach().cpu()), "bilinear_mse": float(F.mse_loss(r2, h).detach().cpu())}
                pbar.set_postfix({**last, "batch_size": batch_size})
                break
            except RuntimeError as e:
                if not is_cuda_oom(e) or batch_size <= 1:
                    snap = cuda_memory_snapshot(device)
                    raise RuntimeError(f"SAE training failed at batch_size={batch_size}; GPU memory snapshot={snap}") from e
                batch_size = max(1, batch_size // 2)
                clear_cuda_cache(device)
                if logger is not None:
                    logger.warning("SAE OOM; retrying with batch_size=%d.", batch_size)
                log_cuda_memory(logger, "after_sae_oom_retry", device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"linear": linear.state_dict(), "bilinear": bilinear.state_dict()}, checkpoint_dir / f"sae_seed_{seed}.pt")
    logger.info("Saved SAE checkpoint for seed %d", seed)
    linear.eval()
    bilinear.eval()
    return linear, bilinear, last
