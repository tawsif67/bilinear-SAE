from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.sae import BilinearSAE
from utils.gpu_memory import clear_cuda_cache, cuda_memory_snapshot, is_cuda_oom, log_cuda_memory


def train_ctcr_sae(residuals: torch.Tensor, cfg: Dict[str, Any], d_model: int, checkpoint_dir: Path, seed: int, logger) -> Tuple[BilinearSAE, Dict[str, float]]:
    if residuals.size(0) == 0:
        raise RuntimeError("Cannot train CTCR SAE without matched counterfactual residuals.")
    residuals = residuals.float()
    sae_cfg = cfg["sae"]
    model = BilinearSAE(d_model, int(sae_cfg["expansion_factor"]), int(sae_cfg["k_sparse"]), int(sae_cfg["bilinear_rank"])).to(residuals.device)
    batch_size = max(1, int(sae_cfg["batch_size"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(sae_cfg["lr"]))
    last = {"ctcr_mse": 0.0, "ctcr_bilinear_l2": 0.0}
    pbar = tqdm(range(int(sae_cfg["steps"])), desc=f"CTCR seed {seed}", leave=False)
    for _ in pbar:
        while True:
            idx = torch.randint(0, residuals.size(0), (min(batch_size, residuals.size(0)),), device=residuals.device)
            h = residuals[idx]
            try:
                opt.zero_grad(set_to_none=True)
                acts, recon = model.reconstruct(h)
                bil2 = model.u1.weight.pow(2).mean() + model.u2.weight.pow(2).mean() + model.vout.weight.pow(2).mean()
                loss = F.mse_loss(recon, h) + float(sae_cfg["l1_coeff"]) * acts.mean() + float(sae_cfg["bilinear_l2_coeff"]) * bil2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                last = {"ctcr_mse": float(F.mse_loss(recon, h).detach().cpu()), "ctcr_bilinear_l2": float(bil2.detach().cpu())}
                pbar.set_postfix({**last, "batch_size": batch_size})
                break
            except RuntimeError as e:
                if not is_cuda_oom(e) or batch_size <= 1:
                    snap = cuda_memory_snapshot(residuals.device)
                    raise RuntimeError(f"CTCR SAE training failed at batch_size={batch_size}; GPU memory snapshot={snap}") from e
                batch_size = max(1, batch_size // 2)
                clear_cuda_cache(residuals.device)
                if logger is not None:
                    logger.warning("CTCR SAE OOM; retrying with batch_size=%d.", batch_size)
                log_cuda_memory(logger, "after_ctcr_oom_retry", residuals.device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"ctcr_sae": model.state_dict()}, checkpoint_dir / f"ctcr_sae_seed_{seed}.pt")
    if logger is not None:
        logger.info("Saved CTCR SAE checkpoint for seed %d", seed)
    model.eval()
    return model, last
