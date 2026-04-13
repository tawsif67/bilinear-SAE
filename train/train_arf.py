from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.arf import FingerprintClassifier, sparse_features_for_residuals
from models.sae import LinearSAE
from utils.gpu_memory import clear_cuda_cache, cuda_memory_snapshot, is_cuda_oom, log_cuda_memory


def train_arf_sae_classifier(
    train_residuals: torch.Tensor,
    train_labels: Sequence[str],
    cfg: Dict[str, Any],
    d_model: int,
    checkpoint_dir: Path,
    seed: int,
    logger,
) -> Tuple[LinearSAE, FingerprintClassifier, Dict[str, float]]:
    if train_residuals.size(0) == 0:
        raise RuntimeError("Cannot train ARF SAE without attack residuals.")
    arf_cfg = cfg.get("attack_fingerprints", {})
    expansion = int(arf_cfg.get("expansion_factor", 4))
    k_sparse = int(arf_cfg.get("k_sparse", 32))
    batch_size = max(1, int(arf_cfg.get("batch_size", 32)))
    steps = max(1, int(arf_cfg.get("steps", 300)))
    lr = float(arf_cfg.get("lr", 3e-4))
    l1_coeff = float(arf_cfg.get("l1_coeff", 0.001))
    residuals = train_residuals.float()
    model = LinearSAE(d_model, expansion, k_sparse).to(residuals.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    last = {"arf_mse": 0.0, "arf_l1": 0.0}
    pbar = tqdm(range(steps), desc=f"ARF seed {seed}", leave=False)
    for _ in pbar:
        while True:
            idx = torch.randint(0, residuals.size(0), (min(batch_size, residuals.size(0)),), device=residuals.device)
            h = residuals[idx]
            try:
                opt.zero_grad(set_to_none=True)
                acts, recon = model.reconstruct(h)
                mse = F.mse_loss(recon, h)
                l1 = acts.mean()
                loss = mse + l1_coeff * l1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                last = {"arf_mse": float(mse.detach().cpu()), "arf_l1": float(l1.detach().cpu())}
                pbar.set_postfix({**last, "batch_size": batch_size})
                break
            except RuntimeError as e:
                if not is_cuda_oom(e) or batch_size <= 1:
                    snap = cuda_memory_snapshot(residuals.device)
                    raise RuntimeError(f"ARF SAE training failed at batch_size={batch_size}; GPU memory snapshot={snap}") from e
                batch_size = max(1, batch_size // 2)
                clear_cuda_cache(residuals.device)
                if logger is not None:
                    logger.warning("ARF SAE OOM; retrying with batch_size=%d.", batch_size)
                log_cuda_memory(logger, "after_arf_oom_retry", residuals.device)
    model.eval()
    families = sorted(set(train_labels))
    x_train = sparse_features_for_residuals(model, residuals)
    label_to_id = {name: i for i, name in enumerate(families)}
    y_train = np.array([label_to_id[label] for label in train_labels], dtype=int)
    clf = FingerprintClassifier(families=families)
    clf.fit(x_train, y_train)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"arf_sae": model.state_dict(), "families": families}, checkpoint_dir / f"arf_sae_seed_{seed}.pt")
    if logger is not None:
        logger.info("Saved ARF SAE checkpoint for seed %d", seed)
    return model, clf, last
