from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSAE(nn.Module):
    def __init__(self, d_model: int, expansion_factor: int = 8, k_sparse: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_sparse = d_model * expansion_factor
        self.k_sparse = k_sparse
        self.enc = nn.Linear(d_model, self.d_sparse, bias=False)
        self.dec = nn.Linear(self.d_sparse, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.enc_bias = nn.Parameter(torch.zeros(self.d_sparse))

    def get_sparse_acts(self, h: torch.Tensor) -> torch.Tensor:
        pre = self.enc(h - self.pre_bias) + self.enc_bias
        vals, idx = torch.topk(pre, min(self.k_sparse, pre.size(-1)), dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, F.relu(vals))

    def reconstruct(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        acts = self.get_sparse_acts(h)
        return acts, self.dec(acts) + self.pre_bias


class BilinearSAE(nn.Module):
    def __init__(self, d_model: int, expansion_factor: int = 8, k_sparse: int = 64, rank: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_sparse = d_model * expansion_factor
        self.k_sparse = k_sparse
        self.enc = nn.Linear(d_model, self.d_sparse, bias=False)
        self.dec = nn.Linear(self.d_sparse, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.enc_bias = nn.Parameter(torch.zeros(self.d_sparse))
        self.u1 = nn.Linear(self.d_sparse, rank, bias=False)
        self.u2 = nn.Linear(self.d_sparse, rank, bias=False)
        self.vout = nn.Linear(rank, d_model, bias=False)
        nn.init.normal_(self.u1.weight, std=0.01)
        nn.init.normal_(self.u2.weight, std=0.01)
        nn.init.normal_(self.vout.weight, std=0.01)

    def get_sparse_acts(self, h: torch.Tensor) -> torch.Tensor:
        pre = self.enc(h - self.pre_bias) + self.enc_bias
        vals, idx = torch.topk(pre, min(self.k_sparse, pre.size(-1)), dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, F.relu(vals))

    def interaction(self, acts: torch.Tensor) -> torch.Tensor:
        return self.vout(self.u1(acts) * self.u2(acts))

    def reconstruct(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        acts = self.get_sparse_acts(h)
        return acts, self.dec(acts) + self.pre_bias + self.interaction(acts)


def sae_diagnostics(sae: nn.Module, feat_idx: torch.Tensor, h_bad: torch.Tensor, h_clean: torch.Tensor) -> dict:
    with torch.inference_mode():
        a_bad = sae.get_sparse_acts(h_bad)[:, feat_idx]
        a_clean = sae.get_sparse_acts(h_clean)[:, feat_idx]
        mu_bad = a_bad.float().mean(0)
        mu_clean = a_clean.float().mean(0)
        sigma = ((a_bad.float().std(0, unbiased=False) + a_clean.float().std(0, unbiased=False)) / 2).clamp(min=1e-8)
        d_prime = (mu_bad - mu_clean) / sigma
        dec_dirs = sae.dec.weight[:, feat_idx].T
    return {
        "d_prime": d_prime.cpu(),
        "decoder_norms": dec_dirs.norm(dim=1).cpu(),
        "selectivity": (mu_bad / (mu_clean + 1e-8)).cpu(),
    }
