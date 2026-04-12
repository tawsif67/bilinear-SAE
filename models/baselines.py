from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:
    LogisticRegression = None
    average_precision_score = None
    roc_auc_score = None


@dataclass
class ProbeResult:
    scores: np.ndarray
    auroc: float
    auprc: float


class DenseLinearProbe:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=2000, solver="lbfgs") if LogisticRegression is not None else None
        self.constant_score: float | None = None
        self.weight: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.shape[0] == 0:
            raise RuntimeError("Cannot fit probe on an empty training matrix.")
        if len(np.unique(y)) < 2:
            self.constant_score = float(y[0]) if len(y) else 0.0
            return
        self.constant_score = None
        if self.clf is not None:
            self.clf.fit(x, y)
            self.weight = None
            return
        pos = x[y == 1]
        neg = x[y == 0]
        self.weight = pos.mean(axis=0) - neg.mean(axis=0)
        scale = max(float(np.linalg.norm(self.weight)), 1e-8)
        self.weight = self.weight / scale
        self.bias = -0.5 * float((pos.mean(axis=0) + neg.mean(axis=0)) @ self.weight)

    def score(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> ProbeResult:
        x = np.asarray(x, dtype=float)
        if self.constant_score is not None:
            scores = np.full(x.shape[0], self.constant_score, dtype=float)
        elif self.clf is not None:
            scores = self.clf.predict_proba(x)[:, 1]
        else:
            if self.weight is None:
                raise RuntimeError("Probe must be fit before scoring.")
            raw = x @ self.weight + self.bias
            scores = 1.0 / (1.0 + np.exp(-np.clip(raw, -50, 50)))
        auroc = float(roc_auc_score(y, scores)) if roc_auc_score is not None and y is not None and len(set(y.tolist())) > 1 else 0.0
        auprc = float(average_precision_score(y, scores)) if average_precision_score is not None and y is not None else 0.0
        return ProbeResult(scores=scores, auroc=auroc, auprc=auprc)


class ActivationProbe(DenseLinearProbe):
    def __init__(self):
        self.clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=2000) if LogisticRegression is not None else None
        self.constant_score: float | None = None
        self.weight: np.ndarray | None = None
        self.bias: float = 0.0


class TorchMLPProbe:
    def __init__(self, input_dim: int, hidden_dim: int = 128, steps: int = 200, lr: float = 1e-3, seed: int = 0):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.steps = int(steps)
        self.lr = float(lr)
        self.seed = int(seed)
        self.constant_score: float | None = None
        self.model: torch.nn.Module | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.shape[0] == 0:
            raise RuntimeError("Cannot fit MLP probe on an empty training matrix.")
        if len(np.unique(y)) < 2:
            self.constant_score = float(y[0]) if len(y) else 0.0
            return
        torch.manual_seed(self.seed)
        self.constant_score = None
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y).view(-1, 1)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(max(self.steps, 1)):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(self.model(xt), yt)
            loss.backward()
            opt.step()
        self.model.eval()

    def score(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> ProbeResult:
        x = np.asarray(x, dtype=np.float32)
        if self.constant_score is not None:
            scores = np.full(x.shape[0], self.constant_score, dtype=float)
        else:
            if self.model is None:
                raise RuntimeError("MLP probe must be fit before scoring.")
            with torch.inference_mode():
                scores = torch.sigmoid(self.model(torch.from_numpy(x))).view(-1).cpu().numpy()
        auroc = float(roc_auc_score(y, scores)) if roc_auc_score is not None and y is not None and len(set(y.tolist())) > 1 else 0.0
        auprc = float(average_precision_score(y, scores)) if average_precision_score is not None and y is not None else 0.0
        return ProbeResult(scores=scores, auroc=auroc, auprc=auprc)


def mean_difference_vector(h_bad: torch.Tensor, h_clean: torch.Tensor) -> torch.Tensor:
    if h_bad.numel() == 0 or h_clean.numel() == 0:
        d_model = h_bad.size(-1) if h_bad.ndim else h_clean.size(-1)
        return torch.zeros(d_model, device=h_bad.device if h_bad.numel() else h_clean.device)
    vec = h_bad.float().mean(0) - h_clean.float().mean(0)
    return vec / vec.norm().clamp(min=1e-6)


def top_mean_difference_dims(h_bad: torch.Tensor, h_clean: torch.Tensor, topk: int = 100) -> torch.Tensor:
    if h_bad.numel() == 0 or h_clean.numel() == 0:
        d_model = h_bad.size(-1) if h_bad.ndim else h_clean.size(-1)
        return torch.arange(min(topk, d_model), device=h_bad.device if h_bad.numel() else h_clean.device)
    return torch.argsort(h_bad.float().mean(0) - h_clean.float().mean(0), descending=True)[:topk]
