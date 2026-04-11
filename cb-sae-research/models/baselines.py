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
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn is required for probe baselines. Install requirements.txt.")
        self.clf = LogisticRegression(max_iter=2000, solver="lbfgs")

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.clf.fit(x, y)

    def score(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> ProbeResult:
        scores = self.clf.predict_proba(x)[:, 1]
        auroc = float(roc_auc_score(y, scores)) if roc_auc_score is not None and y is not None and len(set(y.tolist())) > 1 else 0.0
        auprc = float(average_precision_score(y, scores)) if average_precision_score is not None and y is not None else 0.0
        return ProbeResult(scores=scores, auroc=auroc, auprc=auprc)


class ActivationProbe(DenseLinearProbe):
    def __init__(self):
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn is required for activation probes. Install requirements.txt.")
        self.clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=2000)


def mean_difference_vector(h_bad: torch.Tensor, h_clean: torch.Tensor) -> torch.Tensor:
    vec = h_bad.float().mean(0) - h_clean.float().mean(0)
    return vec / vec.norm().clamp(min=1e-6)


def top_mean_difference_dims(h_bad: torch.Tensor, h_clean: torch.Tensor, topk: int = 100) -> torch.Tensor:
    return torch.argsort(h_bad.float().mean(0) - h_clean.float().mean(0), descending=True)[:topk]
