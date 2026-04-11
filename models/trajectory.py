from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    def __init__(self, d_model: int, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.threat_head = nn.Linear(d_model, 1)
        self.delta_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
        out = self.encoder(x, mask=mask)
        return torch.sigmoid(self.threat_head(out).squeeze(-1)), self.delta_head(out).squeeze(-1)


def pool_by_turn_bounds(h: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    pooled = []
    for b in range(h.size(0)):
        b1, b2, b3 = [int(v.item()) for v in bounds[b]]
        p1 = h[b, : b1 + 1].mean(0)
        p2 = h[b, b1 + 1 : b2 + 1].mean(0) if b2 > b1 else h[b, b2]
        p3 = h[b, b2 + 1 : b3 + 1].mean(0) if b3 > b2 else h[b, b3]
        pooled.append(torch.stack([p1, p2, p3]))
    return torch.stack(pooled)
