import torch
import torch.nn as nn


class FusionHead(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, s3: torch.Tensor) -> torch.Tensor:
        x = torch.stack([s1, s2, s3, s1 * s2, s1 * s3, s2 * s3], dim=-1)
        return self.net(x).squeeze(-1)
