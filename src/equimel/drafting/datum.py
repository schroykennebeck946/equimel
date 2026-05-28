from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(in_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.bn(self.dense(features))
        return functional.normalize(projected, dim=-1)
