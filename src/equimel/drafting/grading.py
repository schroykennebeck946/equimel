from __future__ import annotations

import torch
from torch import nn


class FusedClassifier(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, hidden: int = 256) -> None:
        super().__init__()
        fused = vision_dim + text_dim
        self.bn = nn.BatchNorm1d(fused)
        self.hidden = nn.Linear(fused, hidden)
        self.act = nn.GELU()
        self.readout = nn.Linear(hidden, 1)

    def forward(self, vision_cls: torch.Tensor, text_cls: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([vision_cls, text_cls], dim=-1)
        normalized = self.bn(fused)
        logit: torch.Tensor = self.readout(self.act(self.hidden(normalized))).squeeze(-1)
        return logit
