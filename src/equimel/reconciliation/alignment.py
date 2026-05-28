from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class VisionLanguageContrast(nn.Module):
    def forward(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        if vision_embed.shape[0] < 2:
            return vision_embed.new_zeros(())
        similarity = vision_embed @ text_embed.transpose(0, 1)
        logits = similarity / temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        return functional.cross_entropy(logits, targets)
