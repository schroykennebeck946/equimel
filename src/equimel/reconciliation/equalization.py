from __future__ import annotations

import torch
import torch.nn.functional as functional


class GroupDRO:
    def __init__(self, num_groups: int, step: float) -> None:
        self.num_groups = num_groups
        self.step = step
        self.weights = torch.ones(num_groups) / num_groups

    def reset(self) -> None:
        self.weights = torch.ones(self.num_groups) / self.num_groups

    def robust_loss(self, sample_losses: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        present = torch.unique(groups)
        group_losses = sample_losses.new_zeros(self.num_groups)
        active = sample_losses.new_zeros(self.num_groups)
        for group in present:
            index = int(group.item())
            mask = groups == group
            group_losses[index] = sample_losses[mask].mean()
            active[index] = 1.0
        detached = group_losses.detach()
        weights = self.weights.to(sample_losses.device)
        updated = weights * torch.exp(self.step * detached) * active
        if float(updated.sum().item()) <= 0.0:
            return sample_losses.mean()
        updated = updated / updated.sum()
        self.weights = updated.detach().cpu()
        return (updated * group_losses).sum()


def per_sample_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
