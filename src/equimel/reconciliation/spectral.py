from __future__ import annotations

import torch


def normalize_distribution(amplitude: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flattened = amplitude.flatten().clamp(min=0.0) + eps
    return flattened / flattened.sum()


def frequency_divergence(
    local_amplitude: torch.Tensor,
    global_amplitude: torch.Tensor,
) -> torch.Tensor:
    local = normalize_distribution(local_amplitude)
    target = normalize_distribution(global_amplitude)
    return (local * (local.log() - target.log())).sum()
