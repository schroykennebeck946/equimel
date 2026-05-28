from __future__ import annotations

from collections.abc import Sequence

import torch

from equimel.stores.notation import StateDict

_PERSONAL_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def is_personal_parameter(name: str) -> bool:
    return ".bn." in name or name.endswith(_PERSONAL_SUFFIXES)


def proportional_weights(sizes: Sequence[int]) -> list[float]:
    total = float(sum(sizes))
    if total <= 0.0:
        uniform = 1.0 / max(len(sizes), 1)
        return [uniform for _ in sizes]
    return [float(size) / total for size in sizes]


class EquitableAggregation:
    def __init__(self, tau_agg: float, drop_guard: float, drop_boost: float) -> None:
        self.tau_agg = tau_agg
        self.drop_guard = drop_guard
        self.drop_boost = drop_boost
        self._previous: list[float] | None = None

    def reset(self) -> None:
        self._previous = None

    def weights(self, worst_group: Sequence[float], sizes: Sequence[int]) -> list[float]:
        scores = torch.tensor([-value / self.tau_agg for value in worst_group])
        softened = torch.softmax(scores, dim=0)
        weighted = softened * torch.tensor([float(size) for size in sizes])
        if self._previous is not None and len(self._previous) == len(worst_group):
            for index, current in enumerate(worst_group):
                if self._previous[index] - current >= self.drop_guard:
                    weighted[index] = weighted[index] * self.drop_boost
        total = float(weighted.sum().item())
        if total <= 0.0:
            normalized = [1.0 / len(worst_group) for _ in worst_group]
        else:
            normalized = [float(value / total) for value in weighted]
        self._previous = list(worst_group)
        return normalized


def aggregate_states(states: Sequence[StateDict], weights: Sequence[float]) -> StateDict:
    if not states:
        raise ValueError("no client states to aggregate")
    if len(states) != len(weights):
        raise ValueError("states and weights must align")
    merged: StateDict = {}
    for key in states[0]:
        stacked = torch.stack([states[index][key].float() for index in range(len(states))])
        coefficients = torch.tensor(list(weights), dtype=stacked.dtype).view(
            -1, *([1] * (stacked.dim() - 1))
        )
        merged[key] = (stacked * coefficients).sum(dim=0).to(states[0][key].dtype)
    return merged
