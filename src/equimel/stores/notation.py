from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TypedDict

import torch

StateDict = dict[str, torch.Tensor]


class Mechanism(enum.Flag):
    NONE = 0
    VLA = enum.auto()
    DG = enum.auto()
    FAIRNESS = enum.auto()

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> Mechanism:
        merged = cls.NONE
        lookup = {"vla": cls.VLA, "dg": cls.DG, "fairness": cls.FAIRNESS}
        for token in tokens:
            merged |= lookup[token.strip().lower()]
        return merged


@dataclass
class Batch:
    images: torch.Tensor
    token_ids: torch.Tensor
    token_mask: torch.Tensor
    labels: torch.Tensor
    groups: torch.Tensor

    def to(self, device: torch.device) -> Batch:
        return Batch(
            images=self.images.to(device),
            token_ids=self.token_ids.to(device),
            token_mask=self.token_mask.to(device),
            labels=self.labels.to(device),
            groups=self.groups.to(device),
        )

    def __len__(self) -> int:
        return int(self.images.shape[0])


@dataclass
class SiteOutcome:
    site_id: int
    num_samples: int
    state: StateDict
    worst_group_auroc: float
    train_loss: float


@dataclass
class RoundReport:
    index: int
    aggregate_weights: list[float]
    worst_group_auroc: list[float]
    mean_train_loss: float
    history: dict[str, float] = field(default_factory=dict)


class DiscriminationScores(TypedDict):
    auroc: float
    sensitivity: float
    specificity: float
    balanced_accuracy: float
    f1: float


class EquityScores(TypedDict):
    equity_gap: float
    worst_group_auroc: float
    equalized_odds: float
    demographic_parity: float
