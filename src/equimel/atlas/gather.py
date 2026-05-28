from __future__ import annotations

import numpy as np
import torch

from equimel.drafting.montage import FedVlmModel
from equimel.ground.sites import IndexSubset, MiniLoader


def collect_predictions(
    model: FedVlmModel,
    source: IndexSubset,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    was_training = model.training
    model.eval()
    loader = MiniLoader(source, batch_size, shuffle=False, seed=0, max_batches=max_batches)
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            located = batch.to(device)
            output = model(located)
            scores.append(torch.sigmoid(output.logit).detach().cpu().numpy())
            labels.append(located.labels.detach().cpu().numpy())
            groups.append(located.groups.detach().cpu().numpy())
    model.train(was_training)
    if not scores:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(groups)
