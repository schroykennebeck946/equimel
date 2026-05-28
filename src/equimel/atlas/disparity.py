from __future__ import annotations

import numpy as np

from equimel.atlas.accuracy import safe_auroc
from equimel.stores.notation import EquityScores


def group_aurocs(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict[int, float]:
    table: dict[int, float] = {}
    for group in np.unique(groups):
        mask = groups == group
        if np.unique(labels[mask]).size < 2:
            continue
        table[int(group)] = safe_auroc(scores[mask], labels[mask])
    return table


def auroc_equity_gap(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    table = group_aurocs(scores, labels, groups)
    if len(table) < 2:
        return 0.0
    values = list(table.values())
    return float(max(values) - min(values))


def worst_group_auroc(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    table = group_aurocs(scores, labels, groups)
    if not table:
        return 0.5
    return float(min(table.values()))


def _rate(predictions: np.ndarray, mask: np.ndarray) -> float:
    selected = predictions[mask]
    if selected.size == 0:
        return 0.0
    return float(selected.mean())


def equalized_odds_difference(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
) -> float:
    predictions = (scores >= threshold).astype(np.int64)
    true_positive_rates: list[float] = []
    false_positive_rates: list[float] = []
    for group in np.unique(groups):
        in_group = groups == group
        true_positive_rates.append(_rate(predictions, in_group & (labels == 1)))
        false_positive_rates.append(_rate(predictions, in_group & (labels == 0)))
    if len(true_positive_rates) < 2:
        return 0.0
    tpr_gap = max(true_positive_rates) - min(true_positive_rates)
    fpr_gap = max(false_positive_rates) - min(false_positive_rates)
    return float(max(tpr_gap, fpr_gap))


def demographic_parity_difference(
    scores: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
) -> float:
    predictions = (scores >= threshold).astype(np.int64)
    rates = [_rate(predictions, groups == group) for group in np.unique(groups)]
    if len(rates) < 2:
        return 0.0
    return float(max(rates) - min(rates))


def equity_report(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> EquityScores:
    return EquityScores(
        equity_gap=auroc_equity_gap(scores, labels, groups),
        worst_group_auroc=worst_group_auroc(scores, labels, groups),
        equalized_odds=equalized_odds_difference(scores, labels, groups),
        demographic_parity=demographic_parity_difference(scores, groups),
    )
