from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats

Statistic = Callable[[np.ndarray, np.ndarray], float]


def stratified_bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray,
    statistic: Statistic,
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stratum_indices = [np.where(strata == value)[0] for value in np.unique(strata)]
    estimates = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        picked = np.concatenate(
            [block[rng.integers(0, block.size, block.size)] for block in stratum_indices]
        )
        estimates[draw] = statistic(scores[picked], labels[picked])
    lower = float(np.quantile(estimates, alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return lower, upper


def mcnemar_bonferroni(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    num_comparisons: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    discordant_b = float(np.sum(correct_a & ~correct_b))
    discordant_c = float(np.sum(~correct_a & correct_b))
    total = discordant_b + discordant_c
    if total == 0.0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (abs(discordant_b - discordant_c) - 1.0) ** 2 / total
        p_value = float(stats.chi2.sf(statistic, df=1))
    adjusted = alpha / max(num_comparisons, 1)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "adjusted_alpha": adjusted,
        "significant": float(p_value < adjusted),
    }


def cohens_d(values_a: np.ndarray, values_b: np.ndarray) -> float:
    mean_gap = float(np.mean(values_a) - np.mean(values_b))
    pooled = np.sqrt((np.var(values_a, ddof=1) + np.var(values_b, ddof=1)) / 2.0)
    if pooled < 1e-12:
        return 0.0
    return mean_gap / float(pooled)
