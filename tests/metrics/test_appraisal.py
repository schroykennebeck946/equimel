from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from equimel.atlas.accuracy import discrimination_report, safe_auroc
from equimel.atlas.disparity import (
    auroc_equity_gap,
    demographic_parity_difference,
    equalized_odds_difference,
    worst_group_auroc,
)
from equimel.atlas.significance import cohens_d, mcnemar_bonferroni, stratified_bootstrap_ci


def test_safe_auroc_matches_sklearn() -> None:
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2])
    labels = np.array([0, 0, 1, 1, 1, 0])
    assert abs(safe_auroc(scores, labels) - roc_auc_score(labels, scores)) < 1e-9


def test_safe_auroc_degenerate_returns_half() -> None:
    assert safe_auroc(np.array([0.2, 0.6]), np.array([1, 1])) == 0.5


def test_discrimination_perfect_separation() -> None:
    scores = np.array([0.05, 0.1, 0.9, 0.95])
    labels = np.array([0, 0, 1, 1])
    report = discrimination_report(scores, labels)
    assert report["auroc"] == 1.0
    assert report["sensitivity"] == 1.0
    assert report["specificity"] == 1.0
    assert report["f1"] == 1.0


def test_equity_gap_and_worst_group() -> None:
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.45, 0.55, 0.51, 0.49])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    gap = auroc_equity_gap(scores, labels, groups)
    worst = worst_group_auroc(scores, labels, groups)
    assert gap > 0.0
    assert worst <= 1.0 - gap + 1e-9


def test_equalized_odds_and_parity_bounds() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.95, 0.85, 0.6, 0.7])
    labels = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert 0.0 <= equalized_odds_difference(scores, labels, groups) <= 1.0
    assert 0.0 <= demographic_parity_difference(scores, groups) <= 1.0


def test_cohens_d_known_value() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([3.0, 4.0, 5.0, 6.0])
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    assert abs(cohens_d(a, b) - (-2.0 / float(pooled))) < 1e-9


def test_cohens_d_zero_variance_guard() -> None:
    constant = np.array([2.0, 2.0, 2.0, 2.0])
    assert cohens_d(constant, constant) == 0.0


def test_mcnemar_bonferroni_adjustment() -> None:
    correct_a = np.array([True, True, True, True, False, False, True, True])
    correct_b = np.array([False, False, False, False, False, False, True, True])
    result = mcnemar_bonferroni(correct_a, correct_b, num_comparisons=17)
    assert abs(result["adjusted_alpha"] - 0.05 / 17) < 1e-12
    assert result["p_value"] >= 0.0


def test_bootstrap_ci_orders_bounds() -> None:
    rng = np.random.default_rng(1)
    scores = rng.random(60)
    labels = (rng.random(60) < 0.5).astype(np.int64)
    strata = labels.copy()
    low, high = stratified_bootstrap_ci(scores, labels, strata, safe_auroc, resamples=200, seed=2)
    assert low <= high
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
