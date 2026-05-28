from __future__ import annotations

from equimel.atlas.accuracy import discrimination_report, safe_auroc
from equimel.atlas.disparity import (
    auroc_equity_gap,
    demographic_parity_difference,
    equalized_odds_difference,
    equity_report,
    group_aurocs,
    worst_group_auroc,
)
from equimel.atlas.gather import collect_predictions
from equimel.atlas.portability import average_drop, cross_site_drop
from equimel.atlas.significance import cohens_d, mcnemar_bonferroni, stratified_bootstrap_ci

__all__ = [
    "auroc_equity_gap",
    "average_drop",
    "cohens_d",
    "collect_predictions",
    "cross_site_drop",
    "demographic_parity_difference",
    "discrimination_report",
    "equalized_odds_difference",
    "equity_report",
    "group_aurocs",
    "mcnemar_bonferroni",
    "safe_auroc",
    "stratified_bootstrap_ci",
    "worst_group_auroc",
]
