from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from equimel.stores.notation import DiscriminationScores


def safe_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def discrimination_report(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> DiscriminationScores:
    predictions = (scores >= threshold).astype(np.int64)
    positives = labels == 1
    negatives = labels == 0
    true_positive = float(np.sum(predictions[positives] == 1))
    true_negative = float(np.sum(predictions[negatives] == 0))
    false_positive = float(np.sum(predictions[negatives] == 1))
    false_negative = float(np.sum(predictions[positives] == 0))
    sensitivity = true_positive / max(true_positive + false_negative, 1.0)
    specificity = true_negative / max(true_negative + false_positive, 1.0)
    f1_denominator = max(2 * true_positive + false_positive + false_negative, 1.0)
    return DiscriminationScores(
        auroc=safe_auroc(scores, labels),
        sensitivity=sensitivity,
        specificity=specificity,
        balanced_accuracy=0.5 * (sensitivity + specificity),
        f1=2 * true_positive / f1_denominator,
    )
