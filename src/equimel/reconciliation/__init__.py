from __future__ import annotations

from equimel.reconciliation.alignment import VisionLanguageContrast
from equimel.reconciliation.equalization import GroupDRO, per_sample_bce
from equimel.reconciliation.omnibus import (
    classification_loss,
    combine_domain_generalization,
    combine_local,
    combine_total,
)
from equimel.reconciliation.spectral import frequency_divergence, normalize_distribution

__all__ = [
    "GroupDRO",
    "VisionLanguageContrast",
    "classification_loss",
    "combine_domain_generalization",
    "combine_local",
    "combine_total",
    "frequency_divergence",
    "normalize_distribution",
    "per_sample_bce",
]
