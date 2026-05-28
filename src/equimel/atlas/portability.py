from __future__ import annotations

from collections.abc import Sequence


def cross_site_drop(in_domain_auroc: float, ood_auroc: float) -> float:
    return in_domain_auroc - ood_auroc


def average_drop(in_domain: Sequence[float], ood: Sequence[float]) -> float:
    if len(in_domain) != len(ood) or not in_domain:
        raise ValueError("in-domain and ood sequences must align and be non-empty")
    drops = [cross_site_drop(a, b) for a, b in zip(in_domain, ood, strict=True)]
    return sum(drops) / len(drops)
