from __future__ import annotations

import numpy as np


def dirichlet_partition(
    labels: np.ndarray,
    num_sites: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if num_sites < 1:
        raise ValueError("num_sites must be positive")
    buckets: list[list[int]] = [[] for _ in range(num_sites)]
    for value in np.unique(labels):
        members = np.where(labels == value)[0]
        rng.shuffle(members)
        proportions = rng.dirichlet(np.full(num_sites, alpha))
        cut_points = (np.cumsum(proportions) * len(members)).astype(int)[:-1]
        for site_index, chunk in enumerate(np.split(members, cut_points)):
            buckets[site_index].extend(int(i) for i in chunk)
    assignment = [np.array(sorted(indices), dtype=np.int64) for indices in buckets]
    _rebalance_empty(assignment, rng)
    return assignment


def _rebalance_empty(assignment: list[np.ndarray], rng: np.random.Generator) -> None:
    for site_index, indices in enumerate(assignment):
        if indices.size > 0:
            continue
        donor = int(np.argmax([block.size for block in assignment]))
        if assignment[donor].size < 2:
            continue
        moved = rng.choice(assignment[donor], size=1)
        assignment[donor] = np.setdiff1d(assignment[donor], moved)
        assignment[site_index] = moved


def partition_sizes(assignment: list[np.ndarray]) -> list[int]:
    return [int(block.size) for block in assignment]
