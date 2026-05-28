from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from equimel.ground.parcels import dirichlet_partition
from equimel.ground.sites import (
    DATASET_REGISTRY,
    FederatedData,
    IndexSubset,
    PooledShard,
    build_population,
)
from equimel.survey.commission import RunSpec


def _pool(names: Sequence[str], runspec: RunSpec, seed: int, train: bool) -> PooledShard:
    shards = [
        build_population(
            spec=DATASET_REGISTRY[name],
            size=runspec.samples_per_site,
            image_size=runspec.image_size,
            num_groups=runspec.num_groups,
            max_text_len=runspec.max_text_len,
            vocab_size=runspec.text_vocab,
            seed=seed + 31 * offset,
            train=train,
        )
        for offset, name in enumerate(names)
    ]
    return PooledShard(shards)


def build_federation(
    runspec: RunSpec,
    seed: int,
    datasets: Sequence[str] | None = None,
) -> FederatedData:
    names = list(datasets) if datasets is not None else list(runspec.datasets)
    pool = _pool(names, runspec, seed=seed, train=True)
    rng = np.random.default_rng(seed)
    assignment = dirichlet_partition(pool.labels, runspec.num_clients, runspec.dirichlet_alpha, rng)
    train_sites: list[IndexSubset] = []
    val_sites: list[IndexSubset] = []
    for indices in assignment:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        cut = max(1, int(0.8 * shuffled.size))
        val_indices = shuffled[cut:]
        if val_indices.size == 0:
            val_indices = shuffled[-1:]
        train_sites.append(IndexSubset(pool, shuffled[:cut]))
        val_sites.append(IndexSubset(pool, val_indices))
    probe_pool = _pool(names, runspec, seed=seed + 9973, train=False)
    probe = IndexSubset(probe_pool, np.arange(len(probe_pool)))
    return FederatedData(
        train_sites=train_sites,
        val_sites=val_sites,
        probe=probe,
        num_groups=runspec.num_groups,
    )


def build_dataset_probe(runspec: RunSpec, name: str, seed: int) -> IndexSubset:
    pool = _pool([name], runspec, seed=seed + 4441, train=False)
    return IndexSubset(pool, np.arange(len(pool)))
