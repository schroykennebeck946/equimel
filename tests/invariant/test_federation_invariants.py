from __future__ import annotations

import numpy as np
import torch

from equimel.bureau.consolidation import (
    EquitableAggregation,
    aggregate_states,
    is_personal_parameter,
    proportional_weights,
)
from equimel.ground.parcels import dirichlet_partition, partition_sizes
from equimel.ground.strata import estimate_ita, fitzpatrick_to_group, ita_to_fitzpatrick


def test_dirichlet_partition_is_a_disjoint_cover() -> None:
    labels = (np.arange(120) % 3 == 0).astype(np.int64)
    rng = np.random.default_rng(0)
    blocks = dirichlet_partition(labels, num_sites=4, alpha=0.5, rng=rng)
    union = np.concatenate(blocks)
    assert sorted(union.tolist()) == list(range(120))
    assert sum(partition_sizes(blocks)) == 120
    for first in range(len(blocks)):
        for second in range(first + 1, len(blocks)):
            assert np.intersect1d(blocks[first], blocks[second]).size == 0


def test_eap_upweights_worst_performing_site() -> None:
    aggregator = EquitableAggregation(tau_agg=0.1, drop_guard=0.05, drop_boost=2.0)
    weights = aggregator.weights(worst_group=[0.5, 0.9], sizes=[100, 100])
    assert abs(sum(weights) - 1.0) < 1e-6
    assert weights[0] > weights[1]


def test_eap_boost_on_catastrophic_drop() -> None:
    aggregator = EquitableAggregation(tau_agg=0.1, drop_guard=0.05, drop_boost=2.0)
    aggregator.weights(worst_group=[0.8, 0.8], sizes=[100, 100])
    after = aggregator.weights(worst_group=[0.7, 0.8], sizes=[100, 100])
    assert after[0] > after[1]


def test_proportional_weights_normalize() -> None:
    weights = proportional_weights([1, 3])
    assert abs(sum(weights) - 1.0) < 1e-9
    assert abs(weights[1] - 0.75) < 1e-9


def test_personal_parameter_detection() -> None:
    assert is_personal_parameter("classifier.bn.weight")
    assert is_personal_parameter("vision_projection.bn.running_mean")
    assert not is_personal_parameter("vision.encoder.layers.0.linear1.weight")


def test_aggregate_states_weighted_mean() -> None:
    states = [{"w": torch.zeros(2)}, {"w": torch.ones(2) * 4.0}]
    merged = aggregate_states(states, [0.25, 0.75])
    assert torch.allclose(merged["w"], torch.ones(2) * 3.0)


def test_ita_buckets_are_monotonic() -> None:
    sequence = [ita_to_fitzpatrick(value) for value in (60.0, 50.0, 35.0, 22.0, 14.0, 5.0)]
    assert sequence == [1, 2, 3, 4, 5, 6]
    assert [fitzpatrick_to_group(value) for value in (1, 2, 3, 4, 5, 6)] == [0, 0, 1, 1, 2, 2]


def test_estimate_ita_orders_by_brightness() -> None:
    bright = torch.stack(
        [torch.full((24, 24), 0.90), torch.full((24, 24), 0.75), torch.full((24, 24), 0.60)]
    )
    dark = torch.stack(
        [torch.full((24, 24), 0.45), torch.full((24, 24), 0.32), torch.full((24, 24), 0.25)]
    )
    assert estimate_ita(bright) > estimate_ita(dark)
