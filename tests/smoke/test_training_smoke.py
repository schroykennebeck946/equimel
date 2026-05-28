from __future__ import annotations

from equimel.season.expedition import FederatedLoop
from tests.conftest import tiny_runspec


def test_two_rounds_reduce_training_loss() -> None:
    spec = tiny_runspec(rounds=2)
    loop = FederatedLoop(spec, seed=0)
    reports = loop.run()
    assert len(reports) == 2
    assert reports[-1].mean_train_loss < reports[0].mean_train_loss


def test_probe_metrics_in_range() -> None:
    spec = tiny_runspec(rounds=1)
    loop = FederatedLoop(spec, seed=0)
    loop.run()
    metrics = loop.evaluate(loop.federation.probe)
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert 0.0 <= metrics["equity_gap"] <= 1.0
    assert 0.0 <= metrics["worst_group_auroc"] <= 1.0


def test_aggregate_weights_normalize_each_round() -> None:
    spec = tiny_runspec(rounds=2)
    loop = FederatedLoop(spec, seed=0)
    for report in loop.run():
        assert abs(sum(report.aggregate_weights) - 1.0) < 1e-5
