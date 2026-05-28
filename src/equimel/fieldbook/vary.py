from __future__ import annotations

from equimel.fieldbook._groundwork import launch
from equimel.ground.consortium import build_dataset_probe
from equimel.season.expedition import FederatedLoop
from equimel.stores.logbook import get_logger
from equimel.survey.commission import RunSpec

_LOG = get_logger("fieldbook.vary")

_FAIRNESS_PANELS = ("ddi", "fitzpatrick17k", "pad_ufes_20")


def run(runspec: RunSpec) -> None:
    loop = FederatedLoop(runspec, seed=runspec.seeds[0])
    loop.run()
    parameters = sum(param.numel() for param in loop.model.parameters()) / 1e6
    _LOG.info("configuration mechanisms=%s params=%.1fM", runspec.mechanisms, parameters)
    for name in _FAIRNESS_PANELS:
        probe = build_dataset_probe(runspec, name, seed=runspec.seeds[0])
        scores = loop.evaluate(probe)
        _LOG.info(
            "%-16s auroc=%.4f equity_gap=%.4f worst=%.4f",
            name,
            scores["auroc"],
            scores["equity_gap"],
            scores["worst_group_auroc"],
        )


def entry() -> None:
    launch(run)


if __name__ == "__main__":
    entry()
