from __future__ import annotations

from equimel.fieldbook._groundwork import launch
from equimel.season.expedition import FederatedLoop
from equimel.stores.logbook import get_logger
from equimel.survey.commission import RunSpec

_LOG = get_logger("fieldbook.sweep")


def run(runspec: RunSpec) -> None:
    loop = FederatedLoop(runspec, seed=runspec.seeds[0])
    loop.run()
    scores = loop.evaluate(loop.federation.probe)
    _LOG.info(
        "clients=%d alpha=%.2f auroc=%.4f equity_gap=%.4f worst=%.4f",
        runspec.num_clients,
        runspec.dirichlet_alpha,
        scores["auroc"],
        scores["equity_gap"],
        scores["worst_group_auroc"],
    )


def entry() -> None:
    launch(run)


if __name__ == "__main__":
    entry()
