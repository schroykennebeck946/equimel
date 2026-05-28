from __future__ import annotations

from pathlib import Path

from equimel.fieldbook._groundwork import launch
from equimel.ground.consortium import build_dataset_probe
from equimel.season.archive import load_checkpoint
from equimel.season.expedition import FederatedLoop
from equimel.stores.logbook import get_logger
from equimel.survey.commission import RunSpec

_LOG = get_logger("fieldbook.review")


def run(runspec: RunSpec) -> None:
    loop = FederatedLoop(runspec, seed=runspec.seeds[0])
    checkpoint = Path(runspec.workdir) / "global_last.pt"
    if checkpoint.exists():
        loop.global_state = load_checkpoint(checkpoint)["global_state"]
        _LOG.info("loaded checkpoint %s", checkpoint)
    else:
        _LOG.warning("no checkpoint at %s; evaluating freshly initialized weights", checkpoint)
    for name in runspec.datasets:
        probe = build_dataset_probe(runspec, name, seed=runspec.seeds[0])
        scores = loop.evaluate(probe)
        _LOG.info(
            "%-16s auroc=%.4f equity_gap=%.4f worst=%.4f eqodds=%.4f dempar=%.4f",
            name,
            scores["auroc"],
            scores["equity_gap"],
            scores["worst_group_auroc"],
            scores["equalized_odds"],
            scores["demographic_parity"],
        )


def entry() -> None:
    launch(run)


if __name__ == "__main__":
    entry()
