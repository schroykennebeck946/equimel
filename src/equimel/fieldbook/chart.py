from __future__ import annotations

from pathlib import Path

from equimel.fieldbook._groundwork import launch
from equimel.season.expedition import FederatedLoop
from equimel.stores.dossier import ensure_directory
from equimel.stores.logbook import get_logger
from equimel.survey.commission import RunSpec

_LOG = get_logger("fieldbook.chart")


def run(runspec: RunSpec) -> None:
    workdir = ensure_directory(runspec.workdir)
    loop = FederatedLoop(runspec, seed=runspec.seeds[0])
    reports = loop.run()
    loop.save(Path(workdir) / "global_last.pt", round_index=len(reports))
    summary = loop.evaluate(loop.federation.probe)
    _LOG.info(
        "training complete rounds=%d auroc=%.4f equity_gap=%.4f worst=%.4f",
        len(reports),
        summary["auroc"],
        summary["equity_gap"],
        summary["worst_group_auroc"],
    )


def entry() -> None:
    launch(run)


if __name__ == "__main__":
    entry()
