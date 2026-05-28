from __future__ import annotations

from equimel.atlas.portability import cross_site_drop
from equimel.fieldbook._groundwork import launch
from equimel.ground.consortium import build_dataset_probe
from equimel.season.expedition import FederatedLoop
from equimel.stores.logbook import get_logger
from equimel.survey.commission import RunSpec

_LOG = get_logger("fieldbook.traverse")


def run(runspec: RunSpec) -> None:
    held_out = set(runspec.held_out)
    train_names = [name for name in runspec.datasets if name not in held_out]
    if not held_out:
        _LOG.warning("held_out is empty; bind build_runspec.held_out to enable leave-two-out")
        return
    loop = FederatedLoop(runspec, seed=runspec.seeds[0], datasets=train_names)
    loop.run()
    in_domain = [
        loop.evaluate(build_dataset_probe(runspec, name, seed=runspec.seeds[0]))["auroc"]
        for name in train_names
    ]
    reference = sum(in_domain) / len(in_domain)
    for name in runspec.held_out:
        ood = loop.evaluate(build_dataset_probe(runspec, name, seed=runspec.seeds[0]))["auroc"]
        _LOG.info(
            "held_out=%-16s in_domain=%.4f ood=%.4f drop=%.4f",
            name,
            reference,
            ood,
            cross_site_drop(reference, ood),
        )


def entry() -> None:
    launch(run)


if __name__ == "__main__":
    entry()
