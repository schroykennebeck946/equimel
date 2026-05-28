from __future__ import annotations

from pathlib import Path

import torch

from equimel.atlas.gather import collect_predictions
from equimel.fieldbook._groundwork import launch
from equimel.ground.consortium import build_dataset_probe
from equimel.season.archive import load_checkpoint
from equimel.season.expedition import FederatedLoop
from equimel.stores.logbook import get_logger
from equimel.survey.commission import RunSpec

_LOG = get_logger("fieldbook.locate")


def run(runspec: RunSpec) -> None:
    loop = FederatedLoop(runspec, seed=runspec.seeds[0])
    checkpoint = Path(runspec.workdir) / "global_last.pt"
    if checkpoint.exists():
        loop.global_state = load_checkpoint(checkpoint)["global_state"]
    loop.model.load_global_state(loop.global_state)
    target = runspec.datasets[0]
    probe = build_dataset_probe(runspec, target, seed=runspec.seeds[0])
    scores, labels, groups = collect_predictions(
        loop.model, probe, runspec.batch_size, loop.device, max_batches=1
    )
    for index in range(min(8, scores.shape[0])):
        verdict = "malignant" if scores[index] >= 0.5 else "benign"
        _LOG.info(
            "case=%d dataset=%s p_melanoma=%.3f fitzpatrick_group=%d call=%s truth=%d",
            index,
            target,
            float(scores[index]),
            int(groups[index]),
            verdict,
            int(labels[index]),
        )
    _LOG.info("mean melanoma probability over batch=%.3f", float(torch.tensor(scores).mean()))


def entry() -> None:
    launch(run)


if __name__ == "__main__":
    entry()
