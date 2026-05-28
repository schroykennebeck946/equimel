from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from equimel.atlas.accuracy import discrimination_report
from equimel.atlas.disparity import equity_report
from equimel.atlas.gather import collect_predictions
from equimel.bureau.consolidation import (
    EquitableAggregation,
    aggregate_states,
    proportional_weights,
)
from equimel.drafting.montage import assemble_model
from equimel.ground.consortium import build_federation
from equimel.ground.sites import FederatedData, IndexSubset, MiniLoader
from equimel.party.fieldwork import local_round
from equimel.party.fourier import LowFrequencyMixer
from equimel.stores.entropy import set_seed
from equimel.stores.hardware import resolve_device
from equimel.stores.logbook import get_logger
from equimel.stores.notation import Mechanism, RoundReport, StateDict
from equimel.survey.commission import RunSpec

_LOG = get_logger("season.expedition")


class FederatedLoop:
    def __init__(self, runspec: RunSpec, seed: int, datasets: Sequence[str] | None = None) -> None:
        set_seed(seed)
        self.runspec = runspec
        self.seed = seed
        self.device = resolve_device(runspec.device)
        self.federation: FederatedData = build_federation(runspec, seed, datasets)
        self.model = assemble_model(runspec).to(self.device)
        self.global_state: StateDict = self.model.global_state()
        self.personal_states: list[StateDict] = [
            self.model.personal_state() for _ in range(runspec.num_clients)
        ]
        self.aggregator = EquitableAggregation(
            runspec.tau_agg, runspec.drop_guard, runspec.drop_boost
        )
        self.mixer = LowFrequencyMixer(runspec.freq_ratio, runspec.freq_mix)
        self.global_block = self._global_amplitude()
        self._last_weights: list[float] = []

    def _global_amplitude(self) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        for site in self.federation.train_sites:
            loader = MiniLoader(site, self.runspec.batch_size, shuffle=False, seed=0, max_batches=1)
            for batch in loader:
                blocks.append(self.mixer.amplitude_block(batch.images.to(self.device)))
                break
        return torch.stack(blocks).mean(dim=0)

    def _aggregate(
        self, states: list[StateDict], worst: list[float], sizes: list[int]
    ) -> StateDict:
        if self.runspec.uses(Mechanism.FAIRNESS):
            weights = self.aggregator.weights(worst, sizes)
        else:
            weights = proportional_weights(sizes)
        self._last_weights = weights
        return aggregate_states(states, weights)

    def run(self, rounds: int | None = None) -> list[RoundReport]:
        horizon = rounds if rounds is not None else self.runspec.rounds
        reports: list[RoundReport] = []
        for index in range(horizon):
            outcomes = []
            for site_id in range(self.runspec.num_clients):
                self.model.load_global_state(self.global_state)
                self.model.load_state_dict(self.personal_states[site_id], strict=False)
                outcome = local_round(
                    model=self.model,
                    train=self.federation.train_sites[site_id],
                    val=self.federation.val_sites[site_id],
                    runspec=self.runspec,
                    global_block=self.global_block,
                    device=self.device,
                    seed=self.seed + index * 1000 + site_id,
                    site_id=site_id,
                )
                self.personal_states[site_id] = self.model.personal_state()
                outcomes.append(outcome)
            sizes = [outcome.num_samples for outcome in outcomes]
            worst = [outcome.worst_group_auroc for outcome in outcomes]
            self.global_state = self._aggregate([o.state for o in outcomes], worst, sizes)
            report = RoundReport(
                index=index,
                aggregate_weights=self._last_weights,
                worst_group_auroc=worst,
                mean_train_loss=sum(o.train_loss for o in outcomes) / len(outcomes),
            )
            if (index + 1) % self.runspec.eval_every == 0:
                report.history = self.evaluate(self.federation.probe)
            _LOG.info(
                "round %d mean_loss=%.4f worst=%s",
                index,
                report.mean_train_loss,
                [round(value, 3) for value in worst],
            )
            reports.append(report)
        return reports

    def evaluate(self, source: IndexSubset) -> dict[str, float]:
        self.model.load_global_state(self.global_state)
        scores, labels, groups = collect_predictions(
            self.model, source, self.runspec.batch_size, self.device
        )
        discrimination = discrimination_report(scores, labels)
        equity = equity_report(scores, labels, groups)
        return {
            "auroc": discrimination["auroc"],
            "sensitivity": discrimination["sensitivity"],
            "specificity": discrimination["specificity"],
            "balanced_accuracy": discrimination["balanced_accuracy"],
            "f1": discrimination["f1"],
            "equity_gap": equity["equity_gap"],
            "worst_group_auroc": equity["worst_group_auroc"],
            "equalized_odds": equity["equalized_odds"],
            "demographic_parity": equity["demographic_parity"],
        }

    def save(self, destination: str | Path, round_index: int) -> None:
        from equimel.season.archive import save_checkpoint

        save_checkpoint(
            destination, self.global_state, self.personal_states, round_index, self.seed
        )
