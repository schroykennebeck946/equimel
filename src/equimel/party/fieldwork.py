from __future__ import annotations

import math

import torch

from equimel.atlas.disparity import worst_group_auroc
from equimel.atlas.gather import collect_predictions
from equimel.drafting.montage import FedVlmModel
from equimel.ground.sites import IndexSubset, MiniLoader
from equimel.party.averaging import DenseWeightAverager
from equimel.party.fourier import LowFrequencyMixer
from equimel.reconciliation.alignment import VisionLanguageContrast
from equimel.reconciliation.equalization import GroupDRO, per_sample_bce
from equimel.reconciliation.omnibus import classification_loss
from equimel.reconciliation.spectral import frequency_divergence
from equimel.stores.notation import Mechanism, SiteOutcome
from equimel.survey.commission import RunSpec


def _validation_loss(model: FedVlmModel, probe: MiniLoader, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        for batch in probe:
            located = batch.to(device)
            value = classification_loss(model(located).logit, located.labels)
            model.train()
            return float(value.item())
    model.train()
    return float("inf")


def local_round(
    model: FedVlmModel,
    train: IndexSubset,
    val: IndexSubset,
    runspec: RunSpec,
    global_block: torch.Tensor | None,
    device: torch.device,
    seed: int,
    site_id: int,
) -> SiteOutcome:
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=runspec.learning_rate,
        weight_decay=runspec.weight_decay,
    )
    steps_per_epoch = max(
        1, min(runspec.num_steps_per_epoch, math.ceil(len(train) / runspec.batch_size))
    )
    total_steps = max(1, runspec.local_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    contrast = VisionLanguageContrast()
    dro = GroupDRO(runspec.num_groups, runspec.dro_step)
    mixer = LowFrequencyMixer(runspec.freq_ratio, runspec.freq_mix)
    averager = DenseWeightAverager(runspec.swad_period, runspec.swad_warmup)
    block = global_block.to(device) if global_block is not None else None
    descriptor = block.flatten() if block is not None else None

    running = 0.0
    seen = 0
    step = 0
    for epoch in range(runspec.local_epochs):
        loader = MiniLoader(
            train,
            runspec.batch_size,
            shuffle=True,
            seed=seed + epoch,
            max_batches=steps_per_epoch,
        )
        for batch in loader:
            located = batch.to(device)
            if len(located) < 2:
                continue
            if runspec.uses(Mechanism.DG) and block is not None:
                with torch.no_grad():
                    located.images = mixer.mix_into(located.images, block)
            output = model(located)
            if runspec.uses(Mechanism.FAIRNESS):
                base = dro.robust_loss(per_sample_bce(output.logit, located.labels), located.groups)
            else:
                base = classification_loss(output.logit, located.labels)
            loss = base
            if runspec.uses(Mechanism.VLA):
                vl = contrast(output.vision_embed, output.text_embed, output.temperature)
                loss = loss + runspec.lambda_vl * vl
            if runspec.uses(Mechanism.DG) and descriptor is not None:
                local_descriptor = mixer.descriptor(located.images)
                loss = loss + runspec.beta_freq * frequency_divergence(local_descriptor, descriptor)
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            optimizer.step()
            scheduler.step()
            running += float(loss.item())
            seen += 1
            step += 1
            if runspec.uses(Mechanism.DG):
                probe = MiniLoader(val, runspec.batch_size, shuffle=False, seed=seed, max_batches=1)
                averager.observe(step, model.state_dict(), _validation_loss(model, probe, device))

    if runspec.uses(Mechanism.DG):
        if not averager.has_average():
            averager.force(model.state_dict())
        model.load_state_dict(averager.consolidate(model.state_dict()))

    scores, labels, groups = collect_predictions(model, val, runspec.batch_size, device)
    worst = worst_group_auroc(scores, labels, groups)
    return SiteOutcome(
        site_id=site_id,
        num_samples=len(train),
        state=model.global_state(),
        worst_group_auroc=worst,
        train_loss=running / max(seen, 1),
    )
