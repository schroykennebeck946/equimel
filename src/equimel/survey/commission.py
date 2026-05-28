from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from equimel.stores.notation import Mechanism
from equimel.survey.sextant import configurable


@dataclass(frozen=True)
class RunSpec:
    workdir: str = "runs/default"
    seeds: tuple[int, ...] = (42, 123, 256, 512, 1024)
    num_clients: int = 5
    dirichlet_alpha: float = 0.5
    rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    image_size: int = 224
    patch_size: int = 16
    vision_dim: int = 768
    vision_depth: int = 12
    vision_heads: int = 12
    text_dim: int = 768
    text_depth: int = 12
    text_heads: int = 12
    text_vocab: int = 30522
    max_text_len: int = 64
    embed_dim: int = 512
    num_groups: int = 6
    num_steps_per_epoch: int = 64
    temperature: float = 0.07
    lambda_vl: float = 0.5
    beta_freq: float = 0.3
    tau_agg: float = 0.1
    dro_step: float = 0.01
    swad_period: int = 1
    swad_warmup: int = 1
    freq_ratio: float = 0.1
    freq_mix: float = 0.5
    drop_guard: float = 0.05
    drop_boost: float = 2.0
    datasets: tuple[str, ...] = (
        "isic2019",
        "ham10000",
        "fitzpatrick17k",
        "ddi",
        "pad_ufes_20",
        "derm7pt",
        "slice3d",
    )
    mechanisms: Mechanism = Mechanism.VLA | Mechanism.DG | Mechanism.FAIRNESS
    use_real_backbones: bool = True
    device: str = "auto"
    eval_every: int = 5
    samples_per_site: int = 256
    held_out: tuple[str, ...] = ()

    def uses(self, mechanism: Mechanism) -> bool:
        return bool(self.mechanisms & mechanism)

    def effective_batch(self) -> int:
        return self.batch_size * self.num_clients


@configurable
def build_runspec(
    workdir: str = "runs/default",
    seeds: Sequence[int] = (42, 123, 256, 512, 1024),
    num_clients: int = 5,
    dirichlet_alpha: float = 0.5,
    rounds: int = 100,
    local_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    patch_size: int = 16,
    vision_dim: int = 768,
    vision_depth: int = 12,
    vision_heads: int = 12,
    text_dim: int = 768,
    text_depth: int = 12,
    text_heads: int = 12,
    text_vocab: int = 30522,
    max_text_len: int = 64,
    embed_dim: int = 512,
    num_groups: int = 6,
    num_steps_per_epoch: int = 64,
    temperature: float = 0.07,
    lambda_vl: float = 0.5,
    beta_freq: float = 0.3,
    tau_agg: float = 0.1,
    dro_step: float = 0.01,
    swad_period: int = 1,
    swad_warmup: int = 1,
    freq_ratio: float = 0.1,
    freq_mix: float = 0.5,
    drop_guard: float = 0.05,
    drop_boost: float = 2.0,
    datasets: Sequence[str] = (
        "isic2019",
        "ham10000",
        "fitzpatrick17k",
        "ddi",
        "pad_ufes_20",
        "derm7pt",
        "slice3d",
    ),
    mechanisms: Sequence[str] | None = None,
    use_real_backbones: bool = True,
    device: str = "auto",
    eval_every: int = 5,
    samples_per_site: int = 256,
    held_out: Sequence[str] = (),
) -> RunSpec:
    enabled = (
        Mechanism.VLA | Mechanism.DG | Mechanism.FAIRNESS
        if mechanisms is None
        else Mechanism.from_tokens(list(mechanisms))
    )
    return RunSpec(
        workdir=workdir,
        seeds=tuple(seeds),
        num_clients=num_clients,
        dirichlet_alpha=dirichlet_alpha,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        image_size=image_size,
        patch_size=patch_size,
        vision_dim=vision_dim,
        vision_depth=vision_depth,
        vision_heads=vision_heads,
        text_dim=text_dim,
        text_depth=text_depth,
        text_heads=text_heads,
        text_vocab=text_vocab,
        max_text_len=max_text_len,
        embed_dim=embed_dim,
        num_groups=num_groups,
        num_steps_per_epoch=num_steps_per_epoch,
        temperature=temperature,
        lambda_vl=lambda_vl,
        beta_freq=beta_freq,
        tau_agg=tau_agg,
        dro_step=dro_step,
        swad_period=swad_period,
        swad_warmup=swad_warmup,
        freq_ratio=freq_ratio,
        freq_mix=freq_mix,
        drop_guard=drop_guard,
        drop_boost=drop_boost,
        datasets=tuple(datasets),
        mechanisms=enabled,
        use_real_backbones=use_real_backbones,
        device=device,
        eval_every=eval_every,
        samples_per_site=samples_per_site,
        held_out=tuple(held_out),
    )
