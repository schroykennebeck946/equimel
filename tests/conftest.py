from __future__ import annotations

import pytest

from equimel.ground.sites import DATASET_REGISTRY, build_population, collate
from equimel.stores.notation import Batch, Mechanism
from equimel.survey.commission import RunSpec


def tiny_runspec(**overrides: object) -> RunSpec:
    base: dict[str, object] = {
        "workdir": "runs/test",
        "seeds": (0,),
        "num_clients": 2,
        "rounds": 2,
        "local_epochs": 1,
        "batch_size": 8,
        "num_steps_per_epoch": 3,
        "samples_per_site": 40,
        "image_size": 32,
        "patch_size": 16,
        "vision_dim": 32,
        "vision_depth": 2,
        "vision_heads": 2,
        "text_dim": 32,
        "text_depth": 2,
        "text_heads": 2,
        "text_vocab": 128,
        "max_text_len": 16,
        "embed_dim": 32,
        "num_groups": 6,
        "temperature": 0.07,
        "lambda_vl": 0.5,
        "beta_freq": 0.3,
        "tau_agg": 0.1,
        "dro_step": 0.01,
        "swad_period": 1,
        "swad_warmup": 0,
        "freq_ratio": 0.25,
        "freq_mix": 0.5,
        "use_real_backbones": False,
        "device": "cpu",
        "eval_every": 1,
        "datasets": ("ddi", "fitzpatrick17k"),
        "mechanisms": Mechanism.VLA | Mechanism.DG | Mechanism.FAIRNESS,
    }
    base.update(overrides)
    return RunSpec(**base)


def make_batch(spec: RunSpec, count: int, seed: int = 0, name: str = "ddi") -> Batch:
    shard = build_population(
        spec=DATASET_REGISTRY[name],
        size=max(count, 2),
        image_size=spec.image_size,
        num_groups=spec.num_groups,
        max_text_len=spec.max_text_len,
        vocab_size=spec.text_vocab,
        seed=seed,
        train=True,
    )
    return collate([shard[index] for index in range(count)])


@pytest.fixture
def spec() -> RunSpec:
    return tiny_runspec()
