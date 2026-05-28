from __future__ import annotations

from pathlib import Path

from equimel.stores.notation import Mechanism
from equimel.survey.charting import materialize

_ROOT = Path(__file__).resolve().parents[2]


def _config(name: str) -> str:
    return str(_ROOT / "configs" / "experiment" / name)


def test_smoke_config_materializes() -> None:
    spec = materialize([_config("_smoke.gin")])
    assert spec.num_clients == 2
    assert spec.use_real_backbones is False
    assert "ddi" in spec.datasets


def test_main_config_resolves_includes() -> None:
    spec = materialize([_config("main.gin")])
    assert spec.num_clients == 5
    assert spec.rounds == 100
    assert spec.image_size == 224
    assert len(spec.seeds) == 15
    assert spec.uses(Mechanism.VLA)


def test_fl_only_ablation_disables_components() -> None:
    spec = materialize([_config("ablation_fl_only.gin")])
    assert spec.mechanisms == Mechanism.NONE
    assert spec.datasets == ("ddi", "fitzpatrick17k", "pad_ufes_20")
