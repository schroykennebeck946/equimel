from __future__ import annotations

import torch

from equimel.drafting.montage import assemble_model
from equimel.reconciliation.alignment import VisionLanguageContrast
from equimel.reconciliation.omnibus import classification_loss
from tests.conftest import make_batch, tiny_runspec


def test_all_parameters_receive_finite_gradients() -> None:
    spec = tiny_runspec()
    model = assemble_model(spec)
    batch = make_batch(spec, count=8)
    output = model(batch)
    contrast = VisionLanguageContrast()
    loss = classification_loss(output.logit, batch.labels)
    loss = loss + contrast(output.vision_embed, output.text_embed, output.temperature)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_temperature_parameter_is_trainable() -> None:
    spec = tiny_runspec()
    model = assemble_model(spec)
    batch = make_batch(spec, count=8)
    output = model(batch)
    contrast = VisionLanguageContrast()
    contrast(output.vision_embed, output.text_embed, output.temperature).backward()
    assert model.log_temperature.grad is not None
    assert torch.isfinite(model.log_temperature.grad).all()
