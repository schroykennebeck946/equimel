from __future__ import annotations

import torch

from equimel.drafting.datum import ProjectionHead
from equimel.drafting.grading import FusedClassifier
from equimel.drafting.montage import assemble_model
from equimel.drafting.relief import VisionTower
from tests.conftest import make_batch, tiny_runspec


def test_model_output_shapes() -> None:
    spec = tiny_runspec()
    model = assemble_model(spec)
    batch = make_batch(spec, count=6)
    output = model(batch)
    assert output.vision_embed.shape == (6, spec.embed_dim)
    assert output.text_embed.shape == (6, spec.embed_dim)
    assert output.logit.shape == (6,)
    assert output.temperature.ndim == 0


def test_projection_is_unit_norm() -> None:
    head = ProjectionHead(16, 8)
    head.eval()
    features = torch.randn(5, 16)
    norms = head(features).norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-4)


def test_vision_tower_pools_to_dim() -> None:
    tower = VisionTower(image_size=32, patch_size=16, dim=24, depth=1, heads=2)
    pooled = tower(torch.randn(3, 3, 32, 32))
    assert pooled.shape == (3, 24)


def test_classifier_emits_scalar_logit() -> None:
    classifier = FusedClassifier(vision_dim=12, text_dim=10)
    logit = classifier(torch.randn(4, 12), torch.randn(4, 10))
    assert logit.shape == (4,)
