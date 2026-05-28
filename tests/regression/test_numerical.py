from __future__ import annotations

import torch

from equimel.party.fourier import LowFrequencyMixer
from equimel.reconciliation.alignment import VisionLanguageContrast
from equimel.reconciliation.spectral import frequency_divergence, normalize_distribution


def test_normalize_distribution_sums_to_one() -> None:
    raw = torch.tensor([1.0, 2.0, 3.0, 4.0])
    distribution = normalize_distribution(raw)
    assert abs(float(distribution.sum()) - 1.0) < 1e-6


def test_frequency_divergence_zero_for_identical() -> None:
    amplitude = torch.rand(32)
    assert float(frequency_divergence(amplitude, amplitude)) < 1e-6


def test_contrastive_loss_minimal_when_aligned() -> None:
    embed = torch.eye(4)
    contrast = VisionLanguageContrast()
    value = contrast(embed, embed, torch.tensor(0.07))
    assert float(value) < 1e-3


def test_amplitude_roundtrip_is_identity() -> None:
    torch.manual_seed(0)
    image = torch.rand(3, 32, 32)
    mixer = LowFrequencyMixer(ratio=0.25, mix=1.0)
    block = mixer.amplitude_block(image)
    restored = mixer.mix_into(image, block)
    assert torch.allclose(restored, image, atol=1e-4)


def test_descriptor_dimension_tracks_ratio() -> None:
    mixer = LowFrequencyMixer(ratio=0.5, mix=0.5)
    descriptor = mixer.descriptor(torch.rand(2, 3, 32, 32))
    assert descriptor.numel() == 3 * 16 * 16
