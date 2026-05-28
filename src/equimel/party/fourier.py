from __future__ import annotations

import torch


class LowFrequencyMixer:
    def __init__(self, ratio: float, mix: float) -> None:
        self.ratio = ratio
        self.mix = mix

    def _bounds(self, size: int) -> tuple[int, int]:
        width = max(1, int(size * self.ratio))
        start = (size - width) // 2
        return start, start + width

    def amplitude_block(self, images: torch.Tensor) -> torch.Tensor:
        batched = images if images.dim() == 4 else images.unsqueeze(0)
        spectrum = torch.fft.fftshift(torch.fft.fft2(batched, dim=(-2, -1)), dim=(-2, -1))
        amplitude = spectrum.abs().mean(dim=0)
        top, bottom = self._bounds(amplitude.shape[-2])
        left, right = self._bounds(amplitude.shape[-1])
        block: torch.Tensor = amplitude[..., top:bottom, left:right]
        return block

    def descriptor(self, images: torch.Tensor) -> torch.Tensor:
        return self.amplitude_block(images).flatten()

    def mix_into(self, image: torch.Tensor, global_block: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1))
        amplitude = spectrum.abs()
        phase = spectrum.angle()
        top, bottom = self._bounds(amplitude.shape[-2])
        left, right = self._bounds(amplitude.shape[-1])
        blended = (1.0 - self.mix) * amplitude[..., top:bottom, left:right]
        blended = blended + self.mix * global_block
        amplitude[..., top:bottom, left:right] = blended
        recomposed = amplitude * torch.exp(1j * phase)
        restored = torch.fft.ifft2(torch.fft.ifftshift(recomposed, dim=(-2, -1)), dim=(-2, -1))
        result: torch.Tensor = restored.real.to(image.dtype)
        return result
