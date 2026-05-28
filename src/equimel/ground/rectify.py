from __future__ import annotations

import torch
import torch.nn.functional as functional

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class ImageTransform:
    def __init__(self, image_size: int, train: bool, seed: int = 0) -> None:
        self.image_size = image_size
        self.train = train
        self._generator = torch.Generator().manual_seed(seed)
        self._mean = torch.tensor(_MEAN).view(3, 1, 1)
        self._std = torch.tensor(_STD).view(3, 1, 1)

    def _draw(self) -> float:
        return float(torch.rand(1, generator=self._generator).item())

    def _resize(self, image: torch.Tensor) -> torch.Tensor:
        resized = functional.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        if self._draw() < 0.5:
            image = torch.flip(image, dims=[2])
        if self._draw() < 0.5:
            image = torch.flip(image, dims=[1])
        quarter_turns = int(self._draw() * 4.0)
        if quarter_turns:
            image = torch.rot90(image, k=quarter_turns, dims=[1, 2])
        brightness = 0.8 + 0.4 * self._draw()
        contrast = 0.8 + 0.4 * self._draw()
        mean_value = image.mean()
        image = (image - mean_value) * contrast + mean_value * brightness
        return image.clamp(0.0, 1.0)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        resized = self._resize(image)
        if self.train:
            resized = self._augment(resized)
        return (resized - self._mean) / self._std


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(_STD, device=image.device).view(3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def build_transform(image_size: int, train: bool, seed: int = 0) -> ImageTransform:
    return ImageTransform(image_size=image_size, train=train, seed=seed)
