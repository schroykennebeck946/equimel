from __future__ import annotations

import math

import torch

_RGB_TO_XYZ = (
    (0.4124, 0.3576, 0.1805),
    (0.2126, 0.7152, 0.0722),
    (0.0193, 0.1192, 0.9505),
)
_WHITE = (0.95047, 1.0, 1.08883)
_ITA_EDGES = (55.0, 41.0, 28.0, 19.0, 10.0)


def _linearize(channel: torch.Tensor) -> torch.Tensor:
    threshold = channel <= 0.04045
    low = channel / 12.92
    high = ((channel + 0.055) / 1.055).clamp(min=0.0) ** 2.4
    return torch.where(threshold, low, high)


def _lab_f(ratio: torch.Tensor) -> torch.Tensor:
    delta = 6.0 / 29.0
    cube = ratio > delta**3
    root = ratio.clamp(min=1e-8) ** (1.0 / 3.0)
    linear = ratio / (3 * delta**2) + 4.0 / 29.0
    return torch.where(cube, root, linear)


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    linear = _linearize(image.clamp(0.0, 1.0))
    red, green, blue = linear[0], linear[1], linear[2]
    x = _RGB_TO_XYZ[0][0] * red + _RGB_TO_XYZ[0][1] * green + _RGB_TO_XYZ[0][2] * blue
    y = _RGB_TO_XYZ[1][0] * red + _RGB_TO_XYZ[1][1] * green + _RGB_TO_XYZ[1][2] * blue
    z = _RGB_TO_XYZ[2][0] * red + _RGB_TO_XYZ[2][1] * green + _RGB_TO_XYZ[2][2] * blue
    fx = _lab_f(x / _WHITE[0])
    fy = _lab_f(y / _WHITE[1])
    fz = _lab_f(z / _WHITE[2])
    lightness = 116.0 * fy - 16.0
    a_axis = 500.0 * (fx - fy)
    b_axis = 200.0 * (fy - fz)
    return torch.stack([lightness, a_axis, b_axis], dim=0)


def estimate_ita(image: torch.Tensor, border: float = 0.18) -> float:
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError("expected a [3, H, W] image tensor")
    height, width = image.shape[1], image.shape[2]
    band_h = max(1, int(round(height * border)))
    band_w = max(1, int(round(width * border)))
    mask = torch.zeros(height, width, dtype=torch.bool)
    mask[:band_h, :] = True
    mask[-band_h:, :] = True
    mask[:, :band_w] = True
    mask[:, -band_w:] = True
    ring = image[:, mask]
    mean_rgb = ring.mean(dim=1, keepdim=True).unsqueeze(-1)
    lab = rgb_to_lab(mean_rgb).flatten()
    lightness = float(lab[0].item())
    b_axis = float(lab[2].item())
    if abs(b_axis) < 1e-6:
        b_axis = math.copysign(1e-6, b_axis if b_axis != 0 else 1.0)
    return math.degrees(math.atan((lightness - 50.0) / b_axis))


def ita_to_fitzpatrick(ita: float) -> int:
    for index, edge in enumerate(_ITA_EDGES):
        if ita > edge:
            return index + 1
    return 6


def fitzpatrick_to_group(fitzpatrick: int) -> int:
    return min(2, (fitzpatrick - 1) // 2)


def proxy_group(image: torch.Tensor) -> int:
    return ita_to_fitzpatrick(estimate_ita(image))
