from __future__ import annotations

import torch
from torch import nn

from equimel.stores.logbook import get_logger

_LOG = get_logger("drafting.relief")


class VisionTower(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        grid = image_size // patch_size
        self.num_patches = grid * grid
        self.dim = dim
        self.patchify = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(patches.shape[0], -1, -1)
        sequence = torch.cat([cls, patches], dim=1) + self.position
        encoded = self.encoder(sequence)
        pooled: torch.Tensor = self.norm(encoded[:, 0])
        return pooled

    def warm_start(self, identifier: str) -> bool:
        try:
            from transformers import AutoModel

            source = AutoModel.from_pretrained(identifier)
        except Exception as error:
            _LOG.warning("vision warm-start unavailable (%s); using local initialization", error)
            return False
        loaded = {k: v for k, v in source.state_dict().items() if k in self.state_dict()}
        missing = self.load_state_dict(loaded, strict=False)
        _LOG.info("vision warm-start matched %d tensors", len(loaded))
        return not missing.unexpected_keys
