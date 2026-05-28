from __future__ import annotations

import torch
from torch import nn

from equimel.stores.logbook import get_logger

_LOG = get_logger("drafting.toponym")


class LanguageTower(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        dim: int,
        depth: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.position = nn.Parameter(torch.zeros(1, max_len, dim))
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
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        length = token_ids.shape[1]
        embedded = self.token_embedding(token_ids) + self.position[:, :length]
        padding = token_mask == 0
        encoded = self.encoder(embedded, src_key_padding_mask=padding)
        pooled: torch.Tensor = self.norm(encoded[:, 0])
        return pooled

    def warm_start(self, identifier: str) -> bool:
        try:
            from transformers import AutoModel

            source = AutoModel.from_pretrained(identifier)
        except Exception as error:
            _LOG.warning("text warm-start unavailable (%s); using local initialization", error)
            return False
        loaded = {k: v for k, v in source.state_dict().items() if k in self.state_dict()}
        self.load_state_dict(loaded, strict=False)
        _LOG.info("text warm-start matched %d tensors", len(loaded))
        return True
