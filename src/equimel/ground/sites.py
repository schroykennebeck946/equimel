from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from equimel.ground.legend import (
    ABCDE_FEATURES,
    ANATOMICAL_SITES,
    BENIGN_DIAGNOSES,
    LesionMeta,
    render_descriptor,
)
from equimel.ground.rectify import ImageTransform, build_transform
from equimel.ground.strata import fitzpatrick_to_group
from equimel.stores.notation import Batch

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class ShardLike(Protocol):
    labels: np.ndarray
    groups: np.ndarray

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]: ...


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    modality: str
    fitzpatrick_source: str
    text_mode: str
    positive_rate: float
    tone_bias: float


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "isic2019": DatasetSpec("isic2019", "dermoscopic", "ita_proxy", "class", 0.20, 0.25),
    "ham10000": DatasetSpec("ham10000", "dermatoscopic", "ita_proxy", "class", 0.18, 0.20),
    "fitzpatrick17k": DatasetSpec("fitzpatrick17k", "clinical", "expert", "patient", 0.30, 0.55),
    "ddi": DatasetSpec("ddi", "clinical", "expert", "patient", 0.40, 0.50),
    "pad_ufes_20": DatasetSpec("pad_ufes_20", "smartphone", "expert", "patient", 0.35, 0.60),
    "derm7pt": DatasetSpec("derm7pt", "dermoscopic", "expert", "patient", 0.32, 0.45),
    "slice3d": DatasetSpec("slice3d", "dermoscopic", "ita_proxy", "class", 0.12, 0.30),
}


class HashingTokenizer:
    def __init__(self, vocab_size: int, max_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_len = max_len

    def _token_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return 2 + value % (self.vocab_size - 2)

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        words = _TOKEN_RE.findall(text.lower())[: self.max_len - 1]
        ids = torch.zeros(self.max_len, dtype=torch.long)
        mask = torch.zeros(self.max_len, dtype=torch.long)
        ids[0] = 1
        mask[0] = 1
        for position, token in enumerate(words, start=1):
            ids[position] = self._token_id(token)
            mask[position] = 1
        return ids, mask


class SyntheticShard:
    def __init__(
        self,
        spec: DatasetSpec,
        size: int,
        image_size: int,
        num_groups: int,
        tokenizer: HashingTokenizer,
        transform: ImageTransform,
        seed: int,
    ) -> None:
        self.spec = spec
        self.size = size
        self.image_size = image_size
        self.num_groups = num_groups
        self.tokenizer = tokenizer
        self.transform = transform
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.labels = (rng.random(size) < spec.positive_rate).astype(np.int64)
        tone = rng.beta(spec.tone_bias * 6.0 + 1.0, (1.0 - spec.tone_bias) * 6.0 + 1.0, size=size)
        self.fitzpatrick = np.clip((tone * num_groups).astype(np.int64) + 1, 1, num_groups)
        self.groups = np.array(
            [fitzpatrick_to_group(int(f)) for f in self.fitzpatrick], dtype=np.int64
        )

    def __len__(self) -> int:
        return self.size

    def _meta(self, index: int) -> LesionMeta:
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        label = int(self.labels[index])
        if label == 1:
            diagnosis = "melanoma"
            pool = ABCDE_FEATURES[:6]
        else:
            diagnosis = str(rng.choice(BENIGN_DIAGNOSES))
            pool = ABCDE_FEATURES[3:]
        count = int(rng.integers(1, 4))
        chosen = tuple(str(f) for f in rng.choice(pool, size=count, replace=False))
        return LesionMeta(
            diagnosis=diagnosis,
            location=str(rng.choice(ANATOMICAL_SITES)),
            fitzpatrick=int(self.fitzpatrick[index]),
            features=chosen,
            age=int(rng.integers(20, 90)),
            sex=str(rng.choice(("female", "male"))),
        )

    def _image(self, index: int) -> torch.Tensor:
        rng = np.random.default_rng(self.seed * 7_700_001 + index)
        axis = np.linspace(0.0, 1.0, self.image_size, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
        phase = np.pi * (0.5 if self.labels[index] == 1 else 0.0)
        pattern = np.sin(2.0 * np.pi * (grid_x + grid_y) + phase)
        contrast = 0.6 * float(self.labels[index]) - 0.3
        noise = rng.normal(0.0, 1.0, size=(3, self.image_size, self.image_size)).astype(np.float32)
        canvas = 0.5 + 0.15 * contrast * pattern[None, :, :] + 0.05 * noise
        tone_scale = 1.0 - 0.13 * (float(self.fitzpatrick[index]) - 1.0)
        canvas = canvas * tone_scale
        return torch.from_numpy(canvas).clamp(0.0, 1.0).float()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        meta = self._meta(index)
        text = render_descriptor(meta, mode=self.spec.text_mode)
        ids, mask = self.tokenizer.encode(text)
        image = self.transform(self._image(index))
        return {
            "image": image,
            "token_ids": ids,
            "token_mask": mask,
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
            "group": torch.tensor(self.groups[index], dtype=torch.long),
        }


class PooledShard:
    def __init__(self, shards: list[SyntheticShard]) -> None:
        if not shards:
            raise ValueError("at least one shard is required")
        self.shards = shards
        self._offsets = np.cumsum([0] + [len(shard) for shard in shards])
        self.labels = np.concatenate([shard.labels for shard in shards])
        self.groups = np.concatenate([shard.groups for shard in shards])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index = int(np.searchsorted(self._offsets, index, side="right") - 1)
        local = index - int(self._offsets[shard_index])
        return self.shards[shard_index][local]


class IndexSubset:
    def __init__(self, shard: ShardLike, indices: np.ndarray) -> None:
        self.shard = shard
        self.indices = indices
        self.labels = shard.labels[indices]
        self.groups = shard.groups[indices]

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        return self.shard[int(self.indices[position])]


@dataclass
class FederatedData:
    train_sites: list[IndexSubset]
    val_sites: list[IndexSubset]
    probe: IndexSubset
    num_groups: int


def collate(items: list[dict[str, torch.Tensor]]) -> Batch:
    return Batch(
        images=torch.stack([item["image"] for item in items]),
        token_ids=torch.stack([item["token_ids"] for item in items]),
        token_mask=torch.stack([item["token_mask"] for item in items]),
        labels=torch.stack([item["label"] for item in items]),
        groups=torch.stack([item["group"] for item in items]),
    )


class MiniLoader:
    def __init__(
        self,
        source: IndexSubset,
        batch_size: int,
        shuffle: bool,
        seed: int,
        max_batches: int | None = None,
    ) -> None:
        self.source = source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.max_batches = max_batches
        self._order: np.ndarray = np.arange(len(source))
        self._cursor = 0
        self._emitted = 0

    def __iter__(self) -> MiniLoader:
        order = np.arange(len(self.source))
        if self.shuffle:
            np.random.default_rng(self.seed).shuffle(order)
        self._order = order
        self._cursor = 0
        self._emitted = 0
        return self

    def __next__(self) -> Batch:
        if self._cursor >= len(self._order):
            raise StopIteration
        if self.max_batches is not None and self._emitted >= self.max_batches:
            raise StopIteration
        chunk = self._order[self._cursor : self._cursor + self.batch_size]
        self._cursor += self.batch_size
        self._emitted += 1
        return collate([self.source[int(i)] for i in chunk])


def build_population(
    spec: DatasetSpec,
    size: int,
    image_size: int,
    num_groups: int,
    max_text_len: int,
    vocab_size: int,
    seed: int,
    train: bool,
) -> SyntheticShard:
    tokenizer = HashingTokenizer(vocab_size=vocab_size, max_len=max_text_len)
    transform = build_transform(image_size=image_size, train=train, seed=seed)
    return SyntheticShard(
        spec=spec,
        size=size,
        image_size=image_size,
        num_groups=num_groups,
        tokenizer=tokenizer,
        transform=transform,
        seed=seed,
    )
