from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from equimel.drafting.datum import ProjectionHead
from equimel.drafting.grading import FusedClassifier
from equimel.drafting.relief import VisionTower
from equimel.drafting.toponym import LanguageTower
from equimel.stores.notation import Batch, StateDict
from equimel.survey.commission import RunSpec


@dataclass
class ModelOutput:
    vision_embed: torch.Tensor
    text_embed: torch.Tensor
    logit: torch.Tensor
    temperature: torch.Tensor


class FedVlmModel(nn.Module):
    def __init__(self, runspec: RunSpec) -> None:
        super().__init__()
        self.vision = VisionTower(
            image_size=runspec.image_size,
            patch_size=runspec.patch_size,
            dim=runspec.vision_dim,
            depth=runspec.vision_depth,
            heads=runspec.vision_heads,
        )
        self.language = LanguageTower(
            vocab_size=runspec.text_vocab,
            max_len=runspec.max_text_len,
            dim=runspec.text_dim,
            depth=runspec.text_depth,
            heads=runspec.text_heads,
        )
        self.vision_projection = ProjectionHead(runspec.vision_dim, runspec.embed_dim)
        self.text_projection = ProjectionHead(runspec.text_dim, runspec.embed_dim)
        self.classifier = FusedClassifier(runspec.vision_dim, runspec.text_dim)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(runspec.temperature)))

    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1e-3, max=1.0)

    def encode(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        vision_cls = self.vision(batch.images)
        text_cls = self.language(batch.token_ids, batch.token_mask)
        return vision_cls, text_cls

    def forward(self, batch: Batch) -> ModelOutput:
        vision_cls, text_cls = self.encode(batch)
        logit = self.classifier(vision_cls, text_cls)
        return ModelOutput(
            vision_embed=self.vision_projection(vision_cls),
            text_embed=self.text_projection(text_cls),
            logit=logit,
            temperature=self.temperature(),
        )

    def personal_parameter_names(self) -> set[str]:
        personal: set[str] = set()
        for module_name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                for param_name, _ in module.named_parameters(recurse=False):
                    personal.add(f"{module_name}.{param_name}")
                for buffer_name, _ in module.named_buffers(recurse=False):
                    personal.add(f"{module_name}.{buffer_name}")
        return personal

    def global_state(self) -> StateDict:
        personal = self.personal_parameter_names()
        return {k: v.detach().clone() for k, v in self.state_dict().items() if k not in personal}

    def personal_state(self) -> StateDict:
        personal = self.personal_parameter_names()
        return {k: v.detach().clone() for k, v in self.state_dict().items() if k in personal}

    def load_global_state(self, payload: StateDict) -> None:
        self.load_state_dict(payload, strict=False)


def assemble_model(runspec: RunSpec) -> FedVlmModel:
    model = FedVlmModel(runspec)
    if runspec.use_real_backbones:
        model.vision.warm_start("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        model.language.warm_start("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    return model
