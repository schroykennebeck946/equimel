from __future__ import annotations

import torch
import torch.nn.functional as functional


def classification_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return functional.binary_cross_entropy_with_logits(logit, target)


def combine_local(cls_loss: torch.Tensor, vl_loss: torch.Tensor, lambda_vl: float) -> torch.Tensor:
    return cls_loss + lambda_vl * vl_loss


def combine_domain_generalization(
    dro_loss: torch.Tensor,
    frequency_loss: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return dro_loss + beta * frequency_loss


def combine_total(
    cls_loss: torch.Tensor,
    vl_loss: torch.Tensor,
    dg_loss: torch.Tensor,
    lambda_vl: float,
) -> torch.Tensor:
    return cls_loss + lambda_vl * vl_loss + dg_loss
