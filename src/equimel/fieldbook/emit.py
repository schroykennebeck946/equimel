from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from equimel.drafting.montage import FedVlmModel, assemble_model
from equimel.fieldbook._groundwork import launch
from equimel.stores.dossier import ensure_directory
from equimel.stores.logbook import get_logger
from equimel.stores.notation import Batch
from equimel.survey.commission import RunSpec

_LOG = get_logger("fieldbook.emit")


class _Exportable(nn.Module):
    def __init__(self, model: FedVlmModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        rows = images.shape[0]
        batch = Batch(
            images=images,
            token_ids=token_ids,
            token_mask=token_mask,
            labels=torch.zeros(rows),
            groups=torch.zeros(rows, dtype=torch.long),
        )
        logit: torch.Tensor = self.model(batch).logit
        return logit


def run(runspec: RunSpec) -> None:
    model = assemble_model(runspec).eval()
    wrapper = _Exportable(model).eval()
    images = torch.randn(2, 3, runspec.image_size, runspec.image_size)
    token_ids = torch.randint(0, runspec.text_vocab, (2, runspec.max_text_len))
    token_mask = torch.ones(2, runspec.max_text_len, dtype=torch.long)
    destination = Path(ensure_directory(runspec.workdir)) / "model.onnx"
    try:
        torch.onnx.export(
            wrapper,
            (images, token_ids, token_mask),
            str(destination),
            input_names=["images", "token_ids", "token_mask"],
            output_names=["logit"],
            dynamic_axes={"images": {0: "batch"}, "logit": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        _LOG.info("exported ONNX graph to %s", destination)
    except Exception as error:
        _LOG.error("ONNX export failed (%s)", error)


def entry() -> None:
    launch(run)


if __name__ == "__main__":
    entry()
