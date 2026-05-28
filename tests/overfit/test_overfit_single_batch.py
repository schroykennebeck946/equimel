from __future__ import annotations

import torch

from equimel.drafting.montage import assemble_model
from equimel.reconciliation.omnibus import classification_loss
from tests.conftest import make_batch, tiny_runspec


def test_classifier_memorizes_one_batch() -> None:
    torch.manual_seed(0)
    spec = tiny_runspec()
    model = assemble_model(spec)
    batch = make_batch(spec, count=16, seed=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    first = classification_loss(model(batch).logit, batch.labels).item()
    last = first
    for _ in range(250):
        optimizer.zero_grad()
        loss = classification_loss(model(batch).logit, batch.labels)
        loss.backward()
        optimizer.step()
        last = float(loss.item())
    assert last < first
    assert last < 0.1
