from __future__ import annotations

import os
from typing import Any

from equimel.stores.dossier import atomic_save, load_payload
from equimel.stores.notation import StateDict


def save_checkpoint(
    destination: str | os.PathLike[str],
    global_state: StateDict,
    personal_states: list[StateDict],
    round_index: int,
    seed: int,
) -> None:
    payload: dict[str, Any] = {
        "global_state": global_state,
        "personal_states": personal_states,
        "round_index": round_index,
        "seed": seed,
    }
    atomic_save(payload, destination)


def load_checkpoint(source: str | os.PathLike[str]) -> dict[str, Any]:
    return load_payload(source)
