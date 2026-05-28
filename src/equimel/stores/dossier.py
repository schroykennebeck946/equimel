from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def ensure_directory(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def atomic_save(payload: dict[str, Any], destination: str | os.PathLike[str]) -> Path:
    target = Path(destination)
    ensure_directory(target.parent)
    handle, staging = tempfile.mkstemp(dir=str(target.parent), suffix=".staging")
    os.close(handle)
    staging_path = Path(staging)
    try:
        torch.save(payload, staging_path)
        os.replace(staging_path, target)
    finally:
        if staging_path.exists():
            staging_path.unlink()
    return target


def load_payload(source: str | os.PathLike[str]) -> dict[str, Any]:
    loaded = torch.load(Path(source), map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint payload must be a mapping")
    return loaded
