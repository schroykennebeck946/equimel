from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar, cast

import gin

F = TypeVar("F", bound=Callable[..., object])


def configurable(target: F) -> F:
    return cast(F, gin.configurable(target))


def parse_configs(files: Sequence[str], bindings: Sequence[str]) -> None:
    gin.parse_config_files_and_bindings(list(files), list(bindings), finalize_config=False)


def operative_summary() -> str:
    return str(gin.operative_config_str())


def reset() -> None:
    gin.clear_config(clear_constants=True)
