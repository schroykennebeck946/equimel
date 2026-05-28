from __future__ import annotations

from collections.abc import Sequence

from equimel.survey.commission import RunSpec, build_runspec
from equimel.survey.sextant import parse_configs, reset


def materialize(
    configs: Sequence[str],
    bindings: Sequence[str] = (),
    fresh: bool = True,
) -> RunSpec:
    if fresh:
        reset()
    parse_configs(configs, bindings)
    return build_runspec()
