from __future__ import annotations

from collections.abc import Callable, Sequence

from absl import app, flags

from equimel.survey.charting import materialize
from equimel.survey.commission import RunSpec

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("config", [], "Path to a gin configuration file (repeatable).")
flags.DEFINE_multi_string("bind", [], "Inline gin binding override (repeatable).")


def runspec_from_flags() -> RunSpec:
    return materialize(FLAGS.config, FLAGS.bind)


def launch(handler: Callable[[RunSpec], None]) -> None:
    def _entrypoint(argv: Sequence[str]) -> None:
        del argv
        handler(runspec_from_flags())

    app.run(_entrypoint)
