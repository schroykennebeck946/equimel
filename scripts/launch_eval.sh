#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiment/main.gin}"
WORKDIR="${2:-runs/main}"

python -m equimel.fieldbook.review \
    --config "${CONFIG}" \
    --bind "build_runspec.workdir='${WORKDIR}'"
