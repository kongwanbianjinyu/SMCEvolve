#!/usr/bin/env bash
# Run SMCEvolve on the problem.
set -euo pipefail

cd "$(dirname "$0")"

uv run python -m smcevolve.main \
  problem=circle_packing \
  algo=medium \