#!/usr/bin/env bash
# Launch the SMCEvolve visualization server.
#   ./viz.sh                    # http://127.0.0.1:5173
#   ./viz.sh --port 8080
#   ./viz.sh --host 0.0.0.0     # bind all interfaces
set -euo pipefail

cd "$(dirname "$0")"
exec uv run python -m viz.server "$@"
