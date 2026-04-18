"""Flask server for the SMCEvolve visualization.

Discovers runs under <repo>/outputs/<problem>/<timestamp>/events.jsonl and
serves them as JSON to the static frontend in viz/static.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flask import Flask, abort, jsonify, send_from_directory

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUTS_DIR = REPO_ROOT / "outputs"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/runs")
def list_runs():
    runs = []
    if OUTPUTS_DIR.exists():
        for events_file in OUTPUTS_DIR.glob("**/events.jsonl"):
            run_dir = events_file.parent
            try:
                rel = run_dir.relative_to(OUTPUTS_DIR)
            except ValueError:
                continue
            stat = events_file.stat()
            runs.append(
                {
                    "id": str(rel).replace("\\", "/"),
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return jsonify(runs)


@app.route("/api/run/<path:run_id>")
def get_run(run_id: str):
    run_dir = (OUTPUTS_DIR / run_id).resolve()
    if not str(run_dir).startswith(str(OUTPUTS_DIR.resolve())):
        abort(403)
    events_file = run_dir / "events.jsonl"
    if not events_file.is_file():
        abort(404)
    records = []
    with events_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return jsonify({"run_id": run_id, "events": records})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173)
    args = parser.parse_args()
    print(f"SMCEvolve viz: http://{args.host}:{args.port}")
    print(f"Watching: {OUTPUTS_DIR}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
