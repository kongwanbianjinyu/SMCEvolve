# SMCEvolve

Sequential Monte Carlo sampler for LLM-driven program evolution. Islands of
particles (candidate programs) are mutated by an LLM proposer, re-weighted by
a task-specific evaluator, and periodically migrated between islands.

## Quick start

### 1. Install dependencies with `uv`

The project uses [uv](https://docs.astral.sh/uv/) for environment management.
Install uv once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Sync the environment (creates `.venv/` from `pyproject.toml` + `uv.lock`) and
activate it:

```bash
uv sync
source .venv/bin/activate
```

All subsequent `python` / `pip` commands use the project venv.

### 2. Configure API credentials

Copy the example env file and fill in your own key:

```bash
cp .env.example .env
# edit .env
```

`.env` holds:

```
OPENAI_API_KEY=sk-...
API_BASE_URL=https://litellm.cloud.osu.edu # or https://api.openai.com/v1 or any OpenAI-compatible endpoint
```

Any OpenAI-compatible endpoint works (OpenAI, Azure, LiteLLM, local vLLM,
Ollama, etc.). `.env` is gitignored — do not commit real keys.

### 3. Run

With the venv active:

```bash
python -m smcevolve.main problem=circle_packing algo=medium
```

Override any Hydra config from the CLI. Examples:

```bash
# Swap problems and algo sizes
python -m smcevolve.main problem=target_value    algo=small
python -m smcevolve.main problem=blackbox_optimization algo=large

# Change specific parameters
python -m smcevolve.main problem=circle_packing algo=medium \
    algo.max_iterations=30 algo.n_islands=4 seed=7
```

Available presets:

| Group     | Options                                                        |
|-----------|----------------------------------------------------------------|
| `problem` | `circle_packing`, `target_value`, `blackbox_optimization`      |
| `algo`    | `smc`, `small`, `medium`, `large`                              |
| `llm`     | `openai`                                                       |

See [`configs/`](configs/) for the full config tree.

## Outputs

Every run writes to a timestamped directory:

```
outputs/<problem>/<YYYY-MM-DD_HH-MM-SS>/
├── events.jsonl      # one JSON record per event (proposals, resamples, migrations, final)
├── main.log          # human-readable log
└── event_logs/
    ├── island_0/     # per-island event streams
    ├── island_1/
    └── _final.json   # best program + summary
```

The visualization server reads `events.jsonl` directly.

## Visualize

Launch the Flask-based viewer (venv active):

```bash
python -m viz.server                    # http://127.0.0.1:5173
python -m viz.server --port 8080
python -m viz.server --host 0.0.0.0     # bind all interfaces
```

It lists every run under `outputs/` (newest first) and renders the event
stream — reward trajectories, particle populations, LLM cost, best program.

## Add your own problem

A problem is just three files in `problems/<your_problem>/`:

1. **`task.md`** — natural-language description shown to the LLM.
2. **`initial_program.py`** — seed program that the evolver mutates.
3. **`evaluator.py`** — must define `evaluate(program: str) -> float`. Return
   a scalar reward (higher = better). Return `0.0` on any failure so malformed
   programs don't crash the run.

Then register it as a Hydra config at `configs/problem/<your_problem>.yaml`:

```yaml
name: your_problem
dir: problems/your_problem
initial_program: initial_program.py
evaluator: evaluator.py
task_file: task.md
timeout: 15.0        # seconds per candidate evaluation
```

Run it:

```bash
python -m smcevolve.main problem=your_problem algo=medium
```

### Minimal `evaluator.py` template

```python
def evaluate(program: str) -> float:
    namespace: dict = {"__name__": "__main__"}
    try:
        exec(program, namespace)
        result = namespace.get("result")
        return float(result) if result is not None else 0.0
    except Exception:
        return 0.0
```

See [`problems/circle_packing/evaluator.py`](problems/circle_packing/evaluator.py)
and [`problems/target_value/evaluator.py`](problems/target_value/evaluator.py)
for working examples.

## Project layout

```
smcevolve/          # core library: controller, islands, proposer, prompts
configs/            # Hydra configs (algo / llm / problem groups)
problems/           # task definitions (initial program + evaluator + task.md)
viz/                # Flask viz server + static frontend
outputs/            # run artifacts (gitignored)
docs/               # design notes
```

## Docs

See [`docs/SMCEvolve_Technical_Document.md`](docs/SMCEvolve_Technical_Document.md)
for the algorithmic details (annealing schedule, kernel mixing, MAP-Elites
inspiration selection, migration).
