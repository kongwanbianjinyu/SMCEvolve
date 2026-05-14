# SMCEvolve Experiment Reproduction Guide

Walking top-to-bottom through this file lets you reproduce SMCEvolve from
scratch on all 4 problem categories. All commands assume you are at the
**repo root** (`SMCEvolve/`).

---

## 1. Problem inventory

There are **148** registered problems in total, split across 4 categories:

| Category       | Count | Hydra name prefix | Per-candidate characteristics                                | Resource    |
|----------------|-------|-------------------|--------------------------------------------------------------|-------------|
| `math`         | 10    | `math_*`          | NumPy or JAX; 0.5–30 s per candidate                         | CPU         |
| `algotune`     | 8     | `algotune_*`      | Wall-clock time as reward (`speedup_score`); **extremely sensitive to CPU contention** | CPU (serial)|
| `symreg`       | 129   | `symreg_*`        | BFGS fitting; ~1–5 s per candidate (many, each fast)         | CPU         |
| `autoresearch` | 1     | `autoresearch`    | 60+ s GPU training per candidate; **exclusive 24 GB A5000**  | GPU         |

List them at any time:

```bash
./experiments/run_exp.sh problems            # all
./experiments/run_exp.sh problems symreg     # one category
```

---

## 2. Setup

### 2.1 Install uv and sync the Python environment

The project uses [uv](https://docs.astral.sh/uv/) to manage the venv. Install
uv once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate the venv from the repo root (`uv sync` reads
`pyproject.toml` + `uv.lock`):

```bash
uv sync
source .venv/bin/activate
```

All subsequent `python` / `pip` commands then use `.venv` by default.

### 2.2 Configure LLM API credentials

```bash
cp .env.example .env
# edit .env: fill in OPENAI_API_KEY and API_BASE_URL
```

Example `.env` contents (any OpenAI-compatible endpoint works: OpenAI / Azure /
LiteLLM / vLLM / Ollama / …):

```
OPENAI_API_KEY=sk-...
API_BASE_URL=https://litellm.cloud.osu.edu
```

`.env` is gitignored and will not be committed.

### 2.3 Install per-category extra dependencies

You only need to do this once per category.

```bash
# (a) math — the three autocorr problems use JAX; the others run on NumPy alone
uv pip install jax optax sympy

# (b) symreg — materialize data for the 129 tasks + generate Hydra configs
uv pip install datasets huggingface_hub h5py sympy scipy scikit-learn pyyaml
python problems/symbolic_regression/data_api.py
python problems/symbolic_regression/generate_smc_configs.py
# Afterwards configs/problem/symreg_*.yaml should contain ~129 entries

# (c) algotune
# Clone one directory above SMCEvolve
git clone https://github.com/oripress/AlgoTune ../AlgoTune

# Install AlgoTune's own Python deps (jax / cvxpy / pulp / pot / numba / scikit-learn ...)
uv pip install -r problems/algotune/requirements.txt

# The 8 algotune task folders are already materialized in the repo
# (`problems/algotune/<task>/`); no need to regenerate.
# To regenerate or add a task, see [`problems/algotune/README.md`](problems/algotune/README.md).


# (d) autoresearch — fetch data + train tokenizer
uv pip install -r problems/autoresearch/requirements.txt
python problems/autoresearch/prepare.py
# On older CUDA drivers (cu124), swap torch for the cu124 build:
# uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124


```


### 2.4 Verify each category runs end-to-end

For each category pick the shortest problem and run it under `algo=small`
(each should finish in a few minutes):

```bash
./experiments/run_exp.sh single math_kissing_number
./experiments/run_exp.sh single algotune_affine_transform_2d
./experiments/run_exp.sh single symreg_phys_osc_PO0
./experiments/run_exp.sh single autoresearch
```

Once all four return `ok`, proceed to the large-scale runs.

---

## 4. Running a whole category

The most common entry point. `run_exp.sh` handles evaluator-side resource
contention, LLM API concurrency, and output isolation. The default budget is
`algo=small`.

```bash
./experiments/run_exp.sh category math           # 10 problems, parallel=3
./experiments/run_exp.sh category symreg         # 129 problems, parallel=4
./experiments/run_exp.sh category algotune       # 8 problems, forced serial (time-sensitive)
AR_GPUS=0,1,2,3 ./experiments/run_exp.sh category autoresearch   # 1 problem, exclusive GPU 0
```

**Default concurrency** is preset per category:

| Category       | Default `PARALLEL` | Max in-flight LLM requests when combined with `LLM_CONCURRENCY=4` |
|----------------|--------------------|--------------------------------------------------------------------|
| `math`         | 3                  | 12                                                                 |
| `symreg`       | 4                  | 16                                                                 |
| `algotune`     | 1 (forced)         | 4                                                                  |
| `autoresearch` | 1 (forced)         | 4                                                                  |

`PARALLEL=<n>` can override this, **but raising algotune above 1 distorts the
wall-clock reward — change with care**.

Sweep all 4 categories (recommended overnight run, ~4–8 h, `algo=small`):

```bash
SWEEP_TAG=full_$(date +%Y%m%d) ./experiments/run_exp.sh all
```

Internally, `all` runs in a fixed order:

```
symreg (CPU, parallel 4) → math (CPU, parallel 3) → algotune (CPU, serial) → autoresearch (GPU)
```

— so CPU and GPU work don't contend, and algotune's timing measurements aren't
polluted by symreg/math.

Follow progress live:

```bash
tail -f outputs/_sweep/full_$(date +%Y%m%d)/_sweep.log
```

---

## 5. Output layout

Each sweep produces two kinds of directories:

```
outputs/
├── _sweep/<TAG>/                          # sweep-level metadata
│   ├── _sweep.log                         # consolidated start/ok/FAIL log (the main summary)
│   └── <problem>.log                      # stdout+stderr for each problem
│
└── <category>/<problem>/sweep_<TAG>/      # per-run output (also the Hydra run dir)
    ├── events.jsonl                       # SMCEvolve event stream (consumed by viz/)
    ├── main.log                           # human-readable Hydra + SMCEvolve log
    └── event_logs/
        ├── _final.json                    # final best program + summary
        └── island_<i>/iter_<n>/...        # per-island per-iter prompt/response/program
```

`<category>` ∈ `math` | `algotune` | `symreg` | `autoresearch`.
**Rerunning the same problem under the same `SWEEP_TAG` overwrites the previous
output** — pick a new `SWEEP_TAG` if you want to keep both.

Visualize:

```bash
./viz.sh                # http://127.0.0.1:5173
./viz.sh --port 8080
./viz.sh --host 0.0.0.0
```

The sidebar lays out every run as a `<category>/<problem>/sweep_<TAG>` tree.

Common triage commands:

```bash
# List all failed problems in a sweep
grep -E '^\[.*\] FAIL' outputs/_sweep/<TAG>/_sweep.log

# Pull the final score for one problem
grep -hE '"type": *"final"' outputs/<cat>/<problem>/sweep_<TAG>/events.jsonl \
  | python -c 'import json,sys; [print(json.loads(l).get("best_reward")) for l in sys.stdin]'

# Pull the final score for every problem in a sweep
for f in outputs/*/sweep_<TAG>/events.jsonl; do
  echo -n "$(dirname "$f" | sed 's|outputs/||'): "
  grep -hE '"type": *"final"' "$f" \
    | python -c 'import json,sys; [print(json.loads(l).get("best_reward")) for l in sys.stdin]'
done
```

---

## 6. More commands

### 6.1 Run a single problem

```bash
./experiments/run_exp.sh single math_kissing_number
./experiments/run_exp.sh single autoresearch                  # auto-pins GPU 0
./experiments/run_exp.sh single symreg_phys_osc_PO0
```

### 6.2 Run an explicit list of problems

Grouped automatically by category; the algotune / autoresearch group is forced
to serial:

```bash
./experiments/run_exp.sh list \
    math_heilbronn_triangle \
    math_kissing_number \
    algotune_fft_convolution \
    autoresearch
# Execution order: symreg/math parallel → algotune serial → autoresearch exclusive GPU
```

### 6.3 Dry run (print only, don't execute)

```bash
DRY_RUN=1 ./experiments/run_exp.sh all | head -40
```

### 6.4 Tunable environment variables

| Variable          | Default        | Description                                                          |
|-------------------|----------------|----------------------------------------------------------------------|
| `ALGO`            | `small`        | Hydra algo preset: `small` / `medium` / `large` / `smc`              |
| `SEED`            | `42`           | Random seed                                                          |
| `GPU`             | `0`            | Legacy single-GPU entry for autoresearch; used when `AR_GPUS` is unset|
| `AR_GPUS`         | `$GPU`         | autoresearch GPU pool (comma-separated physical GPU IDs, e.g. `"0,1,2,3"`). The evaluator round-robins each evaluation to a GPU subprocess; effective concurrency equals pool size. |
| `PARALLEL`        | category-default | Override per-category default concurrency (change with care for algotune/autoresearch) |
| `LLM_CONCURRENCY` | `4`            | Per-run `llm.max_concurrency`; total in-flight = `PARALLEL × LLM_CONCURRENCY` |
| `AR_TIME_BUDGET`  | `60`           | Training seconds per autoresearch candidate; affects autoresearch only |
| `SWEEP_TAG`       | `date+time`    | Output rooted at `outputs/<cat>/<problem>/sweep_<TAG>/`              |
| `DRY_RUN`         | `0`            | `1` = print only, no execution                                       |

Combination examples:

```bash
# Sweep everything under the medium preset
ALGO=medium SWEEP_TAG=medium_$(date +%Y%m%d) ./experiments/run_exp.sh all

# autoresearch with a longer training budget
AR_TIME_BUDGET=180 ./experiments/run_exp.sh category autoresearch

# Temporarily raise symreg concurrency (idle machine, high API quota)
PARALLEL=8 ./experiments/run_exp.sh category symreg

# Run autoresearch on GPU 2
GPU=2 ./experiments/run_exp.sh category autoresearch

# Use 4 GPUs concurrently for autoresearch: each candidate evaluation still
# occupies 1 GPU, but SMC runs up to 4 evaluations in parallel (auto round-robin
# over GPUs 0/1/2/3)
AR_GPUS=0,1,2,3 ./experiments/run_exp.sh category autoresearch

# Same idea, but only GPUs 1 and 3 (max 2-way concurrency)
AR_GPUS=1,3 ./experiments/run_exp.sh category autoresearch
```

> **Where algorithm-level concurrency comes from**: for autoresearch,
> `algo.particles_per_island` × `algo.n_islands` sets the number of
> evaluations that can run simultaneously within one SMC step. The GPU pool is
> only an upper bound — if the pool has 4 GPUs but `particles_per_island=1`
> and `n_islands=1`, only 1 evaluation actually runs at any moment. To
> saturate 4 GPUs:
> ```bash
> AR_GPUS=0,1,2,3 ALGO=medium ./experiments/run_exp.sh category autoresearch
> # Or set Hydra fields directly: algo.particles_per_island=4
> ```

### 6.5 Run via Hydra directly (bypass `run_exp.sh`)

`run_exp.sh` is a wrapper around Hydra; the following is equivalent to
`./experiments/run_exp.sh single math_kissing_number`:

```bash
CUDA_VISIBLE_DEVICES= python -m smcevolve.main \
    problem=math_kissing_number \
    algo=small \
    seed=42 \
    llm.max_concurrency=4 \
    hydra.run.dir=outputs/math/math_kissing_number/sweep_manual
```

Any Hydra field can be overridden from the CLI:

```bash
python -m smcevolve.main problem=circle_packing algo=medium \
    algo.max_iterations=30 algo.n_islands=4 seed=7
```

### 6.6 Rough time / cost estimate (`algo=small`, 6 iter × 1 island × 4 particles × 1 proposal = 24 LLM calls / run)

| Category          | Per-run time | Serial total | Default-parallel total      |
|-------------------|--------------|--------------|-----------------------------|
| symreg (129)      | 3–7 min      | 6–15 h       | 1.5–4 h (PARALLEL=4)        |
| math (10)         | 5–20 min     | 50–200 min   | 20–70 min (PARALLEL=3)      |
| algotune (8)      | 5–15 min     | 40–120 min   | 40–120 min (forced serial)  |
| autoresearch (1)  | ~25 min      | ~25 min      | ~25 min                     |
| **Total**         |              | **~10–20 h** | **~4–8 h**                  |

`ALGO=medium` is roughly 3–5× the cost of `small`. The LLM bill scales
proportionally — cost is written to
`outputs/<cat>/<problem>/sweep_<TAG>/events.jsonl` under
`proposal.proposal_metadata.cost_usd` and
`proposal.proposal_metadata.cumulative_cost_usd`.

---

## 7. Interruption and resume

The script itself does not implement resume. If a sweep dies mid-run:

1. Inspect `outputs/_sweep/<TAG>/_sweep.log` and find which problems did not
   reach `ok`;
2. Use `list` to rerun the ones that failed (**reusing the same `SWEEP_TAG`
   overwrites the previous output**):

```bash
SWEEP_TAG=<original TAG> ./experiments/run_exp.sh list \
    math_erdos_min_overlap \
    symreg_chem_react_CR17
```

---

## 8. One-liner summary

```bash
# Overnight full sweep (small preset, ~4–8 h)
SWEEP_TAG=run1 ./experiments/run_exp.sh all

# Watch progress live
tail -f outputs/_sweep/run1/_sweep.log

# Inspect results
./viz.sh           # open http://127.0.0.1:5173 in a browser
```
