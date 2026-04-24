# Symbolic Regression — LLM-SRBench

Full port of the OpenEvolve symbolic-regression benchmark
([`examples/symbolic_regression/`](../../examples/symbolic_regression/),
which wraps [LLM-SRBench](https://arxiv.org/pdf/2504.10415)).

Unlike the math category, individual task files are **generated on
demand** from the LLM-SRBench dataset on HuggingFace —
`bio_pop_growth` (24 tasks), `chem_react` (36), `matsci` (25),
`phys_osc` (44). We don't commit the 129 generated task folders;
you materialize them once via `data_api.py`.

Each generated task folder contains:
- `initial_program.py` — a linear-model seed (verbatim from upstream).
- `openevolve_evaluator.py` — the upstream OpenEvolve evaluator
  (verbatim; BFGS-fits `params`, returns `combined_score = -log10(MSE)`).
- `evaluator.py` — SMCEvolve wrapper on top of
  [`smcevolve/openevolve_bridge.py`](../../smcevolve/openevolve_bridge.py).
- `task.md` — concise LLM prompt.
- `X_train_for_eval.npy`, `y_train_for_eval.npy`, and test / OOD
  counterparts — LLM-SRBench data.
- `config.yaml` — the upstream OpenEvolve config (kept for
  reference; SMCEvolve runs from a separate Hydra config).

## Install extra dependencies

```bash
uv pip install datasets huggingface_hub h5py sympy scipy scikit-learn pyyaml
```

(The OpenEvolve scaffolding uses `datasets` + `h5py` to pull
LLM-SRBench from the `nnheui/llm-srbench` HuggingFace repo.)

## Generate tasks

From the repo root:

```bash
# One-time dataset materialization (downloads from HF on first run).
python problems/symbolic_regression/data_api.py
```

This populates `problems/symbolic_regression/generated/<split>/<equation_idx>/`
with ~129 task folders. Then emit Hydra configs for each:

```bash
python problems/symbolic_regression/generate_smc_configs.py
```

This writes `configs/problem/symreg_<split>_<equation_idx>.yaml`.

To limit which splits are materialized, edit the `splits_data`
dict near the bottom of `data_api.py` before running.

## Run

```bash
# pick any materialized task
python -m smcevolve.main problem=symreg_phys_osc_PO0 algo=medium
python -m smcevolve.main problem=symreg_bio_pop_growth_BPG0 algo=medium
```

Use `ls configs/problem/symreg_*.yaml` to list the full set of
Hydra config names.

## Evaluation

Identical to OpenEvolve's: the evaluator BFGS-fits the 10
`params` on `X_train_for_eval.npy` (single restart per call) and
reports `combined_score = max(0, -log10(train_MSE + 1e-9))`.

To assess final test / OOD performance after an SMCEvolve run,
use the upstream `eval.py`:

```bash
python problems/symbolic_regression/eval.py problems/symbolic_regression/generated/phys_osc/PO0
```

Note: `eval.py` expects OpenEvolve-style output at
`<task>/openevolve_output/best/best_program.py`; adapt it if
pointing at SMCEvolve outputs under `outputs/<problem>/...`.
