# AlgoTune — Runtime optimization tasks

Full port of the OpenEvolve AlgoTune benchmark harness
([`examples/algotune/`](../../examples/algotune/), see that
folder's [`README.md`](../../examples/algotune/README.md) for
background and results).

AlgoTune tasks are materialized by importing task definitions from
an external clone of the [AlgoTune](https://github.com/oripress/AlgoTune)
repository.

This repo ships with a curated subset of 8 tasks covering
numerical / signal-processing / linear-algebra workloads:

| Task | Area |
| --- | --- |
| [`affine_transform_2d`](affine_transform_2d/) | 2D image warping (scipy.ndimage) |
| [`convolve2d_full_fill`](convolve2d_full_fill/) | Direct 2D convolution |
| [`eigenvectors_complex`](eigenvectors_complex/) | Complex eigendecomposition |
| [`fft_cmplx_scipy_fftpack`](fft_cmplx_scipy_fftpack/) | Complex FFT |
| [`fft_convolution`](fft_convolution/) | FFT-based convolution |
| [`lu_factorization`](lu_factorization/) | Dense LU (LAPACK) |
| [`polynomial_real`](polynomial_real/) | Real polynomial root finding |
| [`psd_cone_projection`](psd_cone_projection/) | Projection onto the PSD cone |

Each task folder contains:
- `initial_program.py` — AlgoTune's reference `solve` extracted as
  a seed (verbatim, via `task_adapter.AlgoTuneTaskAdapter`).
- `openevolve_evaluator.py` — the upstream OpenEvolve evaluator
  (verbatim; measures wall-clock speedup vs. the AlgoTune baseline,
  returns `speedup_score` + `correctness_score`).
- `evaluator.py` — SMCEvolve wrapper on top of
  [`smcevolve/openevolve_bridge.py`](../../smcevolve/openevolve_bridge.py),
  keying on `speedup_score`.
- `task.md` — LLM prompt (drawn from AlgoTune's `description.txt`).
- `config.yaml` — the upstream OpenEvolve config (kept for
  reference; SMCEvolve runs from a separate Hydra config).

## One-time setup

```bash
# 1) Clone AlgoTune next to SMCEvolve (or anywhere you like).
git clone https://github.com/oripress/AlgoTune ../AlgoTune

# 2) Install AlgoTune's Python requirements. The list is large
#    (jax, cvxpy, pulp, pot, numba, scikit-learn, ...); see the
#    requirements.txt in examples/algotune or AlgoTune's own docs.
uv pip install -r problems/algotune/requirements.txt
```

The 8 task folders above are already materialized in this repo, so
step 1/2 are only needed if you want to regenerate them or add
more tasks.

## Emit Hydra configs

```bash
python problems/algotune/generate_smc_configs.py
```

This writes `configs/problem/algotune_<task>.yaml` for each of the
8 task folders. Re-run after regenerating or adding any task.

## Run

```bash
python -m smcevolve.main problem=algotune_psd_cone_projection algo=medium
python -m smcevolve.main problem=algotune_lu_factorization    algo=medium
python -m smcevolve.main problem=algotune_fft_convolution     algo=medium
```

Use `ls configs/problem/algotune_*.yaml` to list the full set of
Hydra config names.

The generated `openevolve_evaluator.py` imports from `AlgoTuneTasks`
at evaluation time; make sure your AlgoTune checkout stays in place
or update the fallback paths in
[`task_adapter.py`](task_adapter.py) → `setup_algotune_paths`.

## Regenerating / adding tasks

To refresh one of the 8 task folders from your AlgoTune checkout:

```bash
python problems/algotune/create_task.py \
    --algotune-path ../AlgoTune \
    --task psd_cone_projection
```

To add a new task, run the same command with a different
`--task <name>` (see `AlgoTuneTasks/<name>/` in your AlgoTune
checkout for available names), then rerun
`generate_smc_configs.py`.

`generate_all_tasks.py` materializes **every** AlgoTune task; it is
left in the repo as a convenience but is not needed for the curated
8-task setup.

## Notes

- Speedup is computed against AlgoTune's reference `solve` via the
  timing harness in `openevolve_evaluator.py`; see the upstream
  examples for how cascade evaluation, `num_runs`, and `warmup_runs`
  are tuned per task type.
- For JIT-heavy tasks (JAX, Numba) you'll likely want to raise the
  per-candidate timeout in the Hydra config, e.g.
  `problem.timeout=900`.
