# Math — AlphaEvolve problems

Ported subset of the AlphaEvolve math problems (10 tasks) from
[`examples/alphaevolve_math_problems/`](../../examples/alphaevolve_math_problems/),
which in turn adapts
[`google-deepmind/alphaevolve_results`](https://github.com/google-deepmind/alphaevolve_results)
(Apache 2.0).

**What was removed from the full upstream set**
- `matmul`, `sums_diffs_finite_sets`, `uncertainty_ineq` — JAX seed
  evaluation too slow for SMC-scale search (≥ 60 s per candidate).
- Parameter-only variants of the same mathematical problem — we keep
  one representative per problem:
  - hexagon packing: kept `n=11`, dropped `n=12`
  - Heilbronn convex: kept `n=13`, dropped `n=14`
  - minimizing max/min pairwise distance: kept `n=16, d=2`, dropped
    `n=14, d=3`

The three autocorrelation inequalities (B.1 / B.2 / B.3) are kept
separately — they target different constants (`C1`, `C2`, `C3`) and
are not parameter-only variants.

See `examples/alphaevolve_math_problems/` if you want to try the
dropped variants.

## Problem layout

Each folder contains:
- `initial_program.py` — the upstream OpenEvolve seed (verbatim).
- `openevolve_evaluator.py` — the upstream OpenEvolve evaluator
  (verbatim; `evaluate(program_path) -> dict`).
- `evaluator.py` — SMCEvolve wrapper that calls
  `openevolve_evaluator.evaluate` via
  [`smcevolve/openevolve_bridge.py`](../../smcevolve/openevolve_bridge.py)
  and exposes `evaluate(program: str) -> float` returning
  `combined_score` (or `0.0` on failure).
- `task.md` — concise LLM prompt for the SMCEvolve proposer.

Evaluation semantics are 100% identical to OpenEvolve's — the
wrapper only adapts the I/O shape. A reward ≥ 1.0 matches or beats
the AlphaEvolve paper benchmark.

## Included problems

| Problem                                                    | Dir                                | Hydra `problem=`                         |
|------------------------------------------------------------|------------------------------------|------------------------------------------|
| 1st autocorrelation inequality (B.1)                       | `first_autocorr_ineq/`             | `math_first_autocorr_ineq`               |
| 2nd autocorrelation inequality (B.2)                       | `second_autocorr_ineq/`            | `math_second_autocorr_ineq`              |
| 3rd autocorrelation inequality (B.3)                       | `third_autocorr_ineq/`             | `math_third_autocorr_ineq`               |
| Erdős minimum overlap (B.5)                                | `erdos_min_overlap/`               | `math_erdos_min_overlap`                 |
| Hexagon packing (B.7, `n=11`)                              | `hexagon_packing_11/`              | `math_hexagon_packing_11`                |
| Min max/min pairwise distance (B.8, `n=16, d=2`)           | `minimizing_max_min_dist_dim2_16/` | `math_minimizing_max_min_dist_dim2_16`   |
| Heilbronn triangle (B.9, `n=11`)                           | `heilbronn_triangle/`              | `math_heilbronn_triangle`                |
| Heilbronn convex (B.10, `n=13`)                            | `heilbronn_convex_13/`             | `math_heilbronn_convex_13`               |
| Kissing number in dim 11 (B.11)                            | `kissing_number/`                  | `math_kissing_number`                    |
| Circle packing in a perimeter-4 rectangle (B.13, `n=21`)   | `circle_packing_rect/`             | `math_circle_packing_rect`               |

(The `circle_packing` / n=26-in-unit-square problem from B.12 is
already the repo's existing [`problems/circle_packing`](../circle_packing/)
— no duplicate port here.)

## Install extra dependencies

The three autocorrelation inequalities and the Erdős problem use
JAX + Optax + sympy in the seed program. Install once:

```bash
uv pip install jax optax sympy
# (tqdm is already in the base venv)
```

Problems that need only NumPy (packings, kissing number, Heilbronn,
min-max-dist) work without additional installs.

## Run

With the venv active (`uv sync && source .venv/bin/activate`):

```bash
# NumPy-only problems (fast evaluation)
python -m smcevolve.main problem=math_heilbronn_triangle            algo=medium
python -m smcevolve.main problem=math_circle_packing_rect           algo=medium
python -m smcevolve.main problem=math_kissing_number                algo=medium
python -m smcevolve.main problem=math_hexagon_packing_11            algo=medium
python -m smcevolve.main problem=math_heilbronn_convex_13           algo=medium
python -m smcevolve.main problem=math_minimizing_max_min_dist_dim2_16 algo=medium

# JAX-based problems (seed evaluation ~3-10 s)
python -m smcevolve.main problem=math_first_autocorr_ineq   algo=medium
python -m smcevolve.main problem=math_second_autocorr_ineq  algo=medium
python -m smcevolve.main problem=math_third_autocorr_ineq   algo=medium
python -m smcevolve.main problem=math_erdos_min_overlap     algo=medium
```

Override any Hydra field inline:

```bash
python -m smcevolve.main problem=math_heilbronn_triangle algo=large \
    algo.max_iterations=50 seed=7
```

Per-candidate evaluation timeout is 600 s by default; adjust with
`problem.timeout=...` if needed.

## Smoke test

Run every evaluator against its seed program in one go:

```bash
source .venv/bin/activate
python problems/math/smoke_test.py                 # default 900 s / problem
python problems/math/smoke_test.py --timeout 180   # faster pass
```

Results are logged to `problems/math/smoke_test.log`.

## Adapter

See [`smcevolve/openevolve_bridge.py`](../../smcevolve/openevolve_bridge.py)
for the one-file bridge that turns any OpenEvolve evaluator into a
SMCEvolve evaluator. Each problem's `evaluator.py` is ~20 lines of
glue that imports `openevolve_evaluator.evaluate` and wraps it.
