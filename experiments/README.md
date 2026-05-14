# 🧪 Experiments

All large-scale experiments and ablation studies for SMCEvolve live here.
The top-level [`run.sh`](../run.sh) is for a single quick run; everything in
this folder is for sweeps and ablations. Run every command **from the repo
root** so relative paths (`configs/`, `outputs/`, `problems/`) resolve.

## 📁 Layout

| File                                            | Purpose                                                  |
|-------------------------------------------------|----------------------------------------------------------|
| 🚀 [`run_exp.sh`](run_exp.sh)                   | Sweep dispatcher for all 4 problem categories (math, algotune, symreg, autoresearch) |
| 📖 [`run_exp.md`](run_exp.md)                   | Full reproduction guide for sweeps                       |
| 🔬 [`run_ablation.sh`](run_ablation.sh)         | Circle-packing ablation runner (β/κ, population, kernel) |
| 📋 [`ablation_plan.md`](ablation_plan.md)       | Ablation design and group descriptions                   |
| 📊 [`ablation_results.md`](ablation_results.md) | Recorded ablation results                                |

## ⚡ Quick examples

```bash
# overnight sweep across all 4 categories (small preset, ~4–8 h)
SWEEP_TAG=run1 ./experiments/run_exp.sh all

# one category
./experiments/run_exp.sh category math

# one problem
./experiments/run_exp.sh single math_kissing_number

# one ablation experiment
./experiments/run_ablation.sh A1

# all 9 ablation runs
./experiments/run_ablation.sh all
```

See [`run_exp.md`](run_exp.md) for the full guide (env-var overrides, output
layout, troubleshooting) and [`ablation_plan.md`](ablation_plan.md) for the
ablation experiment design.
