# SMCEvolve Ablation Study — Circle Packing

## Baseline

A3 = B2 = C1, run once.

```
β=20, κ=0.9, I=2, P=8, K=2, adaptive (all 4 kernels), max_iter=30, seed=42
```

---

## Group A — Effect of β, κ on iteration count

**Fixed**: I=2, P=8, K=2, adaptive, max_iter=30

| ID | β | κ | Design intent |
|----|---|---|---------------|
| **A1** | 5 | 0.9 | Low β → small Δβ → few iterations, weak selection pressure |
| **A2** | 40 | 0.9 | High β → large Δβ → ESS drops sharply → many iterations, strong selection pressure |
| **A3** | 20 | 0.5 | Aggressive κ → low ESS threshold → big jumps, rapid loss of diversity |
| baseline | 20 | 0.9 | Default |

**Comparisons**: A1 vs baseline vs A2 → effect of β on iteration count; A3 vs baseline → aggressive vs conservative κ

---

## Group B — Population architecture (I×P×K = 32)

**Fixed**: β=20, κ=0.9, adaptive, max_iter=30

| ID | I | P | K | Design intent |
|----|---|---|---|---------------|
| **B1** | 1 | 16 | 2 | Single island, large population, no migration; trade particle count for diversity |
| **B2** | 4 | 4 | 2 | Many islands, small populations; rely on migration for diversity |
| **B3** | 1 | 8 | 4 | High best-of-K; pick the best proposal per particle |
| baseline | 2 | 8 | 2 | Default |

**Comparisons**: B1 vs B2 → particle count vs island count; B3 vs baseline → benefit of higher K

---

## Group C — Kernel / Context

**Fixed**: β=20, κ=0.9, I=2, P=8, K=2, max_iter=30

| ID | Strategy | Design intent |
|----|----------|---------------|
| **C1** | force `diff_with_inspo` | **Core question**: is diff + inspiration enough on its own? |
| **C2** | force `diff_no_inspo` | Compared to C1, isolates the value of inspiration |
| baseline | adaptive across all 4 kernels | Adaptive mixture |

**Comparisons**: C1 vs C2 → does inspiration help; baseline vs C1 → mixture strategy vs single kernel

---

## Total

| Group | Experiments | Independent runs |
|-------|-------------|------------------|
| A | baseline + A1, A2, A3 | 3 |
| B | baseline + B1, B2, B3 | 3 |
| C | baseline + C1, C2 | 2 |
| **Total** | **10** | **8 + 1 baseline = 9** |

---

## Running

```bash
./experiments/run_ablation.sh A1          # single experiment
./experiments/run_ablation.sh groupA      # one group
./experiments/run_ablation.sh all         # all 9 runs
SEED=123 ./experiments/run_ablation.sh A1 # change seed
```

Output: `outputs/circle_packing/ablation/{ID}_{timestamp}/`
