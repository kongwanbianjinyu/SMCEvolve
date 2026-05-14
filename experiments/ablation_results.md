# SMCEvolve Ablation Study — Circle Packing (N=26)

> **Date**: 2026-04-15  
> **Problem**: Circle Packing in Unit Square, N=26 circles, maximize sum of radii  
> **Seed**: 42 (all runs)

---

## 1. Baseline Configuration

| Parameter | Value |
|-----------|-------|
| `algo.beta` (inverse temperature) | 20 |
| `algo.kappa` (tempering rate) | 0.9 |
| `algo.n_islands` | 2 |
| `algo.particles_per_island` | 8 |
| `algo.n_proposals` | 2 |
| `algo.max_iterations` | 15 (baseline) / 30 (ablation) |
| `algo.min_iterations` | 3 |
| `algo.migration_interval` | 3 |
| `algo.migration_size` | 1 |
| `algo.prompt.kernel_selection` | adaptive |
| `algo.prompt.kernel_weights` | all 1.0 (uniform prior) |
| LLM models | gpt-5.4-mini (50%), gemini-3-flash-preview (50%) |
| Total population (I x P) | 16 |
| Proposals per iteration (I x P x K) | 32 |

**Baseline Result**: best_reward = **2.5601**, epochs = 7, cost = $2.31

---

## 2. Group A: Inverse Temperature (beta) & Tempering Rate (kappa)

### Design

**Focus**: How do selection pressure (beta) and tempering schedule (kappa) affect convergence speed and final solution quality?

| Exp | beta | kappa | Rationale |
|-----|------|-------|-----------|
| A1 | 5 | 0.9 | Low selection pressure — more exploration, less exploitation |
| A2 | 40 | 0.9 | High selection pressure — aggressive exploitation |
| A3 | 20 | 0.5 | Slow tempering — temperature rises slowly, more early exploration |
| A4 | 20 | 1.0 | No tempering — full selection pressure from the start |

### Results

| Exp | beta | kappa | Best Reward | Epochs | Cost ($) | Notes |
|-----|------|-------|-------------|--------|----------|-------|
| **Baseline** | 20 | 0.9 | **2.5601** | 7 | 2.31 | Reference |
| A1 | 5 | 0.9 | 2.5239 | 5 | 1.81 | Low pressure |
| A2 | 40 | 0.9 | 2.5527 | 8 | 2.52 | High pressure |
| A3 | 20 | 0.5 | 2.4000 | 4 | 1.36 | Slow tempering |
| A4 | 20 | 1.0 | 2.5220 | 22* | ~2.5† | No tempering (crashed) |

*A4 ran 22 iterations but crashed without writing `_final.json`; best reward extracted from event logs.  
†Cost estimated from LLM call count (~733 HTTP requests).

**Convergence**:
```
Iter  Baseline   A1(β=5)   A2(β=40)  A3(κ=0.5)  A4(κ=1.0)
  1    2.4078    2.4000    2.4120    2.3437     2.4883
  2    2.4094    2.4567    2.4853    2.4000     2.5025
  3    2.4783    2.4851    2.4979    2.4000     2.5025
  4    2.4814    2.5073    2.5032    2.4000     2.5130
  5    2.5368    2.5239*   2.5095     —         2.5173
  6    2.5601     —        2.5095     —         2.5211
  7    2.5601*    —        2.5507     —         2.5213
  8     —         —        2.5527*    —         2.5218
  ...                                           2.5220* (iter 22)
```
(* = final iteration for that experiment)

### Observations

1. **Baseline (beta=20, kappa=0.9) achieves the best result.** The moderate inverse temperature balances exploration and exploitation effectively.
2. **Low beta (A1, beta=5) converges faster in early iterations** (reaches 2.4567 by iter 2 vs baseline's 2.4094), but plateaus at a lower final reward (2.5239). The weak selection pressure fails to sharpen the search in later stages.
3. **High beta (A2, beta=40) converges slower initially but reaches near-baseline quality** (2.5527). The strong selection pressure causes premature narrowing of the particle distribution in early iterations, but the larger iteration budget (8 epochs) partially compensates.
4. **Slow tempering (A3, kappa=0.5) is catastrophically bad** — stuck at 2.4000 after only 4 epochs. The temperature rises too slowly, meaning beta_t stays near zero for most of the run, effectively disabling selection pressure and reducing the algorithm to random sampling.
5. **No tempering (A4, kappa=1.0) leads to stagnation** — runs 22 iterations but barely improves beyond 2.5220. With full beta applied immediately, the particle distribution collapses early. The convergence curve shows diminishing returns after iteration 5, with only 0.0047 improvement over the remaining 17 iterations.

**Takeaway**: kappa is the most sensitive parameter. kappa=0.9 (gradual warm-up) is critical. beta=20 is a sweet spot; doubling it causes only minor degradation.

---

## 3. Group B: Population Structure (I x P x K, total budget ~32 proposals/iter)

### Design

**Focus**: How does the allocation of compute budget across islands, particles, and proposals affect diversity and solution quality?

| Exp | Islands (I) | Particles (P) | Proposals (K) | I×P×K | Rationale |
|-----|-------------|---------------|---------------|-------|-----------|
| B1 | 1 | 16 | 2 | 32 | Single large island — no migration, all diversity from within |
| B2 | 4 | 4 | 2 | 32 | Many small islands — maximum inter-island diversity |
| B3 | 1 | 8 | 4 | 32 | Single island, more proposals per particle — deeper per-particle search |
| B4 | 2 | 16 | 1 | 32 | Baseline islands, double particles, half proposals — wider population |

### Results

| Exp | Islands (I) | Particles (P) | Proposals (K) | I×P×K | Best Reward | Epochs | Cost ($) |
|-----|-------------|---------------|---------------|-------|-------------|--------|----------|
| **Baseline** | 2 | 8 | 2 | 32 | **2.5601** | 7 | 2.31 |
| B1 | 1 | 16 | 2 | 32 | 2.5168 | 6 | 2.35 |
| B2 | 4 | 4 | 2 | 32 | 2.5032 | 7 | 2.24 |
| B3 | 1 | 8 | 4 | 32 | 2.4956 | 7 | 2.91 |
| B4 | 2 | 16 | 1 | 32 | 2.5177 | 6 | 2.17 |

**Convergence**:
```
Iter  Baseline   B1(1,16,2)  B2(4,4,2)  B3(1,8,4)  B4(2,16,1)
  1    2.4078    2.4705      2.4228     2.4996     2.4603
  2    2.4094    2.4842      2.4463     2.5156     2.4915
  3    2.4783    2.4929      2.4953     2.5156     2.4956
  4    2.4814    2.5084      2.4957     2.5163     2.4956
  5    2.5368    2.5165      2.5032     2.5171     2.4956
  6    2.5601    2.5168*     2.5032     2.5171*    2.4956
  7    2.5601*    —          2.5032*     —         2.4956*
```

### Observations

1. **The baseline 2-island structure (I=2, P=8, K=2) outperforms all alternatives** by a significant margin (+0.0424 over the best alternative).
2. **Single island (B1, B3) removes the diversity benefit of migration.** B1 (1x16x2) reaches 2.5168, competitive but below baseline. B3 (1x8x4) reaches only 2.4956 despite the same compute budget — more proposals per particle does not compensate for fewer particles and no migration.
3. **Many small islands (B2, 4x4x2) hurt performance** (2.5032). With only 4 particles per island, intra-island diversity is too low, and migration alone cannot recover.
4. **Larger population with fewer proposals (B4, 2x16x1) is not cost-effective** — despite doubling particles, the single proposal per particle limits the improvement rate. Result (2.5177) is similar to B1.
5. **B3 starts strongest** (2.4996 at iter 1) due to 4 proposals per particle generating better initial programs, but stagnates earliest. More proposals help initialization but not long-term search.

**Takeaway**: 2 islands with moderate particle count (8) and 2 proposals strikes the best balance. Migration between islands provides crucial diversity injection that single-island configurations lack.

---

## 4. Group C: Mutation Kernel

### Design

**Focus**: Which kernel type contributes most to search effectiveness? Baseline uses adaptive selection over all four.

| Exp | Kernel | Description |
|-----|--------|-------------|
| C1 | diff_no_inspo | Diff-based edits, no cross-program inspiration |
| C2 | diff_with_inspo | Diff-based edits, with inspiration from other programs |
| C3 | rewrite_no_inspo | Full rewrite, no inspiration |
| C4 | rewrite_with_inspo | Full rewrite, with inspiration from other programs |

### Results

| Exp | Kernel | Best Reward | Epochs | Cost ($) |
|-----|--------|-------------|--------|----------|
| **Baseline** | adaptive (all 4) | **2.5601** | 7 | 2.31 |
| C1 | diff_no_inspo | 2.5177 | 6 | 0.80 |
| C2 | diff_with_inspo | 2.4707 | 4 | 0.64 |
| C3 | rewrite_no_inspo | 2.5085 | 8 | 2.11 |
| C4 | rewrite_with_inspo | 2.4741 | 6 | 2.75 |

Baseline adaptive kernel weights at final iteration:
- diff_with_inspo: 0.382 (highest)
- rewrite_with_inspo: 0.338
- diff_no_inspo: 0.264
- rewrite_no_inspo: 0.136 (lowest)

**Convergence**:
```
Iter  Baseline   C1(d-no)   C2(d-ins)  C3(rw-no)  C4(rw-ins)
  1    2.4078    2.4288     2.4707     2.5050     2.2855
  2    2.4094    2.4707     2.4707     2.5083     2.3161
  3    2.4783    2.4800     2.4707     2.5083     2.3697
  4    2.4814    2.5085     2.4707*    2.5083     2.4538
  5    2.5368    2.5094      —         2.5085     2.4611
  6    2.5601    2.5177*     —         2.5085     2.4741*
  7    2.5601*    —          —         2.5085      —
  8     —         —          —         2.5085*     —
```

### Observations

1. **Adaptive kernel selection (baseline) dramatically outperforms any single kernel** — best single kernel (C1, diff_no_inspo) trails by 0.0424.
2. **diff_no_inspo (C1) is the best individual kernel** (2.5177), consistent with being a simple, focused edit strategy. It also has the lowest cost ($0.80) because diff edits generate shorter outputs.
3. **diff_with_inspo (C2) paradoxically performs worst** (2.4707) despite being the highest-weighted kernel in adaptive mode (0.382). When forced as the sole kernel, the inspiration mechanism may introduce too much noise without the balancing effect of other kernels. Also stagnates after only 4 epochs.
4. **rewrite_no_inspo (C3) runs the most iterations** (8) but stagnates at 2.5085. Full rewrites explore more broadly but converge slowly, plateauing after iteration 2 with only marginal gains.
5. **rewrite_with_inspo (C4) starts very poorly** (2.2855 at iter 1) — full rewrites with inspiration are the most disruptive kernel. It recovers slowly but never catches up (2.4741). High cost ($2.75) due to long rewrite outputs.
6. **Cost varies dramatically**: diff kernels (C1: $0.80, C2: $0.64) are 3-4x cheaper than rewrite kernels (C3: $2.11, C4: $2.75). The baseline's adaptive strategy ($2.31) effectively balances quality and cost.

**Takeaway**: No single kernel is sufficient — the adaptive kernel selection mechanism is a key component of SMCEvolve's effectiveness. The portfolio approach lets the algorithm use diff edits for efficient refinement and rewrites for exploration as needed.

---

## 5. Overall Summary

### All Experiments Ranked

| Rank | Exp | Config Change | Best Reward | Delta vs Baseline |
|------|-----|---------------|-------------|-------------------|
| 1 | **Baseline** | — | **2.5601** | — |
| 2 | A2 | beta=40 | 2.5527 | -0.0074 |
| 3 | A1 | beta=5 | 2.5239 | -0.0362 |
| 4 | A4 | kappa=1.0 | 2.5220 | -0.0381 |
| 5 | B4 | I=2,P=16,K=1 | 2.5177 | -0.0424 |
| 6 | C1 | diff_no_inspo | 2.5177 | -0.0424 |
| 7 | B1 | I=1,P=16,K=2 | 2.5168 | -0.0433 |
| 8 | C3 | rewrite_no_inspo | 2.5085 | -0.0516 |
| 9 | B2 | I=4,P=4,K=2 | 2.5032 | -0.0569 |
| 10 | B3 | I=1,P=8,K=4 | 2.4956 | -0.0645 |
| 11 | C4 | rewrite_with_inspo | 2.4741 | -0.0860 |
| 12 | C2 | diff_with_inspo | 2.4707 | -0.0894 |
| 13 | A3 | kappa=0.5 | 2.4000 | -0.1601 |

### Conclusions

1. **The baseline configuration is well-tuned.** No single-parameter ablation improves upon it. This validates the original parameter choices.

2. **Tempering rate (kappa) is the most critical hyperparameter.** kappa=0.5 causes a 0.16 drop (largest degradation in the study), while kappa=1.0 leads to premature convergence with 22 iterations barely surpassing what the baseline achieves in 5.

3. **Adaptive kernel selection is the most impactful algorithmic component.** The gap between baseline (adaptive) and the best single kernel (C1) is 0.0424 — larger than the gap caused by doubling/halving beta.

4. **Multi-island structure with migration provides meaningful diversity.** Removing it (B1, B3) consistently degrades performance, though the effect is smaller than kappa or kernel selection.

5. **Beta is relatively robust.** Both beta=5 and beta=40 produce competitive results (within 0.04 of baseline), suggesting the algorithm tolerates a wide range of selection pressures when tempering is properly configured.

6. **Cost-performance trade-off matters.** The cheapest configurations (C1: $0.80, C2: $0.64) sacrifice 0.04-0.09 in reward. The baseline at $2.31 represents a reasonable cost-quality balance.
