# SMCEvolve Ablation Study — Circle Packing

## Baseline

A3 = B2 = C1，只跑一次。

```
β=20, κ=0.9, I=2, P=8, K=2, adaptive (all 4 kernels), max_iter=30, seed=42
```

---

## Group A — β, κ 对迭代次数的影响

**固定**: I=2, P=8, K=2, adaptive, max_iter=30

| ID | β | κ | 设计意图 |
|----|---|---|---------|
| **A1** | 5 | 0.9 | 低 β → Δβ 小 → 少迭代、弱选择压力 |
| **A2** | 40 | 0.9 | 高 β → Δβ 大 → ESS 骤降 → 多迭代、强选择压力 |
| **A3** | 20 | 0.5 | 激进 κ → ESS 阈值低 → 大步快进、多样性快速丧失 |
| baseline | 20 | 0.9 | 默认 |

**对比**: A1 vs baseline vs A2 → β 对迭代数的影响；A3 vs baseline → κ 激进 vs 保守

---

## Group B — Population 架构 (I×P×K = 32)

**固定**: β=20, κ=0.9, adaptive, max_iter=30

| ID | I | P | K | 设计意图 |
|----|---|---|---|---------|
| **B1** | 1 | 16 | 2 | 单岛大种群，无 migration，靠粒子数换多样性 |
| **B2** | 4 | 4 | 2 | 多岛小种群，靠 migration 换多样性 |
| **B3** | 1 | 8 | 4 | 高 best-of-K，每个粒子选最优 proposal |
| baseline | 2 | 8 | 2 | 默认 |

**对比**: B1 vs B2 → 粒子数 vs 岛数；B3 vs baseline → 高 K 值的收益

---

## Group C — Kernel / Context

**固定**: β=20, κ=0.9, I=2, P=8, K=2, max_iter=30

| ID | 策略 | 设计意图 |
|----|------|---------|
| **C1** | force `diff_with_inspo` | **核心问题**: 只用 diff + inspiration 够不够？ |
| **C2** | force `diff_no_inspo` | 对比 C1，隔离 inspiration 的价值 |
| baseline | adaptive 全部 4 kernel | 自适应混合 |

**对比**: C1 vs C2 → inspiration 有没有用；baseline vs C1 → 混合策略 vs 单一 kernel

---

## 总计

| Group | 实验 | 独立运行 |
|-------|------|---------|
| A | baseline + A1, A2, A3 | 3 |
| B | baseline + B1, B2, B3 | 3 |
| C | baseline + C1, C2 | 2 |
| **Total** | **10** | **8 + 1 baseline = 9** |

---

## 运行

```bash
./run_ablation.sh A1          # 单个
./run_ablation.sh groupA      # 一组
./run_ablation.sh all         # 全部 9 runs
SEED=123 ./run_ablation.sh A1 # 换 seed
```

输出: `outputs/circle_packing/ablation/{ID}_{timestamp}/`
