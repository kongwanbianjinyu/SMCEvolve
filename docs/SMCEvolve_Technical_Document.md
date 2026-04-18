# SMCEvolve 核心算法技术文档

---

## 目录

1. [核心算法总览](#1-核心算法总览)
2. [Parent 选择与 Weight 计算](#2-parent-选择与-weight-计算)
3. [Context 选择与 Kernel 自适应挑选](#3-context-选择与-kernel-自适应挑选)
4. [Inspiration Program 选择机制](#4-inspiration-program-选择机制)
5. [SMC Process Control](#5-smc-process-control)
6. [LLM 集成方式](#6-llm-集成方式)

---

## 1. 核心算法总览

SMCEvolve 将 **Sequential Monte Carlo (SMC)** 框架与 **LLM 驱动的程序进化** 结合，通过粒子群在程序空间上的迭代采样来搜索高质量的程序解。

### 1.1 整体架构

```
IslandController (K 个独立链 + 迁移)
  ├── SMCIsland_0 (N 个粒子)
  │     ├── 自适应温度调度 (ESS bisection)
  │     ├── Softmax 重采样
  │     └── Best-of-K 变异 (LLM proposal + evaluation)
  ├── SMCIsland_1
  │     └── ...
  └── 周期性迁移 (merge-and-truncate)
```

### 1.2 单步 SMC 迭代 (Algorithm 1)

每个 `SMCIsland.step()` 执行以下三个阶段（`island.py:116-192`）：

```
输入: 粒子集合 {x_i, r_i}_{i=1}^N, 当前温度参数 λ_{t-1}, 目标逆温度 β
输出: 更新后的粒子集合, λ_t

1. 自适应温度调度:
   λ_t = find_next_lambda(rewards, λ_{t-1}, β, κ)   // 求解 ESS(λ) = κN
   Δβ = (λ_t - λ_{t-1}) × β                         // 增量逆温度

2. Softmax 重采样:
   w_i = exp(Δβ × r_i) / Σ_j exp(Δβ × r_j)         // 归一化权重
   {x'_i} ~ Systematic_Resample(w)                    // 系统重采样

3. Best-of-K 变异 (对每个重采样粒子):
   for k = 1, ..., K:
     x̃ ~ LLM(x'_i | task, context, kernel)           // LLM 生成提案
     r̃ = evaluate(x̃)                                 // 评估新程序
     if r̃ > best_reward: best = x̃                    // 保留最优
   x''_i = best

4. 收敛判断: if λ_t >= 1.0 → converged = True
```

### 1.3 Island 并行与迁移

- **K 个独立 SMCIsland** 并行运行（`controller.py:46-57`）
- 每隔 `migration_interval` 个 epoch 进行一次 **merge-and-truncate 迁移**（`controller.py:63-110`）
- 通过 **derangement 置换** 确保每个 island 恰好是一个源和一个目标

---

## 2. Parent 选择与 Weight 计算

### 2.1 权重计算公式

权重计算发生在 `_resample()` 方法中（`island.py:194-212`）：

```python
# 对数权重 (unnormalized)
log_w[i] = Δβ × r_i    # 其中 Δβ = (λ_t - λ_{t-1}) × β_target

# 数值稳定的 exp-normalize 技巧
m = max(log_w)
w[i] = exp(log_w[i] - m)   # 减去最大值防止溢出

# 归一化
weights[i] = w[i] / Σ_j w[j]
```

**关键参数对权重的影响：**

| 参数 | 作用 | 典型值 |
|------|------|--------|
| `β_target` | 目标逆温度，控制选择压力的最终强度 | 20 |
| `Δβ` | 每步增量逆温度，由 ESS bisection 自适应决定 | 动态 |
| `κ` | ESS 阈值，间接控制 Δβ 的大小 | 0.9 |

**物理直觉**：Δβ 越大，高 reward 粒子的权重越集中（更强的选择压力）；κ 越接近 1.0，Δβ 越小（更温和的选择压力）。

### 2.2 Systematic Resampling（系统重采样）

采样方法使用的是 **systematic resampling**（`island.py:214-225`），这是一种低方差重采样方法：

```python
def _systematic_resample(self, weights: list[float]) -> list[int]:
    n = len(weights)
    u = rng.random() / n          # 单个随机数 u ∈ [0, 1/n)
    cumsum = 0.0
    indices = []
    j = 0
    for i in range(n):
        cumsum += weights[i]
        while j < n and u + j/n < cumsum:   # 等间距采样点
            indices.append(i)
            j += 1
    return indices
```

**与其他重采样方法的对比：**

| 方法 | 方差 | 随机数使用 | 特点 |
|------|------|-----------|------|
| Multinomial | 高 | N 个独立 | 最简单但方差大 |
| **Systematic** | **低** | **1 个** | **等间距采样，保持粒子多样性** |
| Stratified | 低 | N 个分层 | 与 systematic 类似 |
| Residual | 低 | 分确定性+随机 | 实现更复杂 |

选择 systematic resampling 的原因：只需**一个随机数**即可产生 N 个采样索引，产生的样本集具有**低方差**特性，在 SMC 文献中被认为是粒子滤波的最佳实践之一。

### 2.3 Best-of-K 变异策略

重采样后，每个粒子进入 `_best_of_k()` 变异阶段（`island.py:227-302`）：

```
对每个重采样粒子 x':
  best = x'          // 父粒子作为保底
  current = x'       // chain tip: 后续提案基于最新版本
  for k = 1, ..., K:
    x̃ = LLM(current | context)     // 生成提案
    r̃ = evaluate(x̃)                // 评估
    current = x̃                     // chain 始终前进（无论是否改进）
    if r̃ > best.reward:
      best = x̃                      // 更新最优
  return best                        // 返回 K 个提案中的最优
```

**设计要点：**

1. **Chain always advances**：`current = child` 无条件执行，使后续提案能基于新的代码状态继续改进，避免重复提出相同修改
2. **Best tracking**：`best` 只在严格改进时更新，最终返回的是所有提案中 reward 最高的
3. **Parent as fallback**：父粒子始终是候选之一，因此 reward **永远不会下降**
4. 这替代了传统 MH accept/reject 机制，避免浪费 LLM 调用在被拒绝的提案上

---

## 3. Context 选择与 Kernel 自适应挑选

### 3.1 四核设计 (2x2 Grid)

SMCEvolve 定义了 4 个进化 kernel，构成 **编辑粒度** × **信息源** 的 2×2 网格（`prompts.py:1-277`）：

```
                    无 Inspiration (单粒子)        有 Inspiration (交互)
                   ┌──────────────────────────┬─────────────────────────────┐
  Diff (局部小步)   │ K1: diff_no_inspo        │ K2: diff_with_inspo         │
                   │ • SEARCH/REPLACE 精确编辑 │ • 从参考程序借鉴技术        │
                   │ • 参数微调、bug 修复       │ • 通过小编辑移植特定模式     │
                   ├──────────────────────────┼─────────────────────────────┤
  Rewrite (全局大步)│ K3: rewrite_no_inspo     │ K4: rewrite_with_inspo      │
                   │ • 从头重新设计算法        │ • 智能交叉/重组              │
                   │ • 完全不同的方法          │ • 综合多个程序的最佳思想      │
                   └──────────────────────────┴─────────────────────────────┘
```

**SMC 理论对应关系：**

| Kernel | SMC 解释 | 提案分布 |
|--------|---------|---------|
| K1 (diff_no_inspo) | 单粒子局部核 K_t(x_{t-1}, ·) | 基于当前程序做小修改 |
| K2 (diff_with_inspo) | 交互局部核 K_t(x_{t-1}, · \| {x^(i)}) | 参考他人做小修改 |
| K3 (rewrite_no_inspo) | 单粒子全局核 | 独立重写整个程序 |
| K4 (rewrite_with_inspo) | 交互全局核（交叉/重组） | 综合多个程序重写 |

### 3.2 Kernel 选择策略

支持两种选择策略（`prompts.py:387-392`）：

#### 策略一：Weighted Sampling（静态加权采样）

```python
# kernel_selection == "weighted"
kernel = rng.choices(kernel_names, weights=weight_values, k=1)[0]
```

- 按照配置文件中 `kernel_weights` 的权重比例随机采样
- 默认四个 kernel 权重均为 1.0（均匀采样）
- 权重不需要归一化，内部自动处理

#### 策略二：Adaptive Thompson Sampling（自适应汤普森采样）

这是默认且推荐的策略（`prompts.py:423-433`）：

```python
# kernel_selection == "adaptive"
def _select_kernel_adaptive(self) -> str:
    best_name = self._kernel_names[0]
    best_sample = -1.0
    for name in self._kernel_names:
        s = self._ts[name]                              # Beta(α, β) 后验
        sample = rng.betavariate(s["alpha"], s["beta"])  # 从后验采样
        if sample > best_sample:
            best_sample = sample
            best_name = name
    return best_name
```

**Thompson Sampling 工作机制：**

1. **初始化**：每个 kernel 维护 Beta(α=1, β=1) 后验分布（即均匀分布）
2. **选择**：从每个 kernel 的 Beta 后验中采样一个值，选择采样值最大的 kernel
3. **更新**（`prompts.py:394-408`）：

```python
def update_kernel(self, kernel_name: str, improved: bool) -> None:
    # 1. 对所有 arm 进行衰减，保持对近期表现的响应性
    for stats in self._ts.values():
        stats["alpha"] = max(1.0, stats["alpha"] * 0.99)  # 衰减因子 γ = 0.99
        stats["beta"]  = max(1.0, stats["beta"]  * 0.99)

    # 2. 更新被拉动的 arm
    if improved:
        self._ts[kernel_name]["alpha"] += 1.0   # 成功: α 增加
    else:
        self._ts[kernel_name]["beta"]  += 1.0   # 失败: β 增加
```

**衰减机制的作用：**

- `γ = 0.99` 使所有 arm 的 α 和 β 缓慢衰减
- 效果：**近期的观测数据权重更高**，使后验能够响应 kernel 有效性随搜索阶段变化的情况
- `max(1.0, ...)` 确保参数不会衰减到 Beta 分布的有效范围以下
- 搜索早期（高 exploration），rewrite kernel 可能更有效；后期（高 exploitation），diff kernel 可能更有效。Thompson Sampling 能自动适应这种变化

### 3.3 编辑模式与响应解析

两种编辑模式对应不同的 LLM 输出格式和解析逻辑（`prompts.py:597-685`）：

**Diff 模式**（K1, K2）：
- LLM 输出 `<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` 格式
- 逐个应用编辑块，后面的块看到前面的修改结果
- 严格要求：SEARCH 必须精确匹配且唯一（byte-for-byte）
- 失败时回退到父程序（作为 no-op proposal）

**Rewrite 模式**（K3, K4）：
- LLM 输出完整的 fenced code block
- 提取代码块内容作为新程序
- 失败时回退到父程序

### 3.4 Kernel Fallback 机制

当带 inspiration 的 kernel 被选中但没有可用的 inspiration 时（如初始化阶段），系统自动降级到对应的 no-inspiration kernel（`prompts.py:467-481`）：

```
diff_with_inspo    → diff_no_inspo
rewrite_with_inspo → rewrite_no_inspo
```

---

## 4. Inspiration Program 选择机制

Inspiration 选择是 SMCEvolve 中实现**粒子间信息交互**的核心机制（`prompts.py:514-583`）。

### 4.1 两阶段选择策略

```
Inspiration 集合 = Top-K (利用) + Diverse-M (探索)
```

#### 阶段一：Top-K Selection（利用导向）

从当前 island 的**存活粒子**中选择 reward 最高的 K 个（排除父粒子自身）：

```python
# 过滤：排除自身、无效程序、NaN/Inf reward
live_pool = [p for p in island_particles if p.id != parent_id and valid(p)]
# 按 reward 降序排列
live_pool.sort(key=lambda p: p.reward, reverse=True)
# 取前 K 个
top_k = live_pool[:self.top_k_inspiration]  # 默认 K=2
```

#### 阶段二：Diverse-M Selection（探索导向，MAP-Elites 风格）

从**完整 archive**（所有历史评估过的程序）中选择 M 个**与 Top-K 最不相似**的程序：

```python
# 1. 构建多样性候选池（排除 parent 和已选的 top-k）
diverse_pool = [p for p in archive if p not in top_k_set and p != parent]

# 2. 嵌入所有候选程序（使用 Embedder，带缓存）
embeddings = await embedder.embed(all_programs)

# 3. 最远点采样 (Farthest-Point Sampling / Greedy k-Center)
selected = _farthest_point_select(embeddings, top_k_indices, diverse_indices, M)
```

### 4.2 Farthest-Point Sampling 算法

这是实现多样性选择的核心算法（`prompts.py:692-729`）：

```
输入: 
  embeddings: 所有候选的嵌入向量
  selected: 已选集合的索引（Top-K）
  remaining: 候选集合的索引（archive 中除 Top-K 外的程序）
  M: 需要选择的数量

算法:
  if selected 为空:
    从 remaining 中取 reward 最高的作为种子
    M -= 1

  for i = 1, ..., M:
    for each r in remaining:
      min_dist[r] = min_{s ∈ selected} ||embed[r] - embed[s]||_2
    best = argmax_{r ∈ remaining} min_dist[r]    // 选距已选集合最远的点
    selected.add(best)
    remaining.remove(best)

输出: selected (Top-K + 新选的 M 个多样性程序)
```

**算法特性：**
- 贪心近似 k-center 问题
- 保证每次选出的点与已选集合的最小距离最大化
- 时间复杂度：O(M × |remaining| × |selected|)

### 4.3 嵌入与缓存

Embedder（`embedder.py`）提供程序代码的向量化表示：

- **模型**：默认使用 Titan Text Embeddings V2（通过 LiteLLM 代理）
- **缓存**：基于 SHA256 哈希的内存缓存，避免重复 API 调用
- **归一化**：返回 L2 归一化的向量，使下游可以直接使用 L2 距离
- **容错**：API 失败时返回零向量

### 4.4 Inspiration 在 Prompt 中的呈现

选出的 inspiration 程序以 markdown 格式注入到 user prompt 中（`prompts.py:748-761`）：

```markdown
## Reference Program 1  (reward = 0.850000, top performer)

```python
# 高 reward 的参考程序代码
\```

## Reference Program 2  (reward = 0.620000, diverse approach)

```python
# 多样性选出的参考程序代码
\```
```

每个 inspiration 标注来源（`top performer` 或 `diverse approach`），帮助 LLM 区分利用与探索方向。

### 4.5 Archive 机制

每个 island 维护一个 `_archive`（`island.py:75-83`）：

```python
# 基于 SHA256 的去重字典: hash -> (program, reward)
def _archive_add(self, program: str, reward: float) -> None:
    key = hashlib.sha256(program.encode("utf-8")).hexdigest()[:16]
    prev = self._archive.get(key)
    if prev is None or reward > prev[1]:   # 同一程序只保留最高 reward
        self._archive[key] = (program, reward)
```

- 记录 island 上**所有历史评估过的程序**
- 同一程序（按内容哈希）只保留最高 reward
- 作为 Diverse-M 选择的候选池

---

## 5. SMC Process Control

### 5.1 Temperature Design（自适应温度调度）

#### 5.1.1 温度参数体系

SMCEvolve 使用三个层次的温度参数：

| 参数 | 符号 | 范围 | 含义 |
|------|------|------|------|
| 逆温度进度 | λ | [0, 1] | 从先验(0)到后验(1)的插值进度 |
| 目标逆温度 | β | 固定(如20) | 后验分布的"锐度" |
| 当前逆温度 | β_t = λ_t × β | [0, β] | 当前步的实际逆温度 |
| 增量逆温度 | Δβ = (λ_t - λ_{t-1}) × β | ≥ 0 | 本步增加的温度 |

**目标分布序列：** π_t(x) ∝ exp(β_t × r(x))，其中 β_t = λ_t × β

- λ = 0 时：均匀分布（完全探索）
- λ = 1 时：π(x) ∝ exp(β × r(x))（集中在高 reward 区域）
- λ 从 0 增长到 1 的过程就是从探索逐渐转向利用

#### 5.1.2 ESS Bisection 算法

自适应确定每步的 λ 增量，核心在 `temperature.py`：

**ESS 计算（`temperature.py:13-21`）：**

```python
def ess(rewards, delta_beta):
    """ESS = (Σ w_i)² / Σ w_i²"""
    if delta_beta == 0.0:
        return float(len(rewards))    # 无温度变化 → ESS = N
    log_w = [delta_beta * r for r in rewards]
    m = max(log_w)                     # 数值稳定
    w = [math.exp(lw - m) for lw in log_w]
    s1 = sum(w)
    s2 = sum(wi * wi for wi in w)
    return (s1 * s1) / s2
```

**Bisection 求解（`temperature.py:24-52`）：**

```python
def find_next_lambda(rewards, lam_prev, beta_target, kappa, 
                     lam_max=1.0, max_delta=None, tol=1e-4):
    n = len(rewards)
    target = kappa * n    # 目标 ESS = κN，如 κ=0.9, N=8 → target=7.2

    if max_delta is not None:
        lam_max = min(lam_max, lam_prev + max_delta)   # 限制最大步长

    # 如果直接跳到 lam_max 都满足 ESS ≥ target，直接跳
    if ess_at(lam_max) >= target:
        return lam_max

    # 二分搜索：找到使 ESS = κN 的 λ
    lo, hi = lam_prev, lam_max
    while hi - lo > tol:     # 收敛容差 1e-4
        mid = 0.5 * (lo + hi)
        if ess_at(mid) >= target:
            lo = mid          # ESS 还够大，可以进一步增加 λ
        else:
            hi = mid          # ESS 太小，需要减小 λ
    return lo
```

**为什么 bisection 可行？** ESS(λ) 是 λ 的**严格递减函数**：
- λ 增大 → Δβ 增大 → 权重更集中 → ESS 减小
- 因此在 (λ_prev, 1] 上存在唯一解

### 5.2 Exploration-Exploitation Control

SMCEvolve 通过多个机制平衡探索与利用：

#### 5.2.1 温度层面

| 阶段 | λ 值 | Δβ | 行为 |
|------|------|-----|------|
| 早期 | 接近 0 | 小 | 权重近似均匀 → **强探索** |
| 中期 | 0.3-0.7 | 中等 | 逐渐偏好高 reward → **平衡** |
| 晚期 | 接近 1 | 大 | 权重高度集中 → **强利用** |

#### 5.2.2 Kernel 层面

- **Rewrite kernels (K3, K4)**：全局探索，生成全新算法
- **Diff kernels (K1, K2)**：局部利用，精细调优
- **Thompson Sampling** 自动适应：早期可能偏好 rewrite（大步探索），后期偏好 diff（精细调优）

#### 5.2.3 Inspiration 层面

- **Top-K**：利用高 reward 粒子的信息
- **Diverse-M**：探索程序空间中不同区域的解法

#### 5.2.4 Island 层面

- **独立 island**：各自独立探索不同区域
- **迁移机制**：周期性分享信息，防止 island 陷入局部最优

### 5.3 ESS 设计

#### 5.3.1 κ 参数的意义

```
ESS 目标 = κ × N
```

- **κ = 0.9**（默认）：保留约 90% 的有效粒子
  - 温和的选择压力，每步只淘汰约 10% 的粒子
  - 需要更多步才能收敛到 λ = 1
- **κ → 1.0**：几乎不重采样，Δβ 极小，收敛极慢
- **κ → 0**：激进重采样，粒子多样性急剧下降（particle depletion）

#### 5.3.2 min_iterations 约束

```python
self.max_delta_lambda = 1.0 / min_iterations  # 如 min_iterations=3 → Δλ ≤ 1/3
```

即使 ESS bisection 允许更大的步长，`max_delta` 也会将其限制住：

```python
# find_next_lambda 中:
if max_delta is not None:
    lam_max = min(lam_max, lam_prev + max_delta)
```

这确保算法**至少经过 min_iterations 次迭代**才能从 λ=0 到达 λ=1，防止因 reward 分布过于均匀而一步跳到终点。

#### 5.3.3 ESS 在日志中的作用

每步记录 ESS 值用于诊断：

```python
snap = {
    "ess_at_lambda": ess_at_lambda,  # 实际 ESS 值
    "lambda": self.lam,               # 当前 λ
    "delta_beta": delta_beta,          # 本步 Δβ
    "beta_t": beta_t,                  # 当前总逆温度
}
```

如果 ESS 持续很低，说明粒子多样性不足；如果 ESS 始终接近 N，说明选择压力不够。

### 5.4 Island 迁移设计

#### 5.4.1 Derangement 置换

```python
def _derangement(self, k: int) -> list[int]:
    """生成随机置换，保证 perm[i] != i"""
    while True:
        perm = list(range(k))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(k)):
            return perm
```

- 保证每个 island 恰好是一个源和一个目标（**不自交换**）
- 对称性：信息流通量对所有 island 均等

#### 5.4.2 Merge-and-Truncate 选择

```python
# 合并 + 截断
combined = dst.particles + migrants
combined.sort(key=lambda p: p.reward, reverse=True)
dst.particles = combined[:dst.n_particles]
```

- 弱迁移体永远无法取代强原住民（**精英保留**）
- 只有当迁移体 reward 排在前 N 时才被接纳
- 不需要额外的接受概率计算

---

## 6. LLM 集成方式

### 6.1 多模型支持

SMCEvolve 支持同时配置多个 LLM 模型（`proposer.py:65-118`）：

```yaml
# configs/llm/openai.yaml
models:
  - name: gpt-5.4-mini-2026-03-17
    weight: 0.5                    # 50% 概率被选中
    input_price_per_mtok: 0.75
    output_price_per_mtok: 4.5
  - name: gemini-3-flash-preview
    weight: 0.5                    # 50% 概率被选中
    input_price_per_mtok: 0.5
    output_price_per_mtok: 3.0
```

每次 proposal 通过加权随机采样选择模型：

```python
def _pick_model(self) -> ModelSpec:
    return rng.choices(self.models, weights=self._weights, k=1)[0]
```

### 6.2 API 调用架构

```
OpenAIProposer
  ├── AsyncOpenAI client (兼容 OpenAI API / LiteLLM 代理)
  ├── Semaphore(max_concurrency=8)   // 并发控制
  └── Per-call flow:
        1. PromptManager.build() → (system_prompt, user_prompt)
        2. _pick_model() → 选择模型
        3. chat.completions.create(
              model=spec.name,
              messages=[system, user],
              temperature=1.0,
              max_tokens=4096,
              timeout=120.0
           )
        4. parse_response() → 新程序
        5. _record_cost() → 累计费用
```

### 6.3 LLM Temperature

```python
temperature=1.0  # 固定的 LLM 采样温度
```

- LLM temperature **固定为 1.0**，保持较高的生成多样性
- **注意区分**：这与 SMC 的逆温度 β 是完全不同的概念
  - LLM temperature：控制 LLM token 采样的随机性
  - SMC β：控制粒子权重的集中度

### 6.4 费用追踪

每次 API 调用后累计费用（`proposer.py:119-134`）：

```python
cost = prompt_tokens * input_price / 1M + completion_tokens * output_price / 1M
```

运行结束后输出完整的费用摘要，包括：
- 总费用
- 总 token 数（prompt + completion）
- 按模型分类的详细统计

### 6.5 Prompt 构建流程

完整的一次 proposal 生成过程（`proposer.py:136-209`）：

```
1. 从 context 提取: parent_reward, parent_id, island_particles, archive
2. PromptManager.build():
   a. select_kernel() → 选择 kernel（weighted 或 Thompson Sampling）
   b. 如果 kernel 需要 inspiration:
      - _select_inspirations() → Top-K + Diverse-M
      - 如果无可用 inspiration → fallback 到无 inspiration 的对应 kernel
   c. 填充模板: task description + 当前程序 + performance metrics + inspirations
3. 选择 LLM 模型
4. API 调用
5. parse_response(): 根据 edit_mode 解析为新程序
6. 返回 Proposal(program, prompt, response, metadata)
```

### 6.6 错误处理

- **LLM 调用失败**：返回父程序作为 proposal（等效于 no-op）
- **解析失败**：回退到父程序，记录 parse_issues
- **评估超时/崩溃**：`Evaluator` 在子进程中运行，超时返回 reward=0.0

---

## 附录 A：关键超参数总表

| 参数 | 配置路径 | 默认值 | 作用 |
|------|---------|--------|------|
| `n_islands` | algo.n_islands | 2 | 并行 island 数量 |
| `particles_per_island` | algo.particles_per_island | 8 | 每个 island 的粒子数 N |
| `beta` | algo.beta | 20 | 目标逆温度 β |
| `kappa` | algo.kappa | 0.9 | ESS 阈值 κ |
| `n_proposals` | algo.n_proposals | 2 | Best-of-K 中的 K |
| `min_iterations` | algo.min_iterations | 3 | 最小 SMC 步数 |
| `migration_interval` | algo.migration_interval | 3 | 迁移间隔（epoch 数） |
| `migration_size` | algo.migration_size | 1 | 每次迁移的粒子数 |
| `max_iterations` | algo.max_iterations | 30 | 全局最大 epoch 数 |
| `kernel_selection` | algo.prompt.kernel_selection | adaptive | kernel 选择策略 |
| `top_k_inspiration` | algo.prompt.top_k_inspiration | 2 | Top-K inspiration 数量 |
| `diverse_inspirations` | algo.prompt.diverse_inspirations | 2 | 多样性 inspiration 数量 |
| `temperature` | llm.temperature | 1.0 | LLM 采样温度 |
| `max_concurrency` | llm.max_concurrency | 8 | 最大并发 API 调用数 |

## 附录 B：核心文件索引

| 文件 | 职责 | 关键类/函数 |
|------|------|------------|
| `island.py` | 单链 SMC 算法 | `SMCIsland`, `step()`, `_resample()`, `_best_of_k()` |
| `temperature.py` | 自适应温度 | `ess()`, `find_next_lambda()` |
| `prompts.py` | Kernel 设计与管理 | `PromptManager`, `parse_response()`, `_farthest_point_select()` |
| `proposer.py` | LLM 调用 | `OpenAIProposer`, `Proposal`, `ModelSpec` |
| `controller.py` | Island 并行与迁移 | `IslandController`, `_migrate()`, `_derangement()` |
| `evaluator.py` | 程序评估 | `Evaluator.evaluate()` |
| `embedder.py` | 嵌入与缓存 | `Embedder.embed()` |
| `main.py` | 入口与编排 | `main()`, `_run()` |
