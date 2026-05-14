# SMCEvolve Core Algorithm Technical Document

---

## Table of Contents

1. [Core Algorithm Overview](#1-core-algorithm-overview)
2. [Parent Selection and Weight Computation](#2-parent-selection-and-weight-computation)
3. [Context Selection and Adaptive Kernel Choice](#3-context-selection-and-adaptive-kernel-choice)
4. [Inspiration Program Selection](#4-inspiration-program-selection)
5. [SMC Process Control](#5-smc-process-control)
6. [LLM Integration](#6-llm-integration)

---

## 1. Core Algorithm Overview

SMCEvolve combines the **Sequential Monte Carlo (SMC)** framework with
**LLM-driven program evolution**, searching for high-quality programs by
iteratively sampling a particle population over program space.

### 1.1 Overall architecture

```
IslandController (K independent chains + migration)
  ├── SMCIsland_0 (N particles)
  │     ├── Adaptive temperature schedule (ESS bisection)
  │     ├── Softmax resampling
  │     └── Best-of-K mutation (LLM proposal + evaluation)
  ├── SMCIsland_1
  │     └── ...
  └── Periodic migration (merge-and-truncate)
```

### 1.2 One SMC step (Algorithm 1)

Each `SMCIsland.step()` runs the following three stages (`island.py:116-192`):

```
Input : particles {x_i, r_i}_{i=1}^N, current temperature param λ_{t-1}, target inverse temperature β
Output: updated particles, λ_t

1. Adaptive temperature schedule:
   λ_t = find_next_lambda(rewards, λ_{t-1}, β, κ)   // solve ESS(λ) = κN
   Δβ = (λ_t - λ_{t-1}) × β                         // incremental inverse temperature

2. Softmax resampling:
   w_i = exp(Δβ × r_i) / Σ_j exp(Δβ × r_j)         // normalized weights
   {x'_i} ~ Systematic_Resample(w)                  // systematic resampling

3. Best-of-K mutation (per resampled particle):
   for k = 1, ..., K:
     x̃ ~ LLM(x'_i | task, context, kernel)          // LLM-generated proposal
     r̃ = evaluate(x̃)                                // evaluate the new program
     if r̃ > best_reward: best = x̃                   // keep the best
   x''_i = best

4. Convergence check: if λ_t >= 1.0 → converged = True
```

### 1.3 Island parallelism and migration

- **K independent SMCIslands** run in parallel (`controller.py:46-57`)
- Every `migration_interval` epochs, a **merge-and-truncate migration** is
  performed (`controller.py:63-110`)
- A **derangement permutation** ensures every island is exactly one source and
  one destination

---

## 2. Parent Selection and Weight Computation

### 2.1 Weight formula

Weights are computed inside `_resample()` (`island.py:194-212`):

```python
# Log weights (unnormalized)
log_w[i] = Δβ × r_i    # where Δβ = (λ_t - λ_{t-1}) × β_target

# Numerically stable exp-normalize trick
m = max(log_w)
w[i] = exp(log_w[i] - m)   # subtract the max to prevent overflow

# Normalization
weights[i] = w[i] / Σ_j w[j]
```

**Key parameters affecting the weights:**

| Parameter   | Role                                                       | Typical value |
|-------------|------------------------------------------------------------|---------------|
| `β_target`  | Target inverse temperature; sets the final selection pressure | 20         |
| `Δβ`        | Per-step inverse-temperature increment, set adaptively by ESS bisection | dynamic |
| `κ`         | ESS threshold; indirectly controls the size of Δβ          | 0.9           |

**Physical intuition**: larger Δβ → weights concentrate on high-reward
particles (stronger selection pressure); κ closer to 1.0 → smaller Δβ (milder
selection pressure).

### 2.2 Systematic resampling

The resampler is **systematic resampling** (`island.py:214-225`), a
low-variance scheme:

```python
def _systematic_resample(self, weights: list[float]) -> list[int]:
    n = len(weights)
    u = rng.random() / n          # single random draw u ∈ [0, 1/n)
    cumsum = 0.0
    indices = []
    j = 0
    for i in range(n):
        cumsum += weights[i]
        while j < n and u + j/n < cumsum:   # evenly spaced sample points
            indices.append(i)
            j += 1
    return indices
```

**Comparison with other resampling schemes:**

| Method        | Variance | Random draws  | Notes                                |
|---------------|----------|---------------|--------------------------------------|
| Multinomial   | High     | N independent | Simplest; high variance              |
| **Systematic**| **Low**  | **1**         | **Evenly spaced; preserves diversity**|
| Stratified    | Low      | N stratified  | Similar to systematic                |
| Residual      | Low      | mixed         | More complex to implement            |

Reasons for choosing systematic resampling: only **one random draw** is needed
to produce N indices, the resulting sample has **low variance**, and it is
widely considered best practice for particle filters in the SMC literature.

### 2.3 Best-of-K mutation

After resampling, each particle goes through `_best_of_k()`
(`island.py:227-302`):

```
For each resampled particle x':
  best = x'          // parent particle as fallback
  current = x'       // chain tip: subsequent proposals build on the latest version
  for k = 1, ..., K:
    x̃ = LLM(current | context)     // generate proposal
    r̃ = evaluate(x̃)                // evaluate
    current = x̃                     // chain always advances (even if not improving)
    if r̃ > best.reward:
      best = x̃                      // update best
  return best                        // return the best of the K proposals
```

**Design points:**

1. **Chain always advances**: `current = child` is executed unconditionally,
   so subsequent proposals build on the latest state of the code and avoid
   repeatedly suggesting the same edits.
2. **Best tracking**: `best` is only updated on a strict improvement; the
   returned program is the highest-reward proposal seen.
3. **Parent as fallback**: the parent is always a candidate, so the reward
   **never decreases**.
4. This replaces a traditional MH accept/reject mechanism, avoiding wasted
   LLM calls on rejected proposals.

---

## 3. Context Selection and Adaptive Kernel Choice

### 3.1 Four-kernel design (2x2 grid)

SMCEvolve defines four evolution kernels arranged on a 2×2 grid of
**edit granularity** × **information source** (`prompts.py:1-277`):

```
                       No inspiration (single particle)   With inspiration (interactive)
                      ┌─────────────────────────────────┬────────────────────────────────┐
  Diff (local, small) │ K1: diff_no_inspo               │ K2: diff_with_inspo            │
                      │ • Precise SEARCH/REPLACE edits  │ • Borrow techniques from refs  │
                      │ • Parameter tweaks, bug fixes   │ • Transplant patterns via small│
                      │                                 │   edits                        │
                      ├─────────────────────────────────┼────────────────────────────────┤
  Rewrite (global,    │ K3: rewrite_no_inspo            │ K4: rewrite_with_inspo         │
  large)              │ • Redesign the algorithm        │ • Smart crossover / recombine  │
                      │ • Completely different approach │ • Combine best ideas from many │
                      └─────────────────────────────────┴────────────────────────────────┘
```

**Correspondence with SMC theory:**

| Kernel | SMC interpretation | Proposal distribution |
|--------|--------------------|-----------------------|
| K1 (diff_no_inspo)     | Single-particle local kernel K_t(x_{t-1}, ·) | Small edits to the current program |
| K2 (diff_with_inspo)   | Interactive local kernel K_t(x_{t-1}, · \| {x^(i)}) | Small edits guided by references |
| K3 (rewrite_no_inspo)  | Single-particle global kernel | Independent rewrite of the whole program |
| K4 (rewrite_with_inspo)| Interactive global kernel (crossover/recombination) | Rewrite combining multiple programs |

### 3.2 Kernel selection strategies

Two selection strategies are supported (`prompts.py:387-392`):

#### Strategy 1: Weighted sampling (static)

```python
# kernel_selection == "weighted"
kernel = rng.choices(kernel_names, weights=weight_values, k=1)[0]
```

- Randomly sample a kernel using `kernel_weights` from the config.
- All four kernels default to weight 1.0 (uniform sampling).
- Weights need not be normalized; this is handled internally.

#### Strategy 2: Adaptive Thompson Sampling

This is the default and recommended strategy (`prompts.py:423-433`):

```python
# kernel_selection == "adaptive"
def _select_kernel_adaptive(self) -> str:
    best_name = self._kernel_names[0]
    best_sample = -1.0
    for name in self._kernel_names:
        s = self._ts[name]                              # Beta(α, β) posterior
        sample = rng.betavariate(s["alpha"], s["beta"])  # sample from posterior
        if sample > best_sample:
            best_sample = sample
            best_name = name
    return best_name
```

**How Thompson Sampling works here:**

1. **Initialization**: each kernel maintains a Beta(α=1, β=1) posterior (i.e.
   the uniform distribution).
2. **Selection**: draw a value from each kernel's Beta posterior and pick the
   kernel with the largest draw.
3. **Update** (`prompts.py:394-408`):

```python
def update_kernel(self, kernel_name: str, improved: bool) -> None:
    # 1. Decay every arm so the posterior remains responsive to recent performance
    for stats in self._ts.values():
        stats["alpha"] = max(1.0, stats["alpha"] * 0.99)  # decay factor γ = 0.99
        stats["beta"]  = max(1.0, stats["beta"]  * 0.99)

    # 2. Update the arm that was pulled
    if improved:
        self._ts[kernel_name]["alpha"] += 1.0   # success: increment α
    else:
        self._ts[kernel_name]["beta"]  += 1.0   # failure: increment β
```

**Role of the decay term:**

- `γ = 0.99` slowly shrinks α and β for every arm.
- Effect: **recent observations are weighted more heavily**, so the posterior
  can respond to how kernel effectiveness changes across search phases.
- `max(1.0, ...)` keeps the parameters within the valid range of the Beta
  distribution.
- Early in the search (high exploration) rewrite kernels may dominate; later
  (high exploitation) diff kernels may dominate. Thompson Sampling adapts to
  this automatically.

### 3.3 Edit modes and response parsing

The two edit modes use different LLM output formats and parsing logic
(`prompts.py:597-685`):

**Diff mode** (K1, K2):
- LLM outputs `<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` blocks.
- Edits are applied sequentially so later blocks see earlier edits.
- Strict requirement: each SEARCH must match exactly and uniquely
  (byte-for-byte).
- On failure, fall back to the parent program (treated as a no-op proposal).

**Rewrite mode** (K3, K4):
- LLM outputs a complete fenced code block.
- The block's contents are used as the new program.
- On failure, fall back to the parent program.

### 3.4 Kernel fallback

When a kernel that requires inspiration is selected but none is available
(e.g. during initialization), the system automatically falls back to the
matching no-inspiration kernel (`prompts.py:467-481`):

```
diff_with_inspo    → diff_no_inspo
rewrite_with_inspo → rewrite_no_inspo
```

---

## 4. Inspiration Program Selection

Inspiration selection is the core mechanism in SMCEvolve for
**information exchange between particles** (`prompts.py:514-583`).

### 4.1 Two-stage selection

```
Inspiration set = Top-K (exploitation) + Diverse-M (exploration)
```

#### Stage 1: Top-K selection (exploitation)

Pick the top K **live particles** by reward from the current island
(excluding the parent itself):

```python
# Filter: drop self, invalid programs, NaN/Inf reward
live_pool = [p for p in island_particles if p.id != parent_id and valid(p)]
# Sort by reward, descending
live_pool.sort(key=lambda p: p.reward, reverse=True)
# Take the top K
top_k = live_pool[:self.top_k_inspiration]  # default K=2
```

#### Stage 2: Diverse-M selection (exploration, MAP-Elites style)

Pick M programs from the **full archive** (every evaluated program in the
island's history) that are **most dissimilar to Top-K**:

```python
# 1. Build the diversity pool (exclude parent and the already-picked top-k)
diverse_pool = [p for p in archive if p not in top_k_set and p != parent]

# 2. Embed all candidate programs (cached via Embedder)
embeddings = await embedder.embed(all_programs)

# 3. Farthest-point sampling (greedy k-center)
selected = _farthest_point_select(embeddings, top_k_indices, diverse_indices, M)
```

### 4.2 Farthest-point sampling algorithm

This is the core routine for diversity selection (`prompts.py:692-729`):

```
Input:
  embeddings: embedding vectors for every candidate
  selected  : indices of the already-selected set (Top-K)
  remaining : indices of candidates (archive minus Top-K)
  M         : number of new programs to select

Algorithm:
  if selected is empty:
    take the highest-reward program from remaining as the seed
    M -= 1

  for i = 1, ..., M:
    for each r in remaining:
      min_dist[r] = min_{s ∈ selected} ||embed[r] - embed[s]||_2
    best = argmax_{r ∈ remaining} min_dist[r]    // farthest point from selected
    selected.add(best)
    remaining.remove(best)

Output: selected (Top-K + the M new diverse programs)
```

**Properties:**
- Greedy approximation to the k-center problem.
- Maximizes the minimum distance from each newly chosen point to the already
  selected set.
- Time complexity: O(M × |remaining| × |selected|).

### 4.3 Embeddings and caching

The `Embedder` (`embedder.py`) provides vector representations of program
code:

- **Model**: defaults to OpenAI `text-embedding-3-small`.
- **Cache**: in-memory cache keyed by SHA-256 hash to avoid repeated API
  calls.
- **Normalization**: returns L2-normalized vectors so downstream code can
  use L2 distance directly.
- **Fault tolerance**: returns a zero vector if the API call fails.

### 4.4 How inspirations appear in the prompt

Selected inspirations are injected into the user prompt as markdown
(`prompts.py:748-761`):

```markdown
## Reference Program 1  (reward = 0.850000, top performer)

```python
# High-reward reference program code
\```

## Reference Program 2  (reward = 0.620000, diverse approach)

```python
# Diversity-selected reference program code
\```
```

Each inspiration is tagged with its source (`top performer` or
`diverse approach`), helping the LLM distinguish exploitation from
exploration directions.

### 4.5 Archive

Each island maintains an `_archive` (`island.py:75-83`):

```python
# SHA-256-keyed dedup dict: hash -> (program, reward)
def _archive_add(self, program: str, reward: float) -> None:
    key = hashlib.sha256(program.encode("utf-8")).hexdigest()[:16]
    prev = self._archive.get(key)
    if prev is None or reward > prev[1]:   # keep the best reward per program
        self._archive[key] = (program, reward)
```

- Records **every program ever evaluated** on the island.
- Each unique program (by content hash) keeps only the highest reward
  observed.
- Serves as the candidate pool for Diverse-M selection.

---

## 5. SMC Process Control

### 5.1 Temperature design (adaptive temperature schedule)

#### 5.1.1 Temperature parameter hierarchy

SMCEvolve uses three layers of temperature parameters:

| Parameter                  | Symbol | Range      | Meaning |
|---------------------------|--------|------------|---------|
| Inverse-temperature progress | λ    | [0, 1]     | Interpolation from prior (0) to posterior (1) |
| Target inverse temperature   | β    | fixed (e.g. 20) | "Sharpness" of the posterior |
| Current inverse temperature  | β_t = λ_t × β | [0, β] | Actual inverse temperature at step t |
| Incremental inverse temperature | Δβ = (λ_t - λ_{t-1}) × β | ≥ 0 | Increase added this step |

**Target distribution sequence:** π_t(x) ∝ exp(β_t × r(x)), with
β_t = λ_t × β.

- λ = 0: uniform distribution (pure exploration).
- λ = 1: π(x) ∝ exp(β × r(x)) (concentrated on high-reward regions).
- Growing λ from 0 to 1 is the transition from exploration to exploitation.

#### 5.1.2 ESS bisection algorithm

The per-step λ increment is determined adaptively in `temperature.py`:

**ESS computation (`temperature.py:13-21`):**

```python
def ess(rewards, delta_beta):
    """ESS = (Σ w_i)² / Σ w_i²"""
    if delta_beta == 0.0:
        return float(len(rewards))    # no temperature change → ESS = N
    log_w = [delta_beta * r for r in rewards]
    m = max(log_w)                     # numerical stability
    w = [math.exp(lw - m) for lw in log_w]
    s1 = sum(w)
    s2 = sum(wi * wi for wi in w)
    return (s1 * s1) / s2
```

**Bisection solver (`temperature.py:24-52`):**

```python
def find_next_lambda(rewards, lam_prev, beta_target, kappa,
                     lam_max=1.0, max_delta=None, tol=1e-4):
    n = len(rewards)
    target = kappa * n    # target ESS = κN, e.g. κ=0.9, N=8 → target=7.2

    if max_delta is not None:
        lam_max = min(lam_max, lam_prev + max_delta)   # cap step size

    # If jumping straight to lam_max already satisfies ESS ≥ target, take it
    if ess_at(lam_max) >= target:
        return lam_max

    # Binary search for λ such that ESS = κN
    lo, hi = lam_prev, lam_max
    while hi - lo > tol:     # convergence tolerance 1e-4
        mid = 0.5 * (lo + hi)
        if ess_at(mid) >= target:
            lo = mid          # ESS still large enough, push λ further
        else:
            hi = mid          # ESS too small, pull λ back
    return lo
```

**Why bisection works**: ESS(λ) is a **strictly decreasing function** of λ:
- λ ↑ → Δβ ↑ → weights concentrate → ESS ↓
- so there is a unique solution on (λ_prev, 1].

### 5.2 Exploration-exploitation control

SMCEvolve balances exploration and exploitation through several mechanisms:

#### 5.2.1 Temperature

| Phase | λ value     | Δβ     | Behavior |
|-------|-------------|--------|----------|
| Early | near 0      | small  | Weights almost uniform → **strong exploration** |
| Mid   | 0.3–0.7     | medium | Gradually favors high reward → **balanced** |
| Late  | near 1      | large  | Weights highly concentrated → **strong exploitation** |

#### 5.2.2 Kernel

- **Rewrite kernels (K3, K4)**: global exploration; generate new algorithms.
- **Diff kernels (K1, K2)**: local exploitation; fine-grained tuning.
- **Thompson Sampling** adapts automatically: rewrite kernels may dominate
  early (large exploratory jumps), diff kernels may dominate later (fine
  tuning).

#### 5.2.3 Inspiration

- **Top-K**: exploits information from high-reward particles.
- **Diverse-M**: explores different regions of program space.

#### 5.2.4 Island

- **Independent islands**: each explores a different region.
- **Migration**: periodically shares information, preventing islands from
  getting stuck in local optima.

### 5.3 ESS design

#### 5.3.1 Meaning of κ

```
target ESS = κ × N
```

- **κ = 0.9** (default): retain ~90% effective particles.
  - Mild selection pressure; ~10% of particles are pruned per step.
  - Needs more steps to converge to λ = 1.
- **κ → 1.0**: almost no resampling; Δβ tiny; convergence extremely slow.
- **κ → 0**: aggressive resampling; particle diversity collapses rapidly
  (particle depletion).

#### 5.3.2 `min_iterations` constraint

```python
self.max_delta_lambda = 1.0 / min_iterations  # e.g. min_iterations=3 → Δλ ≤ 1/3
```

Even when ESS bisection allows a larger step, `max_delta` caps it:

```python
# inside find_next_lambda:
if max_delta is not None:
    lam_max = min(lam_max, lam_prev + max_delta)
```

This ensures the algorithm runs **at least min_iterations** times to go from
λ=0 to λ=1, preventing a single-step jump when the reward distribution is
too uniform.

#### 5.3.3 ESS in logs

ESS is recorded at every step for diagnostics:

```python
snap = {
    "ess_at_lambda": ess_at_lambda,  # actual ESS value
    "lambda": self.lam,               # current λ
    "delta_beta": delta_beta,          # this step's Δβ
    "beta_t": beta_t,                  # current total inverse temperature
}
```

Persistently low ESS signals insufficient particle diversity; ESS that stays
close to N signals insufficient selection pressure.

### 5.4 Island migration design

#### 5.4.1 Derangement permutation

```python
def _derangement(self, k: int) -> list[int]:
    """Random permutation with perm[i] != i for all i."""
    while True:
        perm = list(range(k))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(k)):
            return perm
```

- Guarantees each island is exactly one source and one destination
  (**no self-migration**).
- Symmetric: information flow is balanced across islands.

#### 5.4.2 Merge-and-truncate selection

```python
# merge + truncate
combined = dst.particles + migrants
combined.sort(key=lambda p: p.reward, reverse=True)
dst.particles = combined[:dst.n_particles]
```

- Weak migrants can never displace strong residents (**elitism preserved**).
- A migrant is accepted only if its reward lands in the top N.
- No separate acceptance probability is needed.

---

## 6. LLM Integration

### 6.1 Multi-model support

SMCEvolve supports configuring multiple LLM models at once
(`proposer.py:65-118`):

```yaml
# configs/llm/openai.yaml
models:
  - name: gpt-5.4-mini-2026-03-17
    weight: 0.5                    # 50% chance of being selected
    input_price_per_mtok: 0.75
    output_price_per_mtok: 4.5
  - name: gemini-3-flash-preview
    weight: 0.5                    # 50% chance of being selected
    input_price_per_mtok: 0.5
    output_price_per_mtok: 3.0
```

Each proposal picks a model by weighted random sampling:

```python
def _pick_model(self) -> ModelSpec:
    return rng.choices(self.models, weights=self._weights, k=1)[0]
```

### 6.2 API call architecture

```
OpenAIProposer
  ├── AsyncOpenAI client (OpenAI-API compatible / LiteLLM proxy)
  ├── Semaphore(max_concurrency=8)   // concurrency control
  └── Per-call flow:
        1. PromptManager.build() → (system_prompt, user_prompt)
        2. _pick_model() → choose a model
        3. chat.completions.create(
              model=spec.name,
              messages=[system, user],
              temperature=1.0,
              max_tokens=4096,
              timeout=120.0
           )
        4. parse_response() → new program
        5. _record_cost() → accumulate cost
```

### 6.3 LLM temperature

```python
temperature=1.0  # fixed LLM sampling temperature
```

- LLM temperature is **fixed at 1.0** to keep generations diverse.
- **Important distinction**: this is a different concept from the SMC inverse
  temperature β.
  - LLM temperature: controls randomness of LLM token sampling.
  - SMC β: controls the concentration of particle weights.

### 6.4 Cost tracking

Cost is accumulated after each API call (`proposer.py:119-134`):

```python
cost = prompt_tokens * input_price / 1M + completion_tokens * output_price / 1M
```

A full cost summary is printed at the end of the run, including:
- Total cost
- Total tokens (prompt + completion)
- Per-model breakdown

### 6.5 Prompt construction flow

A complete proposal generation cycle (`proposer.py:136-209`):

```
1. Extract from context: parent_reward, parent_id, island_particles, archive
2. PromptManager.build():
   a. select_kernel() → choose a kernel (weighted or Thompson Sampling)
   b. If the kernel needs inspiration:
      - _select_inspirations() → Top-K + Diverse-M
      - If no inspiration is available → fall back to the matching
        no-inspiration kernel
   c. Fill template: task description + current program + performance metrics + inspirations
3. Pick an LLM model
4. API call
5. parse_response(): parse into a new program according to edit_mode
6. Return Proposal(program, prompt, response, metadata)
```

### 6.6 Error handling

- **LLM call fails**: return the parent program as the proposal (no-op
  equivalent).
- **Parse failure**: fall back to the parent program; record `parse_issues`.
- **Evaluation timeout/crash**: `Evaluator` runs in a subprocess; on timeout
  it returns reward=0.0.

---

## Appendix A: Key hyperparameters

| Parameter              | Config path                       | Default | Role |
|-----------------------|-----------------------------------|---------|------|
| `n_islands`           | algo.n_islands                    | 2       | Number of parallel islands |
| `particles_per_island`| algo.particles_per_island         | 8       | Number of particles N per island |
| `beta`                | algo.beta                         | 20      | Target inverse temperature β |
| `kappa`               | algo.kappa                        | 0.9     | ESS threshold κ |
| `n_proposals`         | algo.n_proposals                  | 2       | K in best-of-K |
| `min_iterations`      | algo.min_iterations               | 3       | Minimum SMC steps |
| `migration_interval`  | algo.migration_interval           | 3       | Migration interval (epochs) |
| `migration_size`      | algo.migration_size               | 1       | Number of migrants per migration |
| `max_iterations`      | algo.max_iterations               | 30      | Global maximum number of epochs |
| `kernel_selection`    | algo.prompt.kernel_selection      | adaptive| Kernel selection strategy |
| `top_k_inspiration`   | algo.prompt.top_k_inspiration     | 2       | Number of Top-K inspirations |
| `diverse_inspirations`| algo.prompt.diverse_inspirations  | 2       | Number of diversity inspirations |
| `temperature`         | llm.temperature                   | 1.0     | LLM sampling temperature |
| `max_concurrency`     | llm.max_concurrency               | 8       | Maximum concurrent API calls |

## Appendix B: Core file index

| File             | Responsibility               | Key classes / functions |
|------------------|------------------------------|--------------------------|
| `island.py`      | Single-chain SMC algorithm   | `SMCIsland`, `step()`, `_resample()`, `_best_of_k()` |
| `temperature.py` | Adaptive temperature         | `ess()`, `find_next_lambda()` |
| `prompts.py`     | Kernel design and management | `PromptManager`, `parse_response()`, `_farthest_point_select()` |
| `proposer.py`    | LLM calls                    | `OpenAIProposer`, `Proposal`, `ModelSpec` |
| `controller.py`  | Island parallelism and migration | `IslandController`, `_migrate()`, `_derangement()` |
| `evaluator.py`   | Program evaluation           | `Evaluator.evaluate()` |
| `embedder.py`    | Embeddings and caching       | `Embedder.embed()` |
| `main.py`        | Entry point and orchestration| `main()`, `_run()` |
