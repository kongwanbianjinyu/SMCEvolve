# Black-Box Optimization (BBOB)

Implement `run_search(problem, budget, seed)` that minimizes a black-box objective function.

## Function Signature

```python
def run_search(problem, budget: int = 1000, seed: int | None = None) -> Tuple[list[float], float, int]:
```

### Parameters
- `problem`: a BBOB Problem object with:
  - `problem.evaluate(params)` — takes a dict `{"x0": v0, "x1": v1, ...}` and returns a float objective value (minimize)
  - `problem.dimension` — int, the search space dimensionality
  - `problem.lower_bounds` / `problem.upper_bounds` — lists of per-dimension bounds (default [-5, 5])
- `budget`: maximum number of evaluations allowed
- `seed`: random seed for reproducibility

### Returns
- `best_x`: list of floats — the best solution found
- `best_value`: float — objective value at best_x (lower is better)
- `evaluations_used`: int — number of evaluations consumed (must be ≤ budget)

## Constraints
- You must NOT exceed `budget` evaluations of `problem.evaluate(...)`.
- The optimizer must be deterministic under the given `seed`.
- All solutions must stay within `[lower_bounds, upper_bounds]`.

## Evaluation
- Problems span mixed BBOB functions (sphere, rosenbrock, rastrigin, ellipsoid, schaffers) across dimensions 5–40.
- Score = 0.7 × value_quality + 0.3 × efficiency (using fewer evaluations is better).
- Value quality compares against a reference solution: score > 1.0 means beating the reference.

## Tips
- Consider population-based methods (CMA-ES, differential evolution) for multimodal functions.
- Adapt step sizes / mutation rates to the budget and dimensionality.
- The budget is tight relative to dimensionality — be efficient.

Maximize the combined score across all test problems.
