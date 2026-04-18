# EVOLVE-BLOCK-START
"""Baseline black-box optimizer for BBOB problems."""

from typing import Sequence, Tuple, Union
import numpy as np


def _get_bounds(problem, dimension: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract bounds from the problem or fall back to a symmetric box."""
    lower = getattr(problem, "lower_bounds", None)
    upper = getattr(problem, "upper_bounds", None)

    if lower is None or upper is None:
        lower = [-5.0] * dimension
        upper = [5.0] * dimension

    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)

    if lower_arr.shape[0] != dimension or upper_arr.shape[0] != dimension:
        lower_arr = np.full(dimension, float(lower_arr.flat[0]))
        upper_arr = np.full(dimension, float(upper_arr.flat[0]))

    return lower_arr, upper_arr


def _sample_uniform(rng: np.random.Generator, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Uniform sample inside the box."""
    return lower + rng.random(size=lower.shape[0]) * (upper - lower)


def _clip_to_bounds(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(x, lower, upper)


def _to_params(x: np.ndarray) -> dict:
    """Convert vector to the dict format expected by bbob Problem.evaluate."""
    return {f"x{i}": float(v) for i, v in enumerate(x)}


def _evaluate_safe(problem, x: np.ndarray) -> float:
    """Evaluate the problem and guard against failures."""
    try:
        params = _to_params(x)
        value = problem.evaluate(params)
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return float("inf")
        return value
    except Exception:
        return float("inf")


def run_search(problem, budget: int = 1000, seed: int | None = None) -> Tuple[list[float], float, int]:
    """
    Simple budgeted random search: sample uniformly in the box and keep the best.
    Deterministic under `seed`.
    """
    rng = np.random.default_rng(seed)

    dimension = getattr(problem, "dimension", None)
    if dimension is None:
        lower_attr = getattr(problem, "lower_bounds", [])
        upper_attr = getattr(problem, "upper_bounds", [])
        dimension = len(lower_attr) or len(upper_attr) or 2

    lower, upper = _get_bounds(problem, dimension)

    evaluations_used = 0
    best_x = None
    best_value = float("inf")

    while evaluations_used < budget:
        candidate = _sample_uniform(rng, lower, upper)
        value = _evaluate_safe(problem, candidate)
        evaluations_used += 1

        if value < best_value:
            best_value = value
            best_x = candidate

    # Fallback if everything failed/returned inf
    if best_x is None or not np.isfinite(best_value):
        best_x = _clip_to_bounds((lower + upper) / 2.0, lower, upper)
        best_value = _evaluate_safe(problem, best_x)
        if not np.isfinite(best_value):
            best_value = float("inf")

    return best_x.tolist(), float(best_value), evaluations_used


# EVOLVE-BLOCK-END


def run_search_entry(problem, budget: int = 1000, seed: int | None = None):
    """
    Thin wrapper kept outside the evolve block in case the block is replaced.
    """
    return run_search(problem, budget=budget, seed=seed)


if __name__ == "__main__":
    # Smoke test with a trivial sphere if optunahub is available
    try:
        import optunahub

        bbob = optunahub.load_module("benchmarks/bbob")
        test_problem = bbob.Problem(function_id=1, dimension=3, instance_id=1)
        x, value, used = run_search(test_problem, budget=100, seed=0)
        print(f"Best value {value:.4e} after {used} evals at x={x}")
    except Exception as exc:
        print(f"Smoke test skipped ({exc})")
