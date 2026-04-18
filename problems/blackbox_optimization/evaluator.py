"""Evaluator for BBOB black-box optimization (SMCEvolve protocol).

Protocol: evaluate(program: str) -> float
  - Executes program source to extract `run_search` function
  - Runs it on configured BBOB problems
  - Returns aggregate score (0.0 on failure)
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import yaml

try:
    import optunahub

    bbob = optunahub.load_module("benchmarks/bbob")
    Problem = getattr(bbob, "Problem", None)
except Exception as exc:
    Problem = None
    OPTUNAHUB_IMPORT_ERROR = str(exc)
else:
    OPTUNAHUB_IMPORT_ERROR = ""


EVALUATOR_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(EVALUATOR_DIR, "config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

PROBLEM_SETS = CONFIG.get("problems", {})
EVALUATOR_CFG = CONFIG.get("evaluator", {})
DEFAULT_TIMEOUT = int(EVALUATOR_CFG.get("timeout", 120))
GLOBAL_SEED = int(CONFIG.get("random_seed", 0) or 0)

# Reference values for scoring (computed via scipy fmin).
REF_VALUES: Dict[str, float] = {
    "sphere_d3_i1": 7.948000e01,
    "rosenbrock_d5_i2": -9.926454e02,
    "rastrigin_d10_i5": -7.173725e00,
    "ellipsoid_d20_i1": 1.677635e05,
    "schaffers_d40_i5": -1.381689e02,
}


def safe_float(value) -> float:
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v
    except Exception:
        return float("inf")


def score_value(best_value: float, ref_value: Optional[float]) -> float:
    if not math.isfinite(best_value):
        return 0.0

    if ref_value is None or not math.isfinite(ref_value):
        if best_value <= 0.0:
            return 1.0
        return float(1.0 / (1.0 + math.log10(1.0 + best_value)))

    if abs(ref_value) < 1e-12:
        improvement = 0.0 if best_value == 0.0 else -abs(best_value)
    else:
        improvement = (ref_value - best_value) / abs(ref_value)

    if improvement >= 0:
        return float(1.0 + improvement)
    return float(1.0 / (1.0 + abs(improvement)))


def score_efficiency(evaluations_used: float, budget: float) -> float:
    if budget <= 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - evaluations_used / budget)))


def _build_problem(case_cfg: Dict):
    if Problem is None:
        raise ImportError(
            "optunahub with benchmarks/bbob module is required. "
            f"Import error: {OPTUNAHUB_IMPORT_ERROR}"
        )
    return Problem(
        function_id=int(case_cfg["function_id"]),
        dimension=int(case_cfg["dimension"]),
        instance_id=int(case_cfg.get("instance_id", 1)),
    )


def _get_stage_cases() -> List[Dict]:
    if not isinstance(PROBLEM_SETS, dict):
        return []
    return PROBLEM_SETS.get("stage2", []) or []


def evaluate_case(run_search_fn, case_cfg: Dict, seed: int) -> Dict:
    problem_name = case_cfg.get("name") or (
        f"f{case_cfg.get('function_id')}_d{case_cfg.get('dimension')}"
    )
    budget = int(case_cfg.get("budget", 1000))

    try:
        problem = _build_problem(case_cfg)
    except Exception as exc:
        return {
            "name": problem_name,
            "status": "failed",
            "reason": f"problem_init_error: {exc}",
            "case_score": 0.0,
        }

    try:
        result = run_search_fn(problem, budget, seed)
    except Exception as exc:
        return {
            "name": problem_name,
            "status": "failed",
            "reason": f"run_error: {exc}",
            "case_score": 0.0,
        }

    if not isinstance(result, tuple) or len(result) not in (2, 3):
        return {
            "name": problem_name,
            "status": "failed",
            "reason": f"invalid_return: {type(result)}",
            "case_score": 0.0,
        }

    if len(result) == 3:
        best_x, best_value, evaluations_used = result
    else:
        best_x, best_value = result
        evaluations_used = budget

    best_value = safe_float(best_value)
    evaluations_used = max(1, min(budget, int(evaluations_used)))

    ref_value = REF_VALUES.get(problem_name)
    value_score = score_value(best_value, ref_value)
    efficiency = score_efficiency(evaluations_used, budget)
    case_score = 0.7 * value_score + 0.3 * efficiency

    return {
        "name": problem_name,
        "status": "ok",
        "best_value": best_value,
        "evaluations_used": evaluations_used,
        "budget": budget,
        "value_score": float(value_score),
        "efficiency": float(efficiency),
        "case_score": float(case_score),
    }


def evaluate(program: str) -> float:
    """SMCEvolve evaluator entry point.

    Executes the candidate program, extracts run_search, evaluates on BBOB
    problems, and returns aggregate score.
    """
    # Execute the program to extract run_search
    namespace: dict = {"__name__": "__evaluator__"}
    try:
        exec(program, namespace)
    except Exception:
        return 0.0

    run_search_fn = namespace.get("run_search") or namespace.get("run_search_entry")
    if not callable(run_search_fn):
        return 0.0

    cases = _get_stage_cases()
    if not cases:
        return 0.0

    case_results = []
    for idx, case_cfg in enumerate(cases):
        seed = GLOBAL_SEED + idx * 17 + int(case_cfg.get("instance_id", 1))
        case_results.append(evaluate_case(run_search_fn, case_cfg, seed))

    completed = [c for c in case_results if c.get("status") == "ok"]
    if not completed:
        return 0.0

    mean_case_score = float(np.mean([c["case_score"] for c in completed]))
    completion_rate = float(len(completed) / len(case_results))
    stage_score = float(0.9 * mean_case_score + 0.1 * completion_rate)

    return stage_score
