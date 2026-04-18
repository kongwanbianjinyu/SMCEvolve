"""Subprocess entry point: import a problem evaluator, run it on stdin program.

Protocol:
  argv[1] = absolute path to the problem's evaluator.py (must define
            `evaluate(program: str) -> float`)
  stdin   = full source of the candidate program
  stdout  = single line containing the float reward
  exit 1  = evaluator raised, output is invalid, or import failed
"""

from __future__ import annotations

import importlib.util
import math
import sys
import traceback
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: _eval_runner.py <evaluator_path>", file=sys.stderr)
        return 2

    evaluator_path = Path(sys.argv[1])
    program = sys.stdin.read()

    spec = importlib.util.spec_from_file_location(
        f"problem_evaluator_{evaluator_path.stem}", evaluator_path
    )
    if spec is None or spec.loader is None:
        print(f"cannot import evaluator from {evaluator_path}", file=sys.stderr)
        return 1
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        traceback.print_exc()
        return 1
    if not hasattr(module, "evaluate"):
        print(f"{evaluator_path} must define evaluate(program: str) -> float", file=sys.stderr)
        return 1

    try:
        reward = float(module.evaluate(program))
    except Exception:
        traceback.print_exc()
        return 1

    if math.isnan(reward):
        reward = 0.0

    print(reward)
    return 0


if __name__ == "__main__":
    sys.exit(main())
