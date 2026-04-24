"""SMCEvolve evaluator — thin wrapper around the OpenEvolve evaluator.

Adapts the upstream evaluate(program_path) -> dict into
evaluate(program: str) -> float (returning combined_score).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from openevolve_evaluator import evaluate as _oe_evaluate  # noqa: E402
from smcevolve.openevolve_bridge import wrap_openevolve_evaluator  # noqa: E402

evaluate = wrap_openevolve_evaluator(_oe_evaluate, score_key="combined_score")
