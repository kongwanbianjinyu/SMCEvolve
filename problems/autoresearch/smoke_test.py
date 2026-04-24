"""End-to-end smoke test: evaluate initial_program.py via the SMCEvolve
evaluator harness, log reward + elapsed time.

Run:
    source .venv/bin/activate
    python problems/autoresearch/smoke_test.py                 # default 600 s timeout
    python problems/autoresearch/smoke_test.py --timeout 300   # tighter
    AR_TIME_BUDGET=20 python problems/autoresearch/smoke_test.py --timeout 120

Prereqs:
  1. `uv pip install -r problems/autoresearch/requirements.txt`
  2. `python problems/autoresearch/prepare.py`   # downloads data + trains BPE
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smcevolve.evaluator import Evaluator  # noqa: E402


async def main_async(timeout: float) -> int:
    evaluator = Evaluator(HERE / "evaluator.py", timeout=timeout)
    program = (HERE / "initial_program.py").read_text(encoding="utf-8")
    print(f"running seed candidate (timeout={timeout:.0f}s)...")
    t0 = time.perf_counter()
    try:
        reward = await evaluator.evaluate(program)
    except Exception as exc:
        print(f"EXCEPTION: {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.perf_counter() - t0
    print(f"reward = {reward:.6f}  (elapsed {elapsed:.1f}s)")
    return 0 if reward > 0 else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeout", type=float, default=600.0,
                   help="per-candidate timeout in seconds (default: 600)")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args().timeout)))
