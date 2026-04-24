"""SMCEvolve evaluator for the autoresearch pretraining problem.

Signature: evaluate(program: str) -> float  (higher = better, 0.0 on failure).

The candidate program is a full-file replacement for `train.py` — a
single-file GPT pretraining script that imports the FIXED harness
constants (`MAX_SEQ_LEN`, `TIME_BUDGET`, `evaluate_bpb`, ...) from
`prepare.py` in this directory. The script must train for at most
`TIME_BUDGET` seconds and print a summary whose final line contains

    val_bpb:   <float>

We parse `val_bpb` and return `max(0, 2.0 - val_bpb)` so that higher
is better and random-model / crashed runs both score 0.0. The 2.0
offset was chosen because a random 8192-vocab model over ~5-byte
tokens scores val_bpb ≈ 2.6, so any trained model lands in (0, 2.0).

This evaluator runs the candidate via `exec()` in-process. That is
safe because SMCEvolve already spawns a fresh subprocess (its own
`_eval_runner.py`) for each candidate — the CUDA context is created
and torn down per evaluation.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

_PROBLEM_DIR = Path(__file__).resolve().parent
if str(_PROBLEM_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBLEM_DIR))

_VAL_BPB_RE = re.compile(r"^\s*val_bpb:\s+(\d+(?:\.\d+)?)", re.MULTILINE)


def _parse_val_bpb(output: str) -> float | None:
    matches = _VAL_BPB_RE.findall(output)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def evaluate(program: str) -> float:
    buf = io.StringIO()
    exec_globals: dict = {
        "__name__": "__main__",
        "__file__": str(_PROBLEM_DIR / "candidate.py"),
    }
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                code = compile(program, "<smcevolve_candidate>", "exec")
            except SyntaxError:
                return 0.0
            try:
                exec(code, exec_globals)
            except SystemExit as e:
                # initial_program.py prints "FAIL" and `raise SystemExit(1)` on
                # NaN/exploding loss — treat as a failed candidate.
                if e.code not in (None, 0):
                    return 0.0
            except BaseException:
                return 0.0
    finally:
        # Best-effort cleanup: drop CUDA memory cache if torch was imported.
        mod = sys.modules.get("torch")
        if mod is not None:
            try:
                if mod.cuda.is_available():
                    mod.cuda.empty_cache()
            except Exception:
                pass

    val_bpb = _parse_val_bpb(buf.getvalue())
    if val_bpb is None:
        return 0.0
    # Lower val_bpb = better; clip so reward >= 0 and random-model ≈ 0.
    return max(0.0, 2.0 - val_bpb)
