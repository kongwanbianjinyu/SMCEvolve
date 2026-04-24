"""Smoke test: verify every AlgoTune evaluator runs end-to-end.

Walks task folders directly under `problems/algotune/`, dynamically
imports each task's `evaluator.py`, calls
`evaluate(initial_program_source)`, and records the returned
`speedup_score`, elapsed wall-clock time, and any exception.

Usage (from repo root):

    python problems/algotune/smoke_test.py
    python problems/algotune/smoke_test.py --tasks psd_cone_projection lu_factorization

Writes:
    problems/algotune/logs/smoke_<UTC-timestamp>.log   (detailed)
    problems/algotune/logs/smoke_<UTC-timestamp>.json  (summary)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent.parent
LOGS_DIR = HERE / "logs"

# Names of scripts / files directly under problems/algotune/ that are
# NOT task folders — skip these when walking the directory.
NON_TASK_NAMES = {
    "create_task.py",
    "generate_all_tasks.py",
    "generate_smc_configs.py",
    "run_benchmark.py",
    "smoke_test.py",
    "task_adapter.py",
    "requirements.txt",
    "README.md",
    "__pycache__",
    "logs",
}

# Modules cached by the task `evaluator.py` imports that would leak
# state between tasks if reused. Every task's evaluator.py does
# `from openevolve_evaluator import evaluate` after prepending its
# own dir to sys.path; without a purge, task #2 gets task #1's
# cached module (with its baked-in paths / imports).
_STALE_MODULE_PREFIXES = ("oe_candidate_", "_smoke_evaluator_")
_STALE_MODULE_NAMES = {
    "openevolve_evaluator",
    "initial_program",
    "solution",
}


def _purge_task_modules() -> None:
    stale = [
        name for name in list(sys.modules)
        if name in _STALE_MODULE_NAMES
        or name.startswith(_STALE_MODULE_PREFIXES)
    ]
    for name in stale:
        sys.modules.pop(name, None)


def _load_evaluator(evaluator_path: Path):
    task = evaluator_path.parent.name
    mod_name = f"_smoke_evaluator_algotune_{task}"
    task_dir = str(evaluator_path.parent)
    added_path = False
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)
        added_path = True
    try:
        spec = importlib.util.spec_from_file_location(mod_name, evaluator_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load spec for {evaluator_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise
        if not hasattr(module, "evaluate"):
            raise AttributeError(f"{evaluator_path} has no `evaluate`")
        return module.evaluate, mod_name
    finally:
        if added_path:
            try:
                sys.path.remove(task_dir)
            except ValueError:
                pass


def _discover_tasks(tasks_filter: list[str] | None):
    missing = []
    if tasks_filter:
        for name in tasks_filter:
            task_dir = HERE / name
            if not task_dir.is_dir():
                missing.append(name)
            elif not (task_dir / "evaluator.py").is_file():
                missing.append(f"{name} (no evaluator.py)")
        if missing:
            raise SystemExit(f"unknown / incomplete tasks: {missing}")
    for child in sorted(HERE.iterdir()):
        if not child.is_dir() or child.name in NON_TASK_NAMES:
            continue
        if tasks_filter and child.name not in tasks_filter:
            continue
        required = {"initial_program.py", "evaluator.py"}
        have = {p.name for p in child.iterdir()}
        if not required.issubset(have):
            continue
        yield child.name, child


def run(tasks_filter: list[str] | None) -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOGS_DIR / f"smoke_{ts}.log"
    summary_path = LOGS_DIR / f"smoke_{ts}.json"

    logger = logging.getLogger(f"algotune_smoke_{ts}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    logger.info("AlgoTune smoke test")
    logger.info("log file: %s", log_path)
    logger.info("tasks filter: %s", tasks_filter or "ALL")

    results: list[dict] = []
    t_start = time.monotonic()

    for task_name, task_dir in _discover_tasks(tasks_filter):
        prog_path = task_dir / "initial_program.py"
        eval_path = task_dir / "evaluator.py"
        program = prog_path.read_text(encoding="utf-8")

        entry: dict = {
            "task": task_name,
            "task_dir": str(task_dir.relative_to(REPO_ROOT)),
            "status": None,
            "score": None,
            "elapsed_s": None,
            "error": None,
        }

        _purge_task_modules()
        t0 = time.monotonic()
        evaluate = None
        mod_name = None
        try:
            evaluate, mod_name = _load_evaluator(eval_path)
            score = evaluate(program)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            entry["status"] = "error"
            entry["elapsed_s"] = round(elapsed, 3)
            entry["error"] = f"{type(exc).__name__}: {exc}"
            logger.error("[ERROR] %-28s  %.2fs  %s",
                         task_name, elapsed, entry["error"])
            logger.error("traceback:\n%s", traceback.format_exc())
        else:
            elapsed = time.monotonic() - t0
            entry["elapsed_s"] = round(elapsed, 3)
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = None
            entry["score"] = score_f
            # speedup_score:
            #   - 0.0 is the wrap_openevolve_evaluator error sentinel
            #     (evaluator raised) and also what the upstream
            #     evaluator returns when correctness fails.
            #   - Otherwise a positive finite number (~1.0 for an
            #     unchanged baseline program) means the eval ran.
            if score_f is None or not math.isfinite(score_f):
                entry["status"] = "fail"
                entry["error"] = f"non-finite score: {score!r}"
                logger.warning(
                    "[FAIL ] %-28s  %.2fs  non-finite score %r",
                    task_name, elapsed, score,
                )
            elif score_f == 0.0:
                entry["status"] = "fail"
                entry["error"] = "score=0.0 (evaluator error or correctness failure)"
                logger.warning(
                    "[FAIL ] %-28s  %.2fs  score=0.0 (bridge sentinel)",
                    task_name, elapsed,
                )
            else:
                entry["status"] = "ok"
                logger.info(
                    "[OK   ] %-28s  %.2fs  speedup_score=%.6g",
                    task_name, elapsed, score_f,
                )
        finally:
            if mod_name is not None:
                sys.modules.pop(mod_name, None)

        results.append(entry)

    total_elapsed = time.monotonic() - t_start
    total = len(results)
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_fail = sum(1 for r in results if r["status"] == "fail")
    n_err = sum(1 for r in results if r["status"] == "error")

    logger.info("")
    logger.info("=" * 70)
    logger.info("summary:  total=%d  ok=%d  fail=%d  error=%d  wall=%.1fs",
                total, n_ok, n_fail, n_err, total_elapsed)
    logger.info("per-task timings:")
    for r in sorted(results, key=lambda r: (r["elapsed_s"] or 0.0), reverse=True):
        logger.info("  %-28s  %-5s  %7.2fs  score=%s",
                    r["task"], r["status"], r["elapsed_s"] or 0.0,
                    "n/a" if r["score"] is None else f"{r['score']:.6g}")

    summary = {
        "timestamp_utc": ts,
        "log_file": str(log_path.relative_to(REPO_ROOT)),
        "tasks_filter": tasks_filter,
        "elapsed_s": round(total_elapsed, 3),
        "totals": {"total": total, "ok": n_ok, "fail": n_fail, "error": n_err},
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("summary json: %s", summary_path)

    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument(
        "--tasks", nargs="*", default=None,
        help="restrict to these task names (default: all task folders)",
    )
    args = p.parse_args()
    summary = run(tasks_filter=args.tasks)
    # Non-zero exit iff any hard error (import / exception). A "fail"
    # (score==0 / non-finite) is also worth surfacing in CI, so
    # treat both error and fail as non-zero.
    totals = summary["totals"]
    return 1 if (totals["error"] or totals["fail"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
