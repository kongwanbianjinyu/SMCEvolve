"""Smoke test: verify every symbolic-regression evaluator runs end-to-end.

Walks `problems/symbolic_regression/generated/<split>/<eq_id>/`, dynamically
imports each task's `evaluator.py`, calls `evaluate(initial_program_source)`,
and records the returned score, elapsed time, and any exception.

Usage (from repo root):

    python problems/symbolic_regression/smoke_test.py
    python problems/symbolic_regression/smoke_test.py --splits phys_osc
    python problems/symbolic_regression/smoke_test.py --limit 3

Writes:
    problems/symbolic_regression/logs/smoke_<UTC-timestamp>.log   (detailed)
    problems/symbolic_regression/logs/smoke_<UTC-timestamp>.json  (summary)
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

# Sentinel the upstream OpenEvolve evaluator returns when something
# blew up before / during the BFGS fit (see openevolve_evaluator.py).
OE_ERROR_SENTINEL = -1e9

HERE = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent.parent
GENERATED = HERE / "generated"
LOGS_DIR = HERE / "logs"


_STALE_MODULE_PREFIXES = ("oe_candidate_", "_smoke_evaluator_")
_STALE_MODULE_NAMES = {"openevolve_evaluator", "initial_program"}


def _purge_task_modules() -> None:
    """Drop cached modules that would leak state between tasks.

    Each task's `evaluator.py` does `from openevolve_evaluator import
    evaluate` after prepending its own dir to sys.path. Without this
    purge, the first task's `openevolve_evaluator` (with its baked-in
    data paths) would be reused for every subsequent task.
    """
    stale = [
        name for name in list(sys.modules)
        if name in _STALE_MODULE_NAMES
        or name.startswith(_STALE_MODULE_PREFIXES)
    ]
    for name in stale:
        sys.modules.pop(name, None)


def _load_evaluator(evaluator_path: Path):
    """Import evaluator.py as a unique module and return its `evaluate`."""
    split = evaluator_path.parent.parent.name
    eq_id = evaluator_path.parent.name
    mod_name = f"_smoke_evaluator_{split}_{eq_id}"
    # Temporarily expose the task dir so `from openevolve_evaluator import
    # evaluate` resolves against THIS task (the evaluator.py prepends the
    # dir itself, but we also purge any cached module above first).
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


def _discover_tasks(splits: list[str] | None):
    """Yield (split, eq_id, task_dir) for every complete task folder."""
    if not GENERATED.is_dir():
        raise SystemExit(
            f"no generated tasks at {GENERATED} — "
            f"run `python {HERE}/data_api.py` first"
        )
    for split_dir in sorted(GENERATED.iterdir()):
        if not split_dir.is_dir():
            continue
        if splits and split_dir.name not in splits:
            continue
        for task_dir in sorted(split_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            required = {"initial_program.py", "evaluator.py"}
            have = {p.name for p in task_dir.iterdir()}
            if not required.issubset(have):
                continue
            yield split_dir.name, task_dir.name, task_dir


def run(splits: list[str] | None, limit: int | None) -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOGS_DIR / f"smoke_{ts}.log"
    summary_path = LOGS_DIR / f"smoke_{ts}.json"

    logger = logging.getLogger(f"symreg_smoke_{ts}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    logger.info("symbolic-regression smoke test")
    logger.info("log file: %s", log_path)
    logger.info("splits filter: %s", splits or "ALL")
    logger.info("limit per split: %s", limit or "none")

    results: list[dict] = []
    counts: dict[str, dict[str, int]] = {}
    per_split_seen: dict[str, int] = {}

    t_start = time.monotonic()
    for split, eq_id, task_dir in _discover_tasks(splits):
        per_split_seen.setdefault(split, 0)
        if limit is not None and per_split_seen[split] >= limit:
            continue
        per_split_seen[split] += 1

        counts.setdefault(split, {"ok": 0, "fail": 0, "error": 0})
        label = f"{split}/{eq_id}"
        prog_path = task_dir / "initial_program.py"
        eval_path = task_dir / "evaluator.py"
        program = prog_path.read_text(encoding="utf-8")

        entry: dict = {
            "split": split,
            "eq_id": eq_id,
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
            counts[split]["error"] += 1
            logger.error(
                "[ERROR] %-32s  %.2fs  %s",
                label, elapsed, entry["error"],
            )
            logger.error("traceback:\n%s", traceback.format_exc())
        else:
            elapsed = time.monotonic() - t0
            entry["elapsed_s"] = round(elapsed, 3)
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = None
            entry["score"] = score_f
            # Failure taxonomy:
            #   - non-numeric / nan / inf           -> fail (evaluator smoke broke)
            #   - exactly 0.0                       -> fail (wrap_openevolve sentinel)
            #   - <= OE_ERROR_SENTINEL / 2 (-5e8)   -> fail (upstream error sentinel)
            # Anything else (incl. negative combined_score for a bad
            # linear fit) counts as OK — the evaluator itself worked.
            if score_f is None or not math.isfinite(score_f):
                entry["status"] = "fail"
                entry["error"] = f"non-finite score: {score!r}"
                counts[split]["fail"] += 1
                logger.warning(
                    "[FAIL ] %-32s  %.2fs  non-finite score %r",
                    label, elapsed, score,
                )
            elif score_f == 0.0:
                entry["status"] = "fail"
                entry["error"] = "score=0.0 (wrap_openevolve_evaluator error sentinel)"
                counts[split]["fail"] += 1
                logger.warning(
                    "[FAIL ] %-32s  %.2fs  score=0.0 (bridge caught exception)",
                    label, elapsed,
                )
            elif score_f <= OE_ERROR_SENTINEL / 2.0:
                entry["status"] = "fail"
                entry["error"] = f"score={score_f} (OpenEvolve error sentinel -1e9)"
                counts[split]["fail"] += 1
                logger.warning(
                    "[FAIL ] %-32s  %.2fs  score=%.3g (upstream -1e9 sentinel)",
                    label, elapsed, score_f,
                )
            else:
                entry["status"] = "ok"
                counts[split]["ok"] += 1
                logger.info(
                    "[OK   ] %-32s  %.2fs  score=%.6g",
                    label, elapsed, score_f,
                )
        finally:
            if mod_name is not None:
                sys.modules.pop(mod_name, None)

        results.append(entry)

    total_elapsed = time.monotonic() - t_start
    total = len(results)
    total_ok = sum(c["ok"] for c in counts.values())
    total_fail = sum(c["fail"] for c in counts.values())
    total_err = sum(c["error"] for c in counts.values())

    logger.info("")
    logger.info("=" * 70)
    logger.info("summary:  total=%d  ok=%d  fail=%d  error=%d  (%.1fs)",
                total, total_ok, total_fail, total_err, total_elapsed)
    for split in sorted(counts):
        c = counts[split]
        logger.info("  %-16s  ok=%3d  fail=%3d  error=%3d",
                    split, c["ok"], c["fail"], c["error"])

    summary = {
        "timestamp_utc": ts,
        "log_file": str(log_path.relative_to(REPO_ROOT)),
        "splits_filter": splits,
        "limit_per_split": limit,
        "elapsed_s": round(total_elapsed, 3),
        "counts": counts,
        "totals": {
            "total": total, "ok": total_ok,
            "fail": total_fail, "error": total_err,
        },
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
        "--splits", nargs="*", default=None,
        help="restrict to these splits (e.g. phys_osc chem_react)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="max tasks per split (for a quick sanity run)",
    )
    args = p.parse_args()
    summary = run(splits=args.splits, limit=args.limit)
    # Non-zero exit iff any hard error (import / exception). A "fail"
    # status (score <= 0) is informational and does not fail CI.
    return 1 if summary["totals"]["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
