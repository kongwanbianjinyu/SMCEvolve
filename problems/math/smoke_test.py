"""Smoke test every math problem's evaluator on its initial_program.

For each subdirectory of ``problems/math``:
  1. Static import check of ``initial_program.py`` (catches missing
     jax/optax/etc. before we pay the subprocess cost).
  2. Invoke the SMCEvolve evaluator via
     ``smcevolve.evaluator.Evaluator`` (the same subprocess path used
     during a real run).
  3. Record reward + wall-clock time + status.

Statuses:
  - OK              : reward > 0.0 (evaluator returned a valid score)
  - ZERO            : reward == 0.0 (may mean the upstream seed is
                       trivially 0 — e.g. np.zeros(...) — or the
                       evaluator rejected the seed; see upstream
                       openevolve_evaluator.py for the rules)
  - NO_DEPS         : static import of initial_program failed with
                       ModuleNotFoundError (jax / optax / etc. missing)
  - IMPORT_ERROR    : static import failed for another reason
  - TIMEOUT         : evaluator exceeded the per-problem timeout
  - EXCEPTION(...)  : unexpected exception in the harness itself

Usage::

    source .venv/bin/activate
    python problems/math/smoke_test.py
    python problems/math/smoke_test.py --timeout 120 --only hexagon_packing_11,kissing_number

Log is streamed to stdout and written to problems/math/smoke_test.log.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smcevolve.evaluator import Evaluator  # noqa: E402


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def static_import_check(initial_path: Path) -> tuple[str, str | None]:
    """Load initial_program.py in-process to surface missing deps early.

    Returns (status, error_message). status is "OK", "NO_DEPS", or
    "IMPORT_ERROR".
    """
    spec = importlib.util.spec_from_file_location(
        f"smoke_seed_{initial_path.parent.name}", initial_path
    )
    if spec is None or spec.loader is None:
        return "IMPORT_ERROR", f"could not build import spec for {initial_path}"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        return "NO_DEPS", f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return "IMPORT_ERROR", f"{type(exc).__name__}: {exc}"
    return "OK", None


async def run_one(name: str, folder: Path, timeout: float) -> dict:
    evaluator_path = folder / "evaluator.py"
    initial_path = folder / "initial_program.py"

    if not evaluator_path.is_file():
        return {"name": name, "status": "MISSING_EVALUATOR", "reward": None, "elapsed": 0.0}
    if not initial_path.is_file():
        return {"name": name, "status": "MISSING_INITIAL", "reward": None, "elapsed": 0.0}

    # Static-check the seed and the upstream openevolve_evaluator so we
    # surface missing deps (jax, optax, sympy, scipy...) before paying
    # the subprocess cost.
    for label, path in (("seed", initial_path),
                        ("openevolve_evaluator",
                         folder / "openevolve_evaluator.py")):
        if not path.is_file():
            continue
        status, err = static_import_check(path)
        if status != "OK":
            return {
                "name": name,
                "status": status,
                "reward": None,
                "elapsed": 0.0,
                "error": f"[{label}] {err}",
            }

    program = initial_path.read_text(encoding="utf-8")
    ev = Evaluator(evaluator_path, timeout=timeout)
    t0 = time.perf_counter()
    try:
        reward = await ev.evaluate(program)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": name,
            "status": f"EXCEPTION({type(exc).__name__})",
            "reward": None,
            "elapsed": time.perf_counter() - t0,
            "error": str(exc),
        }
    elapsed = time.perf_counter() - t0

    # Evaluator.evaluate returns 0.0 on timeout as well as on genuine
    # zero scores. Infer a timeout if we came within 10% of the limit.
    if reward == 0.0 and elapsed >= 0.9 * timeout:
        status = "TIMEOUT"
    elif reward > 0.0:
        status = "OK"
    else:
        status = "ZERO"
    return {"name": name, "status": status, "reward": float(reward), "elapsed": elapsed}


async def main_async(timeout: float, only: set[str] | None) -> int:
    folders = sorted(p for p in HERE.iterdir() if p.is_dir())
    if only is not None:
        folders = [p for p in folders if p.name in only]
    logging.info(
        "smoke test: %d problem(s), per-problem timeout = %.0fs, cwd = %s",
        len(folders), timeout, Path.cwd(),
    )

    results: list[dict] = []
    for folder in folders:
        name = folder.name
        logging.info("--- %s ---", name)
        result = await run_one(name, folder, timeout)
        results.append(result)
        reward_str = (
            f"{result['reward']:.6f}" if result["reward"] is not None else "n/a"
        )
        logging.info(
            "[%-14s] %-40s reward=%s  elapsed=%.2fs",
            result["status"], name, reward_str, result["elapsed"],
        )
        if result.get("error"):
            logging.info("    error: %s", result["error"])

    logging.info("=" * 72)
    logging.info("SUMMARY  (%d problems)", len(results))
    logging.info("=" * 72)
    by_status: dict[str, list[str]] = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r["name"])
    for status in sorted(by_status):
        names = by_status[status]
        logging.info("  %-14s %3d   %s", status, len(names), ", ".join(names))

    # Non-zero exit code iff any harness-level exception occurred.
    bad = any(r["status"].startswith("EXCEPTION") or r["status"] in {"MISSING_EVALUATOR", "MISSING_INITIAL"}
              for r in results)
    return 1 if bad else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--timeout", type=float, default=900.0,
        help="per-problem timeout in seconds (default: 900)",
    )
    p.add_argument(
        "--log", type=Path, default=HERE / "smoke_test.log",
        help="log file path (default: problems/math/smoke_test.log)",
    )
    p.add_argument(
        "--only", type=str, default=None,
        help="comma-separated list of problem folder names to run (default: all)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log)
    only = set(args.only.split(",")) if args.only else None
    return asyncio.run(main_async(args.timeout, only))


if __name__ == "__main__":
    sys.exit(main())
