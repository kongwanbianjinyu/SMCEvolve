"""Emit one Hydra config per materialized symbolic-regression task.

After `python data_api.py` has populated
`problems/symbolic_regression/generated/<split>/<equation_idx>/`,
run this script from the repo root to generate
`configs/problem/symreg_<split>_<equation_idx>.yaml` for every task:

    python problems/symbolic_regression/generate_smc_configs.py

Use `python -m smcevolve.main problem=<name>` to run a task.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent.parent
GENERATED = HERE / "generated"
CONFIGS = REPO_ROOT / "configs" / "problem"


def _config_body(name: str, rel_dir: str, timeout: float) -> str:
    return (
        f"name: {name}\n"
        f"dir: {rel_dir}\n"
        "initial_program: initial_program.py\n"
        "evaluator: evaluator.py\n"
        "task_file: task.md\n"
        f"timeout: {timeout}\n"
    )


def main(timeout: float = 30.0) -> int:
    if not GENERATED.is_dir():
        print(f"no generated tasks found at {GENERATED} — run `python {HERE}/data_api.py` first")
        return 1
    CONFIGS.mkdir(parents=True, exist_ok=True)
    count = 0
    for split_dir in sorted(GENERATED.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for task_dir in sorted(split_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            required = {"initial_program.py", "evaluator.py", "task.md"}
            have = {p.name for p in task_dir.iterdir()}
            if not required.issubset(have):
                print(f"skipping incomplete task: {task_dir}")
                continue
            eq_id = task_dir.name
            config_name = f"symreg_{split}_{eq_id}"
            rel_dir = os.path.relpath(task_dir, REPO_ROOT)
            out_path = CONFIGS / f"{config_name}.yaml"
            out_path.write_text(_config_body(config_name, rel_dir, timeout))
            count += 1
    print(f"wrote {count} symreg configs to {CONFIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
