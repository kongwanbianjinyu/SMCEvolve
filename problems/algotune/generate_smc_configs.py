"""Emit one Hydra config per materialized AlgoTune task.

After `python create_task.py --algotune-path /path/to/AlgoTune --task <name>`
has populated `problems/algotune/<name>/`, run:

    python problems/algotune/generate_smc_configs.py

to write `configs/problem/algotune_<name>.yaml` for every task folder.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent.parent
CONFIGS = REPO_ROOT / "configs" / "problem"

_SKIP = {"__pycache__", "bench"}


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
    CONFIGS.mkdir(parents=True, exist_ok=True)
    count = 0
    for task_dir in sorted(HERE.iterdir()):
        if not task_dir.is_dir() or task_dir.name in _SKIP:
            continue
        required = {"initial_program.py", "evaluator.py", "task.md"}
        have = {p.name for p in task_dir.iterdir()}
        if not required.issubset(have):
            continue
        name = f"algotune_{task_dir.name}"
        rel_dir = os.path.relpath(task_dir, REPO_ROOT)
        out_path = CONFIGS / f"{name}.yaml"
        out_path.write_text(_config_body(name, rel_dir, timeout))
        count += 1
    print(f"wrote {count} algotune configs to {CONFIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
