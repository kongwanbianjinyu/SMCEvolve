"""Subprocess-isolated evaluator.

Each call spawns `python _eval_runner.py <evaluator_path>` and pipes the
candidate program through stdin. Crashes, infinite loops, and resource
blowups in user code stay confined to the worker process.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_RUNNER = Path(__file__).parent / "_eval_runner.py"


class Evaluator:
    def __init__(self, evaluator_path: str | Path, timeout: float = 30.0):
        path = Path(evaluator_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"evaluator not found: {path}")
        self.evaluator_path = str(path)
        self.timeout = timeout

    async def evaluate(self, program: str) -> float:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(_RUNNER),
            self.evaluator_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(program.encode("utf-8")),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            log.warning("evaluator timed out after %.1fs", self.timeout)
            return 0.0

        if proc.returncode != 0:
            log.debug("evaluator exit=%d stderr=%s", proc.returncode, stderr.decode(errors="replace").strip())
            return 0.0

        line = stdout.decode(errors="replace").strip().splitlines()
        if not line:
            return 0.0
        try:
            return float(line[-1])
        except ValueError:
            log.debug("evaluator returned non-float: %r", line[-1])
            return 0.0
