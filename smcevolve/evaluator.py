"""Subprocess-isolated evaluator.

Each call spawns `python _eval_runner.py <evaluator_path>` and pipes the
candidate program through stdin. Crashes, infinite loops, and resource
blowups in user code stay confined to the worker process.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_RUNNER = Path(__file__).parent / "_eval_runner.py"


@dataclass
class EvalResult:
    reward: float
    elapsed: float
    timed_out: bool


class Evaluator:
    """Subprocess-isolated evaluator with optional GPU-pool concurrency.

    If `gpu_pool` is provided, each `evaluate_timed` call acquires one GPU id
    from an asyncio queue (waiting if none is free), spawns the subprocess
    with `CUDA_VISIBLE_DEVICES=<id>`, then releases the id. Total in-flight
    evaluations are therefore capped at `len(gpu_pool)`.
    """

    def __init__(
        self,
        evaluator_path: str | Path,
        timeout: float = 30.0,
        gpu_pool: list[int] | None = None,
    ):
        path = Path(evaluator_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"evaluator not found: {path}")
        self.evaluator_path = str(path)
        self.timeout = timeout
        self.gpu_pool: list[int] | None = list(gpu_pool) if gpu_pool else None
        self._gpu_queue: asyncio.Queue[int] | None = None

    def _ensure_gpu_queue(self) -> None:
        # Lazy init so the queue is bound to the running event loop.
        if self.gpu_pool and self._gpu_queue is None:
            q: asyncio.Queue[int] = asyncio.Queue()
            for g in self.gpu_pool:
                q.put_nowait(g)
            self._gpu_queue = q

    async def evaluate(self, program: str) -> float:
        result = await self.evaluate_timed(program)
        return result.reward

    async def evaluate_timed(self, program: str) -> EvalResult:
        gpu_id: int | None = None
        if self.gpu_pool:
            self._ensure_gpu_queue()
            assert self._gpu_queue is not None
            gpu_id = await self._gpu_queue.get()
        try:
            return await self._run(program, gpu_id)
        finally:
            if gpu_id is not None and self._gpu_queue is not None:
                self._gpu_queue.put_nowait(gpu_id)

    async def _run(self, program: str, gpu_id: int | None) -> EvalResult:
        env = None
        if gpu_id is not None:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        t0 = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(_RUNNER),
            self.evaluator_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(program.encode("utf-8")),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            elapsed = time.perf_counter() - t0
            log.warning("evaluator timed out after %.1fs", self.timeout)
            return EvalResult(reward=0.0, elapsed=elapsed, timed_out=True)

        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            log.debug("evaluator exit=%d stderr=%s", proc.returncode, stderr.decode(errors="replace").strip())
            return EvalResult(reward=0.0, elapsed=elapsed, timed_out=False)

        line = stdout.decode(errors="replace").strip().splitlines()
        if not line:
            return EvalResult(reward=0.0, elapsed=elapsed, timed_out=False)
        try:
            return EvalResult(reward=float(line[-1]), elapsed=elapsed, timed_out=False)
        except ValueError:
            log.debug("evaluator returned non-float: %r", line[-1])
            return EvalResult(reward=0.0, elapsed=elapsed, timed_out=False)
