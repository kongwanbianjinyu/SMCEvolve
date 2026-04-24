"""SMCIsland — one chain of N particles running Algorithm 1.

Each step:
  1. Adaptive temperature: solve ESS(λ) = κN by bisection (Sec 2.3).
  2. Softmax resampling at Δβ = (λ_t − λ_{t-1}) β (Eq. 12).
  3. Metropolis-Hastings mutation: for each resampled particle, run up to
     `n_proposals` LLM proposals. Each proposal is generated from the
     current best-so-far state of the chain and accepted with probability
     α_t(x, x') = min{1, exp(β_t · (R(x') − R(x)))}. If accepted, the
     chain advances; otherwise the best-so-far is kept.

If an `event_logger` is supplied, every proposal and every step
summary is appended to it as a JSON record.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
from typing import Any

from .evaluator import Evaluator
from .event_logger import EventLogger
from .particle import Particle
from .proposer import Proposer
from .temperature import ess, find_next_lambda

log = logging.getLogger(__name__)


class SMCIsland:
    def __init__(
        self,
        island_id: int,
        n_particles: int,
        beta_target: float,
        kappa: float,
        n_proposals: int,
        min_iterations: int,
        proposer: Proposer,
        evaluator: Evaluator,
        task_description: str,
        rng: random.Random | None = None,
        event_logger: EventLogger | None = None,
    ):
        if min_iterations < 1:
            raise ValueError(f"min_iterations must be >= 1, got {min_iterations}")
        self.id = island_id
        self.n_particles = n_particles
        self.beta_target = beta_target
        self.kappa = kappa
        self.n_proposals = n_proposals
        self.min_iterations = min_iterations
        self.max_delta_lambda = 1.0 / min_iterations
        self.proposer = proposer
        self.evaluator = evaluator
        self.task = task_description
        self.rng = rng or random.Random()
        self.event_logger = event_logger

        self.particles: list[Particle] = []
        self.lam: float = 0.0
        self.iteration: int = 0
        self.converged: bool = False
        self.history: list[dict[str, Any]] = []
        # Archive of all evaluated programs: hash -> (program, reward).
        # Used as the diversity pool for MAP-Elites style inspiration.
        self._archive: dict[str, tuple[str, float]] = {}

    # ---- archive -----------------------------------------------------------

    def _archive_add(self, program: str, reward: float) -> None:
        key = hashlib.sha256(program.encode("utf-8")).hexdigest()[:16]
        prev = self._archive.get(key)
        if prev is None or reward > prev[1]:
            self._archive[key] = (program, reward)

    def archive_snapshot(self) -> list[tuple[str, float]]:
        """Return a copy of the archive as [(program, reward), ...]."""
        return list(self._archive.values())

    # ---- lifecycle ------------------------------------------------------

    async def initialize(self, initial_program: str) -> None:
        self.particles = [Particle(program=initial_program) for _ in range(self.n_particles)]
        proposals = await asyncio.gather(
            *[
                self.proposer.propose(p.program, self.task, {"parent_reward": float("nan")})
                for p in self.particles
            ]
        )
        for p, prop in zip(self.particles, proposals):
            p.program = prop.program
        await self._evaluate_all()
        for p in self.particles:
            self._archive_add(p.program, p.reward)
        self._log_event(
            type="init",
            island=self.id,
            n_particles=self.n_particles,
            initial_rewards=[p.reward for p in self.particles],
        )

    async def _evaluate_all(self) -> None:
        rewards = await asyncio.gather(
            *[self.evaluator.evaluate(p.program) for p in self.particles]
        )
        for p, r in zip(self.particles, rewards):
            p.reward = r

    # ---- one SMC step ---------------------------------------------------

    async def step(self) -> None:
        if self.converged:
            return

        self.iteration += 1
        parent_rewards = [p.reward for p in self.particles]
        next_lam = find_next_lambda(
            parent_rewards,
            self.lam,
            self.beta_target,
            self.kappa,
            max_delta=self.max_delta_lambda,
        )
        delta_beta = (next_lam - self.lam) * self.beta_target
        beta_t = next_lam * self.beta_target
        ess_at_lambda = ess(parent_rewards, delta_beta)

        resampled, ancestors, weights = self._resample(delta_beta)

        parents_entries = []
        for p in self.particles:
            entry = {"id": p.id, "reward": p.reward}
            mfrom = p.metadata.get("migrated_from") if p.metadata else None
            if mfrom is not None:
                entry["migrated_from"] = mfrom
            parents_entries.append(entry)

        self._log_event(**{
            "type": "resample",
            "island": self.id,
            "iteration": self.iteration,
            "lambda": next_lam,
            "delta_beta": delta_beta,
            "beta_t": beta_t,
            "ess_at_lambda": ess_at_lambda,
            "parents": parents_entries,
            "weights": weights,
            "ancestors": ancestors,
            "resampled_ids": [p.id for p in resampled],
        })

        mutated = await asyncio.gather(
            *[
                self._mutate_mh(p, beta_t=beta_t, particle_idx=i)
                for i, p in enumerate(resampled)
            ]
        )
        self.particles = mutated
        self.lam = next_lam

        rewards = [p.reward for p in self.particles]
        kernel_stats = None
        if hasattr(self.proposer, "prompt_manager"):
            kernel_stats = self.proposer.prompt_manager.kernel_stats()
        snap = {
            "type": "step_summary",
            "island": self.id,
            "iteration": self.iteration,
            "lambda": self.lam,
            "delta_beta": delta_beta,
            "beta_t": beta_t,
            "ess_at_lambda": ess_at_lambda,
            "best_reward": max(rewards),
            "mean_reward": sum(rewards) / len(rewards),
            "rewards": rewards,
            "kernel_stats": kernel_stats,
        }
        self.history.append(snap)
        self._log_event(**snap)
        log.info(
            "[island %d] iter=%d λ=%.4f Δβ=%.4f β_t=%.4f ESS=%.2f best=%.4f mean=%.4f",
            self.id, self.iteration, self.lam, delta_beta, beta_t,
            ess_at_lambda, snap["best_reward"], snap["mean_reward"],
        )

        if self.lam >= 1.0:
            self.converged = True

    def _resample(
        self, delta_beta: float
    ) -> tuple[list[Particle], list[int], list[float]]:
        rewards = [p.reward for p in self.particles]
        log_w = [delta_beta * r for r in rewards]
        m = max(log_w)
        w = [math.exp(lw - m) for lw in log_w]
        z = sum(w)
        weights = [wi / z for wi in w]
        ancestors = self._systematic_resample(weights)
        resampled = [
            Particle(
                program=self.particles[i].program,
                reward=self.particles[i].reward,
                parent_id=self.particles[i].id,
            )
            for i in ancestors
        ]
        return resampled, ancestors, weights

    def _systematic_resample(self, weights: list[float]) -> list[int]:
        n = len(weights)
        u = self.rng.random() / n
        cumsum = 0.0
        indices: list[int] = []
        j = 0
        for i in range(n):
            cumsum += weights[i]
            while j < n and u + j / n < cumsum:
                indices.append(i)
                j += 1
        return indices

    async def _mutate_mh(self, particle: Particle, beta_t: float, particle_idx: int) -> Particle:
        """MH mutation: propose from best-so-far, accept with α = min(1, e^{β_t Δr})."""
        best = particle  # current state of the chain (also the proposal source)
        for k_step in range(self.n_proposals):
            island_snapshot = [
                (p.id, p.program, p.reward) for p in self.particles
            ]
            proposal = await self.proposer.propose(
                best.program,
                self.task,
                {
                    "parent_reward": best.reward,
                    "parent_id": best.id,
                    "island_particles": island_snapshot,
                    "archive": self.archive_snapshot(),
                },
            )
            new_program = proposal.program
            kernel_used = proposal.metadata.get("kernel")
            if new_program == best.program:
                self._log_event(
                    type="proposal",
                    island=self.id,
                    iteration=self.iteration,
                    particle_idx=particle_idx,
                    k_step=k_step,
                    beta_t=beta_t,
                    parent_id=best.id,
                    parent_reward=best.reward,
                    child_reward=best.reward,
                    accept_prob=0.0,
                    accepted=False,
                    skipped="proposal_unchanged",
                    prompt=proposal.prompt,
                    response=proposal.response,
                    program=new_program,
                    proposal_metadata=proposal.metadata,
                )
                if kernel_used:
                    self.proposer.update_kernel(kernel_used, False)
                continue

            eval_result = await self.evaluator.evaluate_timed(new_program)
            new_reward = eval_result.reward
            self._archive_add(new_program, new_reward)

            delta = new_reward - best.reward
            improved = delta > 0
            if delta >= 0:
                accept_prob = 1.0
            else:
                accept_prob = math.exp(beta_t * delta)
            accepted = self.rng.random() < accept_prob

            child = Particle(
                program=new_program,
                reward=new_reward,
                parent_id=best.id,
            )

            self._log_event(
                type="proposal",
                island=self.id,
                iteration=self.iteration,
                particle_idx=particle_idx,
                k_step=k_step,
                beta_t=beta_t,
                parent_id=best.id,
                parent_reward=best.reward,
                child_reward=new_reward,
                accept_prob=accept_prob,
                accepted=accepted,
                improved=improved,
                eval_time=eval_result.elapsed,
                eval_timed_out=eval_result.timed_out,
                prompt=proposal.prompt,
                response=proposal.response,
                program=new_program,
                proposal_metadata=proposal.metadata,
            )

            if kernel_used:
                self.proposer.update_kernel(kernel_used, improved)

            if accepted:
                best = child
        return best

    # ---- introspection --------------------------------------------------

    def _log_event(self, **record: Any) -> None:
        if self.event_logger is not None:
            self.event_logger.log(record)

    def best_particle(self) -> Particle:
        return max(self.particles, key=lambda p: p.reward)

    def top_k(self, k: int) -> list[Particle]:
        return sorted(self.particles, key=lambda p: p.reward, reverse=True)[:k]
