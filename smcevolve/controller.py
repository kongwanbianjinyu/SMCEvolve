"""IslandController — orchestrates K SMCIslands in parallel and migrates.

Each epoch: every non-converged island runs one SMC step concurrently.
Every `migration_interval` epochs: a random derangement pairs islands
(each island is both exactly one source and exactly one destination).
Each src snapshots its top-K, then dst merges those migrants with its
own particles and keeps the top-N by reward (merge-and-truncate, so a
weaker migrant can never displace a stronger native).
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from .event_logger import EventLogger
from .island import SMCIsland
from .particle import Particle

log = logging.getLogger(__name__)


class IslandController:
    def __init__(
        self,
        islands: list[SMCIsland],
        migration_interval: int,
        migration_size: int,
        rng: random.Random | None = None,
        event_logger: EventLogger | None = None,
    ):
        self.islands = islands
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.rng = rng or random.Random()
        self.event_logger = event_logger
        self.epoch = 0

    async def initialize(self, initial_program: str) -> None:
        await asyncio.gather(
            *[island.initialize(initial_program) for island in self.islands]
        )

    async def run(self, max_iterations: int = 100) -> None:
        while not self._all_converged() and self.epoch < max_iterations:
            self.epoch += 1
            active = [isl for isl in self.islands if not isl.converged]
            await asyncio.gather(*[isl.step() for isl in active])

            if (
                len(self.islands) > 1
                and self.epoch % self.migration_interval == 0
            ):
                self._migrate()

        log.info("Run finished after %d epochs (converged=%s).", self.epoch, self._all_converged())

    def _all_converged(self) -> bool:
        return all(isl.converged for isl in self.islands)

    def _migrate(self) -> None:
        k = len(self.islands)
        # Snapshot first so the swap is atomic across islands.
        outgoing = [isl.top_k(self.migration_size) for isl in self.islands]
        perm = self._derangement(k)

        records: list[dict[str, Any]] = []
        for src in range(k):
            dst_idx = perm[src]
            dst = self.islands[dst_idx]
            migrants = [
                Particle(
                    program=p.program,
                    reward=p.reward,
                    parent_id=p.id,
                    metadata={"migrated_from": src},
                )
                for p in outgoing[src]
            ]
            migrant_ids = {id(m) for m in migrants}
            combined = dst.particles + migrants
            combined.sort(key=lambda p: p.reward, reverse=True)
            new_particles = combined[: dst.n_particles]
            adopted_slots = [
                i for i, p in enumerate(new_particles) if id(p) in migrant_ids
            ]
            dst.particles = new_particles
            records.append({
                "src": src,
                "dst": dst_idx,
                "offered": len(migrants),
                "adopted": len(adopted_slots),
                "slots": adopted_slots,
            })

        if self.event_logger is not None:
            self.event_logger.log({
                "type": "migration",
                "epoch": self.epoch,
                "k": k,
                "permutation": list(perm),
                "migration_size": self.migration_size,
                "records": records,
            })
        log.info(
            "[migration] epoch=%d derangement=%s adopted=%s",
            self.epoch, perm, [r["adopted"] for r in records],
        )

    def _derangement(self, k: int) -> list[int]:
        if k < 2:
            return list(range(k))
        while True:
            perm = list(range(k))
            self.rng.shuffle(perm)
            if all(perm[i] != i for i in range(k)):
                return perm

    def best_overall(self) -> Particle:
        return max(
            (isl.best_particle() for isl in self.islands),
            key=lambda p: p.reward,
        )

    def history(self) -> list[dict[str, Any]]:
        return [snap for isl in self.islands for snap in isl.history]
