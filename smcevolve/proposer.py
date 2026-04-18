"""LLM-style proposal kernel (Sec 3.1).

Proposer.propose draws x' ~ p0(· | x, q, C_t). Under Assumption 3.1
(local modifications), the MH acceptance reduces to min(1, exp(β_t·ΔR))
and we never need the proposal density itself.

The Proposal return type carries the raw prompt/response text alongside
the new program so the event logger can persist it.

Prompt construction (kernel selection, inspiration injection, response
parsing) is delegated to `PromptManager` in `smcevolve.prompts`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from .prompts import PromptManager, parse_response

log = logging.getLogger(__name__)


@dataclass
class Proposal:
    program: str
    prompt: str = ""
    response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    name: str
    weight: float
    input_price_per_mtok: float
    output_price_per_mtok: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_price_per_mtok / 1_000_000
            + completion_tokens * self.output_price_per_mtok / 1_000_000
        )


class Proposer(ABC):
    @abstractmethod
    async def propose(
        self,
        parent_program: str,
        task: str,
        context: dict[str, Any],
    ) -> Proposal:
        ...

    def update_kernel(self, kernel_name: str, improved: bool) -> None:
        """Feed back proposal outcome for adaptive kernel selection."""


class OpenAIProposer(Proposer):
    """Calls an OpenAI-compatible /chat/completions endpoint.

    Reads `OPENAI_API_KEY` and `API_BASE_URL` from the environment so the same
    code works against the OpenAI API or any LiteLLM-style proxy.

    Accepts a list of `ModelSpec` with per-model weights. Each call picks one
    model by weight and records the per-call USD cost in proposal metadata.
    A running total is maintained under `self.total_cost` / `self.usage_by_model`.
    """

    def __init__(
        self,
        models: Sequence[ModelSpec],
        prompt_manager: PromptManager,
        max_concurrency: int = 8,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        request_timeout: float = 120.0,
        rng: random.Random | None = None,
    ):
        from openai import AsyncOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("API_BASE_URL")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set (load .env first)")

        if not models:
            raise ValueError("OpenAIProposer requires at least one model")
        self.models = list(models)
        self._weights = [m.weight for m in self.models]
        if sum(self._weights) <= 0:
            raise ValueError("Model weights must sum to a positive number")

        self.prompt_manager = prompt_manager
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._rng = rng or random.Random()
        self._cost_lock = asyncio.Lock()
        self.total_cost: float = 0.0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.usage_by_model: dict[str, dict[str, float]] = {
            m.name: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
            for m in self.models
        }

    def _pick_model(self) -> ModelSpec:
        return self._rng.choices(self.models, weights=self._weights, k=1)[0]

    async def _record_cost(
        self,
        spec: ModelSpec,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
    ) -> None:
        async with self._cost_lock:
            self.total_cost += cost
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            bucket = self.usage_by_model[spec.name]
            bucket["calls"] += 1
            bucket["prompt_tokens"] += prompt_tokens
            bucket["completion_tokens"] += completion_tokens
            bucket["cost"] += cost

    async def propose(self, parent_program, task, context):
        parent_reward = context.get("parent_reward", float("nan"))
        parent_id = context.get("parent_id")
        island_particles = context.get("island_particles")
        archive = context.get("archive")
        forced_kernel = context.get("kernel")

        sys_prompt, user_prompt, prompt_info = await self.prompt_manager.build(
            parent_program=parent_program,
            parent_reward=parent_reward,
            task=task,
            island_particles=island_particles,
            parent_id=parent_id,
            kernel_name=forced_kernel,
            archive=archive,
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        logged_prompt = f"[SYSTEM]\n{sys_prompt}\n\n[USER]\n{user_prompt}"

        spec = self._pick_model()

        async with self._sem:
            try:
                resp = await self._client.chat.completions.create(
                    model=spec.name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.request_timeout,
                )
            except Exception as exc:
                log.warning("LLM call failed (model=%s): %s", spec.name, exc)
                return Proposal(
                    program=parent_program,
                    prompt=logged_prompt,
                    response=f"<error: {exc}>",
                    metadata={
                        "error": str(exc),
                        "model": spec.name,
                        **prompt_info,
                    },
                )

        raw = resp.choices[0].message.content or ""
        program, parse_issues = parse_response(
            raw, edit_mode=prompt_info["edit_mode"], parent_program=parent_program
        )

        usage = getattr(resp, "usage", None)
        usage_dict = usage.model_dump() if usage is not None else None
        prompt_tokens = int(usage_dict.get("prompt_tokens", 0)) if usage_dict else 0
        completion_tokens = int(usage_dict.get("completion_tokens", 0)) if usage_dict else 0
        cost = spec.cost(prompt_tokens, completion_tokens)
        await self._record_cost(spec, prompt_tokens, completion_tokens, cost)

        return Proposal(
            program=program,
            prompt=logged_prompt,
            response=raw,
            metadata={
                "model": spec.name,
                "finish_reason": resp.choices[0].finish_reason,
                "usage": usage_dict,
                "cost_usd": cost,
                "input_price_per_mtok": spec.input_price_per_mtok,
                "output_price_per_mtok": spec.output_price_per_mtok,
                "cumulative_cost_usd": self.total_cost,
                "parse_issues": parse_issues,
                **prompt_info,
            },
        )

    def update_kernel(self, kernel_name: str, improved: bool) -> None:
        self.prompt_manager.update_kernel(kernel_name, improved)

    def cost_summary(self) -> dict[str, Any]:
        return {
            "total_cost_usd": self.total_cost,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "by_model": self.usage_by_model,
        }
