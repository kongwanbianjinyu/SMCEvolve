"""Hydra entry point for SMCEvolve."""

from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from .controller import IslandController
from .embedder import Embedder
from .evaluator import Evaluator
from .event_logger import EventLogger
from .island import SMCIsland
from .prompts import KERNEL_NAMES, PromptManager
from .proposer import ModelSpec, OpenAIProposer, Proposer

log = logging.getLogger(__name__)


def _build_prompt_manager(
    cfg: DictConfig, rng: random.Random | None = None
) -> PromptManager:
    prompt_cfg = cfg.algo.get("prompt")
    kernel_weights: dict[str, float] | None = None
    kernel_selection = "weighted"
    top_k = 2
    diverse = 0
    language = "python"
    force_kernel: str | None = None
    embedder: Embedder | None = None
    if prompt_cfg is not None:
        raw_weights = prompt_cfg.get("kernel_weights")
        if raw_weights is not None:
            kernel_weights = {
                str(k): float(v) for k, v in OmegaConf.to_container(raw_weights).items()
            }
        kernel_selection = str(prompt_cfg.get("kernel_selection", kernel_selection))
        top_k = int(prompt_cfg.get("top_k_inspiration", top_k))
        diverse = int(prompt_cfg.get("diverse_inspirations", diverse))
        language = str(prompt_cfg.get("language", language))
        force_kernel = prompt_cfg.get("force_kernel")
        if force_kernel is not None:
            force_kernel = str(force_kernel)

        # Build embedder for diversity selection
        if diverse > 0:
            emb_model = str(prompt_cfg.get("embedding_model", "Titan Text Embeddings V2"))
            emb_base = prompt_cfg.get("embedding_base_url")
            if emb_base is not None:
                emb_base = str(emb_base)
            embedder = Embedder(model=emb_model, base_url=emb_base)
            log.info("Embedder: model=%s base_url=%s", emb_model, emb_base)

    log.info(
        "PromptManager: kernels=%s weights=%s selection=%s top_k=%d diverse=%d language=%s force=%s",
        KERNEL_NAMES,
        kernel_weights,
        kernel_selection,
        top_k,
        diverse,
        language,
        force_kernel,
    )
    return PromptManager(
        kernel_weights=kernel_weights,
        kernel_selection=kernel_selection,
        top_k_inspiration=top_k,
        diverse_inspirations=diverse,
        language=language,
        force_kernel=force_kernel,
        embedder=embedder,
        rng=rng,
    )


def _build_proposer(
    cfg: DictConfig,
    prompt_manager: PromptManager,
    rng: random.Random | None = None,
) -> Proposer:
    if cfg.llm.type == "openai":
        models_cfg = cfg.llm.get("models")
        if models_cfg is None:
            # Backwards compat with the single-model form.
            models = [
                ModelSpec(
                    name=cfg.llm.model,
                    weight=1.0,
                    input_price_per_mtok=float(cfg.llm.get("input_price_per_mtok", 0.0)),
                    output_price_per_mtok=float(cfg.llm.get("output_price_per_mtok", 0.0)),
                )
            ]
        else:
            models = [
                ModelSpec(
                    name=m["name"],
                    weight=float(m["weight"]),
                    input_price_per_mtok=float(m["input_price_per_mtok"]),
                    output_price_per_mtok=float(m["output_price_per_mtok"]),
                )
                for m in models_cfg
            ]
        return OpenAIProposer(
            models=models,
            prompt_manager=prompt_manager,
            max_concurrency=cfg.llm.max_concurrency,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            request_timeout=cfg.llm.request_timeout,
            rng=rng,
        )
    raise ValueError(f"Unknown llm.type={cfg.llm.type}")


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else Path(get_original_cwd()) / p


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv(_resolve(".env"))
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    asyncio.run(_run(cfg))


async def _run(cfg: DictConfig) -> None:
    problem_dir = _resolve(cfg.problem.dir)
    initial_program = (problem_dir / cfg.problem.initial_program).read_text()
    task_description = (problem_dir / cfg.problem.task_file).read_text()
    evaluator_path = problem_dir / cfg.problem.evaluator

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    event_logger = EventLogger(output_dir)
    log.info("Event log root: %s", output_dir)

    prompt_manager = _build_prompt_manager(
        cfg, rng=random.Random(cfg.seed + 2_000_003)
    )
    proposer = _build_proposer(
        cfg, prompt_manager=prompt_manager, rng=random.Random(cfg.seed + 999_999)
    )
    eval_timeout = float(cfg.problem.get("timeout", 30.0))

    islands: list[SMCIsland] = []
    for k in range(cfg.algo.n_islands):
        islands.append(
            SMCIsland(
                island_id=k,
                n_particles=cfg.algo.particles_per_island,
                beta_target=cfg.algo.beta,
                kappa=cfg.algo.kappa,
                n_proposals=cfg.algo.n_proposals,
                min_iterations=cfg.algo.min_iterations,
                proposer=proposer,
                evaluator=Evaluator(evaluator_path, timeout=eval_timeout),
                task_description=task_description,
                rng=random.Random(cfg.seed + k),
                event_logger=event_logger,
            )
        )

    controller = IslandController(
        islands=islands,
        migration_interval=cfg.algo.migration_interval,
        migration_size=cfg.algo.migration_size,
        rng=random.Random(cfg.seed + 100_000),
        event_logger=event_logger,
    )

    try:
        await controller.initialize(initial_program)
        await controller.run(max_iterations=cfg.algo.max_iterations)

        best = controller.best_overall()
        cost_summary = (
            proposer.cost_summary() if isinstance(proposer, OpenAIProposer) else {}
        )
        log.info("=" * 60)
        log.info("Best reward: %.6f", best.reward)
        log.info("Best program:\n%s", best.program)
        if cost_summary:
            log.info("-" * 60)
            log.info("Total LLM cost: $%.6f", cost_summary["total_cost_usd"])
            log.info(
                "Total tokens: prompt=%d, completion=%d",
                cost_summary["total_prompt_tokens"],
                cost_summary["total_completion_tokens"],
            )
            for name, bucket in cost_summary["by_model"].items():
                log.info(
                    "  %s: %d calls, prompt=%d, completion=%d, cost=$%.6f",
                    name,
                    bucket["calls"],
                    bucket["prompt_tokens"],
                    bucket["completion_tokens"],
                    bucket["cost"],
                )
        log.info("=" * 60)
        event_logger.log(
            {
                "type": "final",
                "best_reward": best.reward,
                "best_program": best.program,
                "epochs": controller.epoch,
                "llm_cost": cost_summary,
            }
        )
    finally:
        event_logger.close()


if __name__ == "__main__":
    main()
