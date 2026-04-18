"""SMCEvolve Prompt Templates — 2x2 Kernel Design.

Dimension 1: Edit granularity — Diff (local, small step) vs Rewrite (global, large step)
Dimension 2: Information source — No inspiration (single-particle) vs With inspiration (interacting)

Four kernels:
  K1: Diff    x No Inspiration   — local refinement
  K2: Diff    x With Inspiration — local refinement informed by other particles
  K3: Rewrite x No Inspiration   — global exploration
  K4: Rewrite x With Inspiration — global exploration informed by other particles

The `PromptManager` samples a kernel by configurable weights, builds the
system + user messages, and injects the top-K particles on the same island
(excluding the parent) as "inspirations" when the kernel asks for them.
Responses are parsed into a new program via `parse_response`, which handles
both SEARCH/REPLACE diff edits and full-rewrite fenced code blocks.
"""

from __future__ import annotations

import logging
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

log = logging.getLogger(__name__)


# =============================================================================
# Kernel 1: Diff x No Inspiration
# SMC interpretation: single-particle local kernel K_t(x_{t-1}, .)
# =============================================================================

DIFF_NO_INSPO_SYS = """
You are an expert programmer tasked with making targeted improvements to an existing program.
Focus on small, precise edits that improve performance — fix inefficiencies, tune parameters, optimize hot paths, or refine logic.

You MUST respond using an edit name, description, and the exact SEARCH/REPLACE diff format:

<NAME>
A shortened name summarizing the edit. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
Explain the targeted change you are making and why it should improve performance.
</DESCRIPTION>

<DIFF>
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE
</DIFF>

* Every SEARCH section must be copied verbatim from the current file, including indentation, whitespace, and comments — matching is byte-for-byte.
* Every SEARCH section must match EXACTLY ONE location in the current file. If the snippet you want to change appears more than once, add enough surrounding lines of context to make the match unique; ambiguous edits are rejected.
* SEARCH must differ from REPLACE (no-op edits are rejected).
* You can propose multiple independent edits. SEARCH/REPLACE blocks follow one after another with no other text between them; they are applied in order, so later blocks see the result of earlier ones.
* Do not repeat the fence markers (<<<<<<<, =======, >>>>>>>) inside SEARCH or REPLACE bodies.
* Make sure the file still runs after your changes.
* IMPORTANT: Do not rewrite the entire program — focus on targeted, surgical improvements.
""".strip()

DIFF_NO_INSPO_ITER = """# Current Program

```{{language}}
{{code_content}}
```

Performance metrics:
{{performance_metrics}}

# Task

Suggest targeted edits to improve the program's performance.
Focus on the most impactful small changes — parameter tuning, algorithmic micro-optimizations, or bug fixes.
Describe each change with a SEARCH/REPLACE block.
""".strip()


# =============================================================================
# Kernel 2: Diff x With Inspiration
# SMC interpretation: interacting local kernel K_t(x_{t-1}, . | {x^(i)})
# =============================================================================

DIFF_WITH_INSPO_SYS = """
You are an expert programmer tasked with making targeted improvements to an existing program.
You will be shown the current program AND one or more high-performing reference programs that solve the same problem.
Study the reference programs to identify specific techniques, parameter choices, or code patterns that could be surgically transplanted into the current program to improve it.

You MUST respond using an edit name, description, and the exact SEARCH/REPLACE diff format:

<NAME>
A shortened name summarizing the edit. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
Explain which specific technique or pattern you borrowed from the reference programs, how you adapted it, and why it should improve performance.
</DESCRIPTION>

<DIFF>
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE
</DIFF>

* Every SEARCH section must be copied verbatim from the current file, including indentation, whitespace, and comments — matching is byte-for-byte.
* Every SEARCH section must match EXACTLY ONE location in the current file. If the snippet you want to change appears more than once, add enough surrounding lines of context to make the match unique; ambiguous edits are rejected.
* SEARCH must differ from REPLACE (no-op edits are rejected).
* You can propose multiple independent edits. SEARCH/REPLACE blocks follow one after another with no other text between them; they are applied in order, so later blocks see the result of earlier ones.
* Do not repeat the fence markers (<<<<<<<, =======, >>>>>>>) inside SEARCH or REPLACE bodies.
* Make sure the file still runs after your changes.
* IMPORTANT: Do not rewrite the entire program — borrow and adapt specific ideas from the references via targeted edits.
""".strip()

DIFF_WITH_INSPO_ITER = """# Current Program

```{{language}}
{{code_content}}
```

Performance metrics:
{{performance_metrics}}

# Reference Programs

The following high-performing programs solve the same problem. Study their techniques and borrow specific ideas to improve the current program via targeted edits.

{{inspiration_section}}

# Task

Identify the most useful techniques from the reference programs and transplant them into the current program via small, targeted SEARCH/REPLACE edits.
Focus on adapting specific patterns — do not rewrite the entire program.
""".strip()


# =============================================================================
# Kernel 3: Rewrite x No Inspiration
# SMC interpretation: single-particle global kernel K_t(x_{t-1}, .)
# =============================================================================

REWRITE_NO_INSPO_SYS = """
You are an expert algorithm designer. Given a program and its performance, design a fundamentally improved or completely different algorithm to solve the same problem.
Do not just tweak the existing code — rethink the approach from scratch. Consider different data structures, algorithmic paradigms, mathematical formulations, or optimization strategies.

You MUST respond using a short summary name, description, and the full code:

<NAME>
A shortened name summarizing the code you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
Explain the new algorithmic approach you are taking, how it differs from the current one, and why it should perform better.
</DESCRIPTION>

<CODE>
```{{language}}
# The new program here.
```
</CODE>

* Your program must maintain the same inputs and outputs as the original.
* Think broadly — consider entirely different algorithms, not just parameter changes.
* Make sure the file still runs after your changes.
""".strip()

REWRITE_NO_INSPO_ITER = """# Current Program

```{{language}}
{{code_content}}
```

Performance metrics:
{{performance_metrics}}

# Task

Design a new algorithm to replace the current implementation.
Do not make incremental changes — propose a fundamentally different or significantly improved approach.
Your new program must maintain the same inputs and outputs.
""".strip()


# =============================================================================
# Kernel 4: Rewrite x With Inspiration
# SMC interpretation: interacting global kernel (crossover / recombination)
# =============================================================================

REWRITE_WITH_INSPO_SYS = """
You are an expert algorithm designer. You will be shown the current program AND one or more high-performing reference programs that solve the same problem.
Your task is to synthesize a new program that combines the best ideas from all provided programs, or uses them as inspiration to design something even better.
Think of this as intelligent crossover — not just copy-paste, but creative recombination of algorithmic insights.

You MUST respond using a short summary name, description, and the full code:

<NAME>
A shortened name summarizing the code you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
Explain which ideas you drew from each program, how you combined or extended them, and why the synthesis should outperform the originals.
</DESCRIPTION>

<CODE>
```{{language}}
# The new synthesized program here.
```
</CODE>

* Your program must maintain the same inputs and outputs as the original.
* Combine strengths from multiple programs — don't just pick one and copy it.
* Make sure the file still runs after your changes.
""".strip()

REWRITE_WITH_INSPO_ITER = """# Current Program

```{{language}}
{{code_content}}
```

Performance metrics:
{{performance_metrics}}

# Reference Programs

The following high-performing programs solve the same problem. Use them as inspiration to design a new, superior program that combines the best ideas.

{{inspiration_section}}

# Task

Synthesize a new program that combines the best elements from the current program and the reference programs.
Go beyond simple merging — creatively recombine algorithmic insights to produce something better than any individual program.
Your new program must maintain the same inputs and outputs.
""".strip()


# =============================================================================
# Kernel registry
# =============================================================================

KERNEL_CONFIGS: dict[str, dict[str, Any]] = {
    "diff_no_inspo": {
        "sys": DIFF_NO_INSPO_SYS,
        "iter": DIFF_NO_INSPO_ITER,
        "edit_mode": "diff",
        "needs_inspiration": False,
        "description": "Local refinement without reference (single-particle local kernel)",
    },
    "diff_with_inspo": {
        "sys": DIFF_WITH_INSPO_SYS,
        "iter": DIFF_WITH_INSPO_ITER,
        "edit_mode": "diff",
        "needs_inspiration": True,
        "description": "Local refinement with reference programs (interacting local kernel)",
    },
    "rewrite_no_inspo": {
        "sys": REWRITE_NO_INSPO_SYS,
        "iter": REWRITE_NO_INSPO_ITER,
        "edit_mode": "rewrite",
        "needs_inspiration": False,
        "description": "Global exploration without reference (single-particle global kernel)",
    },
    "rewrite_with_inspo": {
        "sys": REWRITE_WITH_INSPO_SYS,
        "iter": REWRITE_WITH_INSPO_ITER,
        "edit_mode": "rewrite",
        "needs_inspiration": True,
        "description": "Global exploration with reference programs (interacting global kernel)",
    },
}

KERNEL_NAMES: list[str] = list(KERNEL_CONFIGS.keys())


# =============================================================================
# PromptManager
# =============================================================================

@dataclass
class InspirationParticle:
    program: str
    reward: float
    source: str = "top_k"  # "top_k" or "diverse"


class PromptManager:
    """Samples a kernel, builds (system, user) prompts, and parses responses.

    Parameters
    ----------
    kernel_weights :
        Mapping from kernel name to sampling weight. Names must be in
        `KERNEL_NAMES`. Weights do not need to sum to 1 — they are
        normalized internally. Kernels with weight <= 0 are disabled.
        If None, all four kernels share equal weight.
    top_k_inspiration :
        Number of "best other particles" to show when an inspiration
        kernel is sampled. Particles are ranked by reward descending,
        the parent is excluded by id.
    diverse_inspirations :
        Number of additional inspirations selected for diversity via
        farthest-point sampling in embedding space (MAP-Elites style).
        Requires an ``embedder``. Set to 0 to disable.
    language :
        Code fence language used in the prompt (e.g. "python").
    force_kernel :
        If set, always use this kernel (ignores weights). Useful for
        ablations.
    embedder :
        Optional async embedding client used for diversity selection.
        Required when ``diverse_inspirations > 0``.
    rng :
        Random source for kernel sampling.
    """

    def __init__(
        self,
        kernel_weights: Mapping[str, float] | None = None,
        kernel_selection: str = "weighted",
        top_k_inspiration: int = 2,
        diverse_inspirations: int = 0,
        language: str = "python",
        force_kernel: str | None = None,
        embedder: Any | None = None,
        rng: random.Random | None = None,
    ):
        if top_k_inspiration < 0:
            raise ValueError(f"top_k_inspiration must be >= 0, got {top_k_inspiration}")
        if diverse_inspirations < 0:
            raise ValueError(f"diverse_inspirations must be >= 0, got {diverse_inspirations}")
        if diverse_inspirations > 0 and embedder is None:
            log.warning(
                "diverse_inspirations=%d but no embedder provided; "
                "diversity selection will be disabled",
                diverse_inspirations,
            )
        if kernel_selection not in ("weighted", "adaptive"):
            raise ValueError(
                f"kernel_selection must be 'weighted' or 'adaptive', got '{kernel_selection}'"
            )

        raw = dict(kernel_weights) if kernel_weights else {n: 1.0 for n in KERNEL_NAMES}
        for name in raw:
            if name not in KERNEL_CONFIGS:
                raise ValueError(
                    f"unknown kernel '{name}'; expected one of {KERNEL_NAMES}"
                )
        active = {n: float(w) for n, w in raw.items() if float(w) > 0}
        if not active:
            raise ValueError("at least one kernel must have a positive weight")

        if force_kernel is not None and force_kernel not in KERNEL_CONFIGS:
            raise ValueError(
                f"force_kernel='{force_kernel}' is not a valid kernel name"
            )

        self.weights = active
        self._kernel_names = list(active.keys())
        self._weight_values = list(active.values())
        self.kernel_selection = kernel_selection
        self.top_k_inspiration = top_k_inspiration
        self.diverse_inspirations = diverse_inspirations
        self.language = language
        self.force_kernel = force_kernel
        self._embedder = embedder
        self._rng = rng or random.Random()

        # Thompson Sampling state for adaptive mode.
        # Each kernel has Beta(alpha, beta); alpha counts successes,
        # beta counts failures.  Initialised at Beta(1,1) = uniform.
        self._ts: dict[str, dict[str, float]] = {
            n: {"alpha": 1.0, "beta": 1.0} for n in self._kernel_names
        }
        # Decay factor applied to all arms on every update so the
        # posterior stays responsive to recent performance shifts.
        self._ts_decay: float = 0.99

    # ------------------------------------------------------------------ public

    def select_kernel(self) -> str:
        if self.force_kernel is not None:
            return self.force_kernel
        if self.kernel_selection == "adaptive":
            return self._select_kernel_adaptive()
        return self._rng.choices(self._kernel_names, weights=self._weight_values, k=1)[0]

    def update_kernel(self, kernel_name: str, improved: bool) -> None:
        """Feed back proposal outcome to adaptive kernel selection."""
        if self.kernel_selection != "adaptive":
            return
        if kernel_name not in self._ts:
            return
        # Decay all arms so recent observations carry more weight
        for stats in self._ts.values():
            stats["alpha"] = max(1.0, stats["alpha"] * self._ts_decay)
            stats["beta"] = max(1.0, stats["beta"] * self._ts_decay)
        # Update the arm that was pulled
        if improved:
            self._ts[kernel_name]["alpha"] += 1.0
        else:
            self._ts[kernel_name]["beta"] += 1.0

    def kernel_stats(self) -> dict[str, dict[str, float]]:
        """Return current Thompson Sampling parameters (for logging)."""
        return {
            name: {
                "alpha": s["alpha"],
                "beta": s["beta"],
                "mean": s["alpha"] / (s["alpha"] + s["beta"]),
            }
            for name, s in self._ts.items()
        }

    # ------------------------------------------------------------------ private

    def _select_kernel_adaptive(self) -> str:
        """Thompson Sampling: sample from each arm's Beta posterior, pick highest."""
        best_name = self._kernel_names[0]
        best_sample = -1.0
        for name in self._kernel_names:
            s = self._ts[name]
            sample = self._rng.betavariate(s["alpha"], s["beta"])
            if sample > best_sample:
                best_sample = sample
                best_name = name
        return best_name

    async def build(
        self,
        parent_program: str,
        parent_reward: float,
        task: str,
        island_particles: Sequence[tuple[str, str, float]] | None = None,
        parent_id: str | None = None,
        kernel_name: str | None = None,
        archive: Sequence[tuple[str, float]] | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        """Construct (system_prompt, user_prompt, info).

        `island_particles` is a sequence of (id, program, reward) triples for
        all particles currently on the island — used for top-k selection.

        `archive` is a sequence of (program, reward) pairs covering all
        programs ever evaluated on this island — used as the diversity
        pool for MAP-Elites style farthest-point selection.

        If an inspiration kernel is sampled but no valid inspirations are
        available (initialization step, k=0, island has only the parent),
        we fall back to the matching no-inspiration kernel so the call
        still produces a useful prompt.
        """
        if kernel_name is None:
            kernel_name = self.select_kernel()
        if kernel_name not in KERNEL_CONFIGS:
            raise ValueError(f"unknown kernel '{kernel_name}'")

        cfg = KERNEL_CONFIGS[kernel_name]
        inspirations: list[InspirationParticle] = []
        fallback = False
        if cfg["needs_inspiration"]:
            inspirations = await self._select_inspirations(
                island_particles, parent_id, parent_program, archive,
            )
            if not inspirations:
                alt = _no_inspo_counterpart(kernel_name)
                if alt is not None:
                    log.debug(
                        "kernel %s has no inspirations available; falling back to %s",
                        kernel_name,
                        alt,
                    )
                    kernel_name = alt
                    cfg = KERNEL_CONFIGS[alt]
                    fallback = True

        sys_prompt = cfg["sys"].replace("{{language}}", self.language)
        user_tmpl = cfg["iter"]
        perf_text = _format_reward(parent_reward)
        inspo_block = (
            _format_inspirations(inspirations, self.language) if inspirations else ""
        )

        user_prompt = (
            user_tmpl
            .replace("{{language}}", self.language)
            .replace("{{code_content}}", parent_program)
            .replace("{{performance_metrics}}", perf_text)
            .replace("{{text_feedback_section}}", "")
            .replace("{{inspiration_section}}", inspo_block)
        )
        user_prompt = f"# Task Description\n\n{task.strip()}\n\n{user_prompt}"

        n_diverse = sum(1 for p in inspirations if p.source == "diverse")
        info = {
            "kernel": kernel_name,
            "edit_mode": cfg["edit_mode"],
            "needs_inspiration": cfg["needs_inspiration"],
            "n_inspirations": len(inspirations),
            "n_diverse_inspirations": n_diverse,
            "inspiration_rewards": [p.reward for p in inspirations],
            "fallback_from_inspo": fallback,
        }
        return sys_prompt, user_prompt, info

    # ------------------------------------------------------------------ helpers

    async def _select_inspirations(
        self,
        island_particles: Sequence[tuple[str, str, float]] | None,
        parent_id: str | None,
        parent_program: str | None,
        archive: Sequence[tuple[str, float]] | None = None,
    ) -> list[InspirationParticle]:
        total_needed = self.top_k_inspiration + self.diverse_inspirations
        if total_needed == 0:
            return []

        # ---- top-k from live island particles ----
        live_pool: list[InspirationParticle] = []
        if island_particles:
            for pid, prog, reward in island_particles:
                if parent_id is not None and pid == parent_id:
                    continue
                if parent_program is not None and prog == parent_program:
                    continue
                if prog is None:
                    continue
                try:
                    r = float(reward)
                except (TypeError, ValueError):
                    continue
                if math.isnan(r) or math.isinf(r):
                    continue
                live_pool.append(InspirationParticle(program=prog, reward=r))
        live_pool.sort(key=lambda p: p.reward, reverse=True)

        top_k = live_pool[: self.top_k_inspiration]
        for p in top_k:
            p.source = "top_k"

        n_diverse = self.diverse_inspirations
        if n_diverse == 0 or self._embedder is None or not archive:
            return top_k

        # ---- diverse-m from full archive via farthest-point ----
        top_k_programs = {p.program for p in top_k}
        diverse_pool: list[InspirationParticle] = []
        for prog, reward in archive:
            if parent_program is not None and prog == parent_program:
                continue
            if prog in top_k_programs:
                continue
            try:
                r = float(reward)
            except (TypeError, ValueError):
                continue
            if math.isnan(r) or math.isinf(r):
                continue
            diverse_pool.append(InspirationParticle(program=prog, reward=r, source="diverse"))
        if not diverse_pool:
            return top_k

        all_candidates = top_k + diverse_pool
        programs = [p.program for p in all_candidates]
        embeddings = await self._embedder.embed(programs)

        selected_idx = list(range(len(top_k)))
        remaining_idx = list(range(len(top_k), len(all_candidates)))
        selected_idx = _farthest_point_select(
            embeddings, selected_idx, remaining_idx, n_diverse,
        )

        result = [all_candidates[i] for i in selected_idx]
        # Sort: top-k first (by reward desc), then diverse (by reward desc)
        result.sort(key=lambda p: (p.source != "top_k", -p.reward))
        return result


# =============================================================================
# Response parsing
# =============================================================================

_CODE_BLOCK_RE = re.compile(r"```(?:[\w+.-]*)\s*\n(.*?)```", re.DOTALL)
_DIFF_RE = re.compile(
    r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE",
    re.DOTALL,
)


def parse_response(
    raw: str,
    edit_mode: str,
    parent_program: str,
) -> tuple[str, list[str]]:
    """Convert an LLM response into a new program.

    Returns `(program, issues)`. `issues` lists non-fatal parse problems
    and is empty on a clean parse. If parsing fails entirely, the parent
    program is returned unchanged — the caller (MH step) treats
    `new_program == parent` as a skipped proposal.

    Diff mode uses byte-for-byte SEARCH/REPLACE matching with a uniqueness
    requirement: each SEARCH block must occur exactly once in the current
    program state (after any prior blocks in the same response have been
    applied). See `apply_diff_edits` for details.
    """
    issues: list[str] = []
    if raw is None:
        issues.append("empty_response")
        return parent_program, issues

    if edit_mode == "diff":
        edits = _DIFF_RE.findall(raw)
        if not edits:
            issues.append("no_diff_blocks")
            return parent_program, issues
        program, apply_issues = apply_diff_edits(parent_program, edits)
        issues.extend(apply_issues)
        return program, issues

    # rewrite mode: the spec asks for a fenced code block inside <CODE>.
    # Be lenient: accept any fenced block in the response.
    m = _CODE_BLOCK_RE.search(raw)
    if m:
        return m.group(1).strip("\n"), issues
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").lstrip("python").lstrip("\n")
        return stripped.rstrip("`").rstrip(), issues
    issues.append("no_code_block")
    return parent_program, issues


def apply_diff_edits(
    parent_program: str,
    edits: Sequence[tuple[str, str]],
) -> tuple[str, list[str]]:
    """Apply SEARCH/REPLACE edits sequentially with exact, unique matching.

    For each (search, replace) block:

    * Empty SEARCH bodies are rejected (`empty_search`).
    * A SEARCH equal to its REPLACE is rejected as a no-op (`noop_edit`).
    * The SEARCH must match byte-for-byte as a substring of the current
      program state. Zero matches -> `search_not_found`; more than one
      match -> `search_ambiguous(count=N)`. In both cases the edit is
      skipped; the model must include enough context to disambiguate.
    * Edits are applied in order, so later blocks see the result of
      earlier ones. If a later block becomes ambiguous or missing because
      of an earlier edit, it is skipped with the same diagnostics.

    If no edits land, the parent program is returned unchanged so the
    caller can treat the proposal as a no-op.
    """
    issues: list[str] = []
    program = parent_program
    applied = 0
    for i, (search, replace) in enumerate(edits):
        tag = f"edit_{i}"
        if search == "":
            issues.append(f"{tag}:empty_search")
            continue
        if search == replace:
            issues.append(f"{tag}:noop_edit")
            continue
        count = program.count(search)
        if count == 0:
            issues.append(f"{tag}:search_not_found")
            continue
        if count > 1:
            issues.append(f"{tag}:search_ambiguous(count={count})")
            continue
        program = program.replace(search, replace, 1)
        applied += 1
    if applied == 0:
        return parent_program, issues
    issues.append(f"applied={applied}/{len(edits)}")
    return program, issues


# =============================================================================
# Farthest-point diversity selection
# =============================================================================

def _farthest_point_select(
    embeddings: list,
    selected: list[int],
    remaining: list[int],
    m: int,
) -> list[int]:
    """Greedily pick *m* points from *remaining* that maximise min-distance
    to the already-*selected* set (greedy k-center / farthest-point sampling).

    Returns the full selected index list (original + newly added).
    """
    import numpy as np

    selected = list(selected)
    remaining = list(remaining)

    # If nothing is selected yet, seed with the first remaining candidate
    # (pool is pre-sorted by reward desc, so this is the best remaining).
    if not selected and remaining:
        selected.append(remaining.pop(0))
        m -= 1

    for _ in range(min(m, len(remaining))):
        best_idx: int | None = None
        best_min_dist = -1.0
        for ri in remaining:
            min_dist = min(
                float(np.linalg.norm(embeddings[ri] - embeddings[si]))
                for si in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = ri
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


# =============================================================================
# Formatting helpers
# =============================================================================

def _format_reward(reward: Any) -> str:
    if reward is None:
        return "unknown (not yet evaluated)"
    try:
        r = float(reward)
    except (TypeError, ValueError):
        return str(reward)
    if math.isnan(r):
        return "unknown (not yet evaluated)"
    return f"reward = {r:.6f}"


def _format_inspirations(
    items: Sequence[InspirationParticle],
    language: str,
) -> str:
    lines: list[str] = []
    for i, p in enumerate(items, 1):
        tag = "top performer" if p.source == "top_k" else "diverse approach"
        lines.append(f"## Reference Program {i}  (reward = {p.reward:.6f}, {tag})")
        lines.append("")
        lines.append(f"```{language}")
        lines.append(p.program)
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip()


def _no_inspo_counterpart(kernel_name: str) -> str | None:
    mapping = {
        "diff_with_inspo": "diff_no_inspo",
        "rewrite_with_inspo": "rewrite_no_inspo",
    }
    return mapping.get(kernel_name)
