"""Prompt-optimizer core — pure, runtime-agnostic (open-core).

A MIPROv2-flavored optimizer for agentic skills. It does NOT depend on the DB,
the platform API, or a specific event loop owner — it talks only to a
`ScoringBackend` (the seam that hides "blocking local eval" vs "fan-out billed
eval suite"). This is the third seam above `RunContext` (execution) and the eval
suite (scoring), so the SAME proposal + search logic drives both local and cloud.

Two ideas are kept from MIPROv2:
  - **Grounded instruction proposal**: an LLM proposer rewrites the skill's
    SKILL.md (and may adjust model/routing) conditioned on the *baseline eval-suite
    report* (per-grader pass-rates) and the lowest-scoring cases' evidence — not
    blind. (`propose_candidates`.)
  - **A cheap→expensive evaluation ladder**: each round screens candidates on a
    mini-batch, prunes by successive halving, then confirms survivors on the full
    dataset. (`run_search`.)

Dropped (per product decision): few-shot demo optimization and a TPE surrogate
(both slot into this file later without touching the seams).

The only LLM dependency is the existing provider seam (`providers.make_provider`)
+ the public model registry (`llm_models`); no langchain/dspy, no new dependency.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from .llm_models import is_known_slug, resolve as resolve_model

# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One point in the search space: a (system_prompt, model, routing) triple.

    `model`/`routing` are None to inherit the skill's deployed value. The baseline
    (current prod prompt) is `source='baseline'`, `is_baseline=True`, round 0.
    """

    system_prompt: str
    model: str | None = None
    routing: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    round_index: int = 0
    source: str = "proposer"  # 'baseline' | 'proposer'
    rationale: str = ""

    @property
    def is_baseline(self) -> bool:
        return self.source == "baseline"

    def as_overlay(self) -> dict[str, Any]:
        """The `prompt_override` / local overlay dict this candidate injects. Only
        keys that change anything are emitted (so an unchanged dimension inherits
        the deployed value)."""
        ov: dict[str, Any] = {"system_prompt": self.system_prompt}
        if self.model is not None:
            ov["model"] = self.model
        if self.routing is not None:
            ov["routing"] = self.routing
        return ov


@dataclass
class ScoreResult:
    """A candidate's measured quality from an eval run (suite or local)."""

    pass_rate: float = 0.0  # 0..1
    mean_score: float = 0.0  # 0..100
    mean_cost_micros: int = 0
    per_grader: list[dict] = field(default_factory=list)
    n: int = 0  # number of (case × repeat) runs scored
    error: str | None = None

    @property
    def total_cost_micros(self) -> int:
        return int(self.mean_cost_micros) * max(0, int(self.n))


@dataclass
class SearchConfig:
    max_candidates: int = 8
    max_rounds: int = 3
    minibatch_size: int = 5
    repeat: int = 1
    keep_fraction: float = 0.5  # successive halving: keep this share of a round
    # A winner must beat the baseline mean_score by >= this (0..100) without
    # regressing pass_rate, else the run keeps the baseline ("no improvement").
    improvement_threshold: float = 0.0
    # Hard spend ceiling across the whole search (micros). None = no cap (local).
    budget_micros: int | None = None


@dataclass
class RoundLog:
    round_index: int
    proposed: int
    survivors: list[str]  # candidate ids kept past the mini-batch screen
    round_winner_id: str | None
    improved: bool
    note: str = ""


@dataclass
class SearchResult:
    baseline: Candidate
    winner: Candidate
    scored: dict[str, tuple[Candidate, ScoreResult]]
    rounds: list[RoundLog]
    spent_micros: int = 0
    stop_reason: str = "complete"  # 'complete' | 'no_improvement' | 'budget_exhausted'


class ScoringBackend(Protocol):
    """Scores one candidate. `case_ids=None` means the full dataset (the confirm
    pass); a subset is the cheap mini-batch screen."""

    async def score(
        self, candidate: Candidate, *, case_ids: list[str] | None, repeat: int
    ) -> ScoreResult: ...


# A proposer callable: given the round + the candidate to improve on + how many to
# emit, return fresh candidates. The default impl is `propose_candidates` (an LLM
# call); tests inject a deterministic fake so `run_search` runs with no LLM.
ProposeFn = Callable[..., Awaitable[list[Candidate]]]


# ---------------------------------------------------------------------------
# Pure ranking helpers (composed by both the local run_search and the cloud resolver)
# ---------------------------------------------------------------------------


def rank_key(sr: ScoreResult) -> tuple:
    """Higher is better. Primary: mean_score; tie-break: cheaper run. A scoring
    error sinks the candidate."""
    if sr.error:
        return (-1.0, 0)
    return (round(sr.mean_score, 3), -int(sr.mean_cost_micros))


def select_survivors(
    scored: list[tuple[Candidate, ScoreResult]], keep: int
) -> list[Candidate]:
    """Top-`keep` candidates by rank (successive halving prune)."""
    ranked = sorted(scored, key=lambda cs: rank_key(cs[1]), reverse=True)
    return [c for c, _ in ranked[: max(1, keep)]]


def beats_baseline(
    cand: ScoreResult, baseline: ScoreResult, improvement_threshold: float
) -> bool:
    """A candidate wins only if it clears the baseline mean_score by the threshold
    AND does not regress pass_rate — guards against a thin, noisy 'win'."""
    if cand.error:
        return False
    return (
        cand.mean_score >= baseline.mean_score + improvement_threshold
        and cand.pass_rate >= baseline.pass_rate
    )


def keep_count(n: int, fraction: float) -> int:
    return max(1, math.ceil(n * fraction))


def worst_case_runs(config: SearchConfig, n_cases: int) -> int:
    """Upper bound on (case × repeat) runs a cloud run can bill — used to gate the
    request before any spend. Every round: a mini-batch screen of all candidates
    plus a full-suite confirm of the survivors; plus the one baseline full suite."""
    mb = min(config.minibatch_size, n_cases)
    survivors = keep_count(config.max_candidates, config.keep_fraction)
    per_round = config.max_candidates * mb + survivors * n_cases * config.repeat
    return n_cases * config.repeat + config.max_rounds * per_round


# ---------------------------------------------------------------------------
# The proposer (one grounded LLM call)
# ---------------------------------------------------------------------------

_PROPOSER_SYSTEM = """\
You are a prompt optimizer for an agentic "skill". A skill is driven by a SYSTEM \
PROMPT (Markdown). The skill runs a tool-using agent loop and must finish by \
calling a `set_output` tool whose argument matches the skill's OUTPUT SCHEMA. Your \
job: propose improved system prompts (and, when useful, a different text model or \
a routing/escalation policy) that will score HIGHER on the skill's eval graders.

Hard rules:
- Preserve the skill's contract: keep any tool usage and the `set_output` step \
intact; never tell the agent to stop using tools or to skip producing the output.
- Keep the prompt self-contained and in the same language/voice as the original.
- Make TARGETED edits tied to the observed grader failures — do not rewrite \
wholesale for its own sake.
- `model` and `routing` are OPTIONAL. Only set `model` to a slug from \
ALLOWED_MODELS. `routing` (if set) is an object like \
{"escalate_to": "<slug>", "on": "schema_error", "after": 2}; set it to null to \
disable escalation. Omit both to inherit the skill's current settings.

Reply with ONLY a JSON object: {"candidates": [{"system_prompt": "...", \
"model": "<slug or null>"?, "routing": {...} or null?, "rationale": "one line"}]}. \
Return exactly N candidates."""


@dataclass
class ProposerInput:
    skill_name: str
    skill_description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    current_prompt: str
    current_model: str | None
    current_routing: dict[str, Any] | None
    # Baseline eval-suite telemetry (what to fix):
    baseline_pass_rate: float | None = None
    baseline_mean_score: float | None = None
    per_grader: list[dict] = field(default_factory=list)
    # Lowest-scoring cases: [{inputs, score, graders|error}] — the failure evidence.
    failing_cases: list[dict] = field(default_factory=list)
    allowed_models: list[str] = field(default_factory=list)


def _dump(v: Any, limit: int = 4000) -> str:
    try:
        s = json.dumps(v, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        s = str(v)
    return s if len(s) <= limit else s[:limit] + "\n…(truncated)"


def _parse_candidates_json(text: str) -> list[dict]:
    """Robustly pull the candidates array out of the model reply (tolerates code
    fences / prose around the JSON, like the rubric judge parser)."""
    t = text.strip()
    if "```" in t:  # strip a ```json … ``` fence if present
        parts = t.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") or p.startswith("["):
                t = p
                break
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        obj = json.loads(t[start : end + 1])
    except ValueError:
        return []
    cands = obj.get("candidates") if isinstance(obj, dict) else None
    return [c for c in cands if isinstance(c, dict)] if isinstance(cands, list) else []


def _coerce_candidate(
    raw: dict, *, parent: Candidate, round_index: int, allowed_models: set[str]
) -> Candidate | None:
    sp = raw.get("system_prompt")
    if not isinstance(sp, str) or not sp.strip():
        return None
    model = raw.get("model")
    # Only honor a model that's a known public slug; otherwise inherit (drop the
    # dimension) rather than fail — a hallucinated slug never breaks the run.
    if not (isinstance(model, str) and model in allowed_models and is_known_slug(model)):
        model = None
    routing = raw.get("routing")
    if routing is not None and not isinstance(routing, dict):
        routing = None
    return Candidate(
        system_prompt=sp,
        model=model,
        routing=routing,
        parent_id=parent.id,
        round_index=round_index,
        source="proposer",
        rationale=str(raw.get("rationale") or "")[:500],
    )


def propose_candidates(
    pin: ProposerInput,
    *,
    n: int,
    parent: Candidate,
    round_index: int,
    proposer_model: str = "claude/opus-4-8",
    provider_factory=None,
) -> list[Candidate]:
    """One grounded proposer call → up to `n` fresh candidates derived from `parent`.

    Blocking (the provider client is sync); async callers should run this in a
    thread. Returns [] on any provider/parse failure so the search degrades to "no
    improvement" instead of crashing the run.
    """
    if provider_factory is None:
        # Lazy import keeps the core pure: importing optimizer_core never pulls the
        # provider SDKs (anthropic/openai) — only an actual proposer call does.
        from .providers import make_provider as provider_factory
    allowed = set(pin.allowed_models) or set()
    user = (
        f"SKILL: {pin.skill_name}\n"
        f"DESCRIPTION: {pin.skill_description or '(none)'}\n\n"
        f"INPUT SCHEMA:\n{_dump(pin.input_schema)}\n\n"
        f"OUTPUT SCHEMA:\n{_dump(pin.output_schema)}\n\n"
        f"CURRENT MODEL: {pin.current_model or '(skill default)'}\n"
        f"CURRENT ROUTING: {_dump(pin.current_routing)}\n"
        f"ALLOWED_MODELS: {sorted(allowed)}\n\n"
        f"CURRENT SYSTEM PROMPT:\n-----\n{pin.current_prompt}\n-----\n\n"
        f"BASELINE EVAL — pass_rate={pin.baseline_pass_rate}, "
        f"mean_score={pin.baseline_mean_score}\n"
        f"PER-GRADER (lower scores = what to fix):\n{_dump(pin.per_grader)}\n\n"
        f"LOWEST-SCORING CASES (inputs + why they failed):\n"
        f"{_dump(pin.failing_cases)}\n\n"
        f"Propose exactly {n} improved candidate(s)."
    )
    try:
        info = resolve_model(proposer_model)
        provider = provider_factory(info.upstream_provider, info.upstream_id)
        resp = provider.messages_create(
            _PROPOSER_SYSTEM,
            [{"role": "user", "content": user}],
            None,
            8192,
            cache_messages=False,
        )
    except Exception:
        return []
    raw_list = _parse_candidates_json("\n".join(resp.text_blocks))
    out: list[Candidate] = []
    for raw in raw_list[:n]:
        c = _coerce_candidate(
            raw, parent=parent, round_index=round_index, allowed_models=allowed
        )
        if c is not None:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# The search (drives the LOCAL path + is unit-testable with a fake backend)
# ---------------------------------------------------------------------------


async def run_search(
    backend: ScoringBackend,
    *,
    seed: Candidate,
    propose_fn: ProposeFn,
    config: SearchConfig,
    all_case_ids: list[str] | None = None,
    on_round: Callable[[RoundLog], None] | None = None,
) -> SearchResult:
    """Successive-halving search, driven synchronously (used by the local optimizer
    and the unit tests). The cloud path does NOT call this — its resolver composes
    the same pure helpers (`propose_candidates`, `select_survivors`, `beats_baseline`)
    across ticks, with the DB as the state — but both share identical selection rules.

    `propose_fn(pin_round=..., n=..., parent=..., round_index=...)` returns the round's
    candidates; `backend.score` measures them; halving prunes; survivors are confirmed
    on the full dataset; the round winner must `beats_baseline` to continue.
    """
    scored: dict[str, tuple[Candidate, ScoreResult]] = {}
    rounds: list[RoundLog] = []
    spent = 0
    stop_reason = "complete"

    def _budget_left(extra: int) -> bool:
        if config.budget_micros is None:
            return True
        return spent + extra <= config.budget_micros

    # Baseline on the full dataset — the bar every candidate must clear.
    baseline_score = await backend.score(seed, case_ids=None, repeat=config.repeat)
    scored[seed.id] = (seed, baseline_score)
    spent += baseline_score.total_cost_micros

    best = seed
    best_score = baseline_score
    minibatch_ids = (
        all_case_ids[: config.minibatch_size]
        if all_case_ids is not None
        else None
    )

    for r in range(1, config.max_rounds + 1):
        if not _budget_left(0):
            stop_reason = "budget_exhausted"
            break
        cands = await propose_fn(
            n=config.max_candidates, parent=best, round_index=r
        )
        if not cands:
            stop_reason = "no_improvement" if r == 1 else stop_reason
            break

        # --- cheap mini-batch screen ---
        mb: list[tuple[Candidate, ScoreResult]] = []
        for c in cands:
            sr = await backend.score(c, case_ids=minibatch_ids, repeat=1)
            scored[c.id] = (c, sr)
            spent += sr.total_cost_micros
            mb.append((c, sr))
            if not _budget_left(0):
                break

        survivors = select_survivors(mb, keep_count(len(mb), config.keep_fraction))

        # --- full-suite confirm of survivors ---
        for c in survivors:
            if not _budget_left(0):
                stop_reason = "budget_exhausted"
                break
            sr = await backend.score(c, case_ids=None, repeat=config.repeat)
            scored[c.id] = (c, sr)
            spent += sr.total_cost_micros

        # Round winner = best survivor confirmed on the full dataset.
        confirmed = [(c, scored[c.id][1]) for c in survivors]
        confirmed.sort(key=lambda cs: rank_key(cs[1]), reverse=True)
        round_winner, round_winner_score = confirmed[0] if confirmed else (None, None)

        improved = bool(
            round_winner is not None
            and round_winner_score.mean_score > best_score.mean_score
            and round_winner_score.pass_rate >= baseline_score.pass_rate
        )
        log = RoundLog(
            round_index=r,
            proposed=len(cands),
            survivors=[c.id for c in survivors],
            round_winner_id=round_winner.id if round_winner else None,
            improved=improved,
        )
        if improved:
            best, best_score = round_winner, round_winner_score
        rounds.append(log)
        if on_round is not None:
            on_round(log)

        if stop_reason == "budget_exhausted":
            break
        if not improved:
            stop_reason = "no_improvement"
            break

    # Final winner: keep the baseline unless `best` clears the improvement threshold.
    winner = (
        best
        if best.id != seed.id
        and beats_baseline(best_score, baseline_score, config.improvement_threshold)
        else seed
    )
    return SearchResult(
        baseline=seed,
        winner=winner,
        scored=scored,
        rounds=rounds,
        spent_micros=spent,
        stop_reason=stop_reason,
    )
