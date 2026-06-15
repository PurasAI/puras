"""Local prompt optimizer — `puras optimize --local` (open-core).

Runs the shared `optimizer_core` search entirely offline on a BYO key. The seam:
a `LocalScoringBackend` whose `score()` writes the candidate's prompt/model/routing
as an overlay and calls the SAME `eval_local.run_eval_local` the hosted suite uses —
so a candidate is scored against the same agent loop + dataset + graders, just with
a swapped prompt. No Postgres, no platform API, no billing.

`run_optimize_local()` is the programmatic entry; the CLI's `puras optimize --local`
calls it. Returns the baseline, the winning candidate, every scored candidate, the
per-round log, and a diff artifact (proposed SKILL.md + skill.yaml patch) for the
user to apply with `puras deploy` — nothing is auto-deployed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from .eval_local import parse_cases_jsonl
from .local_run import LocalRunError, _pick_skill, _prepare_env
from .optimizer_core import (
    Candidate,
    ProposerInput,
    RoundLog,
    ScoreResult,
    SearchConfig,
    propose_candidates,
    run_search,
)


def _aggregate_per_grader(cases: list[dict]) -> list[dict]:
    """Fold a report's case rows into a per-grader breakdown (name, kind, mean_score,
    pass_rate, runs) — the same shape the hosted suite report carries, so the proposer
    sees which grader is failing."""
    acc: dict[str, dict] = {}
    for row in cases:
        for g in row.get("graders") or []:
            if g.get("skipped"):
                continue
            name = g.get("name") or "?"
            a = acc.setdefault(
                name,
                {"name": name, "kind": g.get("kind"), "_score": 0.0, "_pass": 0, "runs": 0},
            )
            a["_score"] += float(g.get("score") or 0.0)
            a["_pass"] += 1 if g.get("passed") else 0
            a["runs"] += 1
    out = []
    for a in acc.values():
        runs = a["runs"] or 1
        out.append(
            {
                "name": a["name"],
                "kind": a["kind"],
                "mean_score": round(a["_score"] / runs, 3),
                "pass_rate": round(a["_pass"] / runs, 3),
                "runs": a["runs"],
            }
        )
    return out


def _report_to_score(report: dict) -> ScoreResult:
    """Map a `run_eval_local` report into the core's ScoreResult. Local runs are on
    a BYO key, so cost is not metered (mean_cost_micros = 0)."""
    return ScoreResult(
        pass_rate=float(report.get("pass_rate_pct") or 0.0) / 100.0,
        mean_score=float(report.get("mean_score") or 0.0),
        mean_cost_micros=0,
        per_grader=_aggregate_per_grader(report.get("cases") or []),
        n=int(report.get("total") or 0),
        error=None,
    )


class LocalScoringBackend:
    """Scores a candidate by running the skill's eval suite offline with the
    candidate's prompt/model/routing overlaid. Caches the full report per candidate
    so the optimizer can ground its proposer on the baseline's failing cases."""

    def __init__(
        self,
        skill_dir: str | Path,
        *,
        skill: str | None,
        api_key: str | None,
        drive_path: str | None,
    ):
        self.skill_dir = skill_dir
        self.skill = skill
        self.api_key = api_key
        self.drive_path = drive_path
        self.reports: dict[str, dict] = {}

    async def score(
        self, candidate: Candidate, *, case_ids: list[str] | None, repeat: int
    ) -> ScoreResult:
        # run_eval_local owns an `asyncio.run`, so it must run in a worker thread to
        # get its own event loop (we're already inside run_search's loop).
        from .eval_local import run_eval_local

        try:
            report = await asyncio.to_thread(
                run_eval_local,
                self.skill_dir,
                skill=self.skill,
                api_key=self.api_key,
                drive_path=self.drive_path,
                case_ids=case_ids,
                repeat=repeat,
                overlay=candidate.as_overlay(),
            )
        except Exception as e:  # a broken candidate scores 0, never crashes the run
            return ScoreResult(error=f"{type(e).__name__}: {e}")
        self.reports[candidate.id] = report
        return _report_to_score(report)


def _load_skill_meta(skill_dir: str | Path, skill: str | None) -> dict:
    """Load the skill once to seed the optimizer: current SKILL.md body, model,
    routing, schemas, description, and the dataset's case ids."""
    from .deployment import ResolvedDeployment  # noqa: F401  (parity w/ eval_local)
    from .manifest import ManifestError, parse_bundle_dir
    from .skill_loader import load as load_skill

    root = Path(skill_dir).expanduser().resolve()
    if not root.is_dir():
        raise LocalRunError(f"bundle dir not found: {root}")
    try:
        manifest = parse_bundle_dir(root)
    except ManifestError as e:
        raise LocalRunError(f"invalid bundle: {e}") from e
    name = _pick_skill(manifest, skill)
    loaded = load_skill(manifest, root, name)
    if not loaded.is_agentic:
        raise LocalRunError(f"`{loaded.name}` is a deterministic skill — nothing to optimize")
    if not loaded.evals or not loaded.eval_dataset:
        raise LocalRunError(f"`{loaded.name}` declares no `evals.dataset` to optimize against")
    dataset = loaded.root / loaded.eval_dataset
    if not dataset.is_file():
        raise LocalRunError(f"eval dataset not found: {dataset}")
    cases = parse_cases_jsonl(dataset.read_text("utf-8"))
    decl = manifest.skill(name)
    return {
        "name": loaded.name,
        "description": getattr(decl, "description", "") or "",
        "input_schema": loaded.input_schema or {},
        "output_schema": loaded.output_schema,
        "system_prompt": loaded.system_prompt or "",
        "model": loaded.model,
        "routing": loaded.routing,
        "cases": cases,
        "case_ids": [c["id"] for c in cases],
    }


def _failing_cases(report: dict | None, cases: list[dict], limit: int = 3) -> list[dict]:
    """Lowest-scoring runs joined back to their dataset inputs + grader detail — the
    proposer's failure evidence."""
    if not report:
        return []
    rows = sorted(
        (report.get("cases") or []),
        key=lambda r: (r.get("score") if r.get("score") is not None else -1),
    )
    by_id = {c["id"]: c for c in cases}
    out = []
    for r in rows[:limit]:
        case = by_id.get(r.get("id"), {})
        out.append(
            {
                "id": r.get("id"),
                "inputs": case.get("inputs"),
                "score": r.get("score"),
                "error": r.get("error"),
                "graders": [
                    {"name": g.get("name"), "passed": g.get("passed"),
                     "score": g.get("score"), "detail": g.get("detail") or g.get("error")}
                    for g in (r.get("graders") or [])
                ],
            }
        )
    return out


def _build_artifact(
    seed: Candidate, winner: Candidate, scored: dict[str, tuple[Candidate, ScoreResult]]
) -> dict:
    """The proposed change, presentation-only (applied manually via `puras deploy`)."""
    base = scored[seed.id][1]
    win = scored[winner.id][1]
    patch: dict[str, Any] = {}
    if winner.model is not None and winner.model != seed.model:
        patch["model"] = winner.model
    if winner.routing is not None:
        patch["routing"] = winner.routing
    return {
        "changed": winner.id != seed.id,
        "proposed_skill_md": winner.system_prompt,
        "skill_yaml_patch": patch,
        "rationale": winner.rationale,
        "deltas": {
            "mean_score": round(win.mean_score - base.mean_score, 2),
            "pass_rate": round(win.pass_rate - base.pass_rate, 3),
        },
    }


def run_optimize_local(
    skill_dir: str | Path,
    *,
    skill: str | None = None,
    api_key: str | None = None,
    drive_path: str | None = None,
    proposer_model: str = "claude/opus-4-8",
    max_candidates: int = 8,
    max_rounds: int = 3,
    minibatch_size: int = 5,
    repeat: int = 1,
    improvement_threshold: float = 0.0,
    on_event: Callable[[str, dict], None] | None = None,
) -> dict[str, Any]:
    """Optimize a skill's prompt offline. Returns
    `{skill, baseline, winner, candidates, rounds, stop_reason, artifact}`."""
    from .llm_models import MODELS

    _prepare_env(api_key, drive_path)
    from .config import get_settings

    get_settings.cache_clear()

    meta = _load_skill_meta(skill_dir, skill)
    sink = on_event or (lambda _t, _p: None)

    seed = Candidate(
        system_prompt=meta["system_prompt"],
        model=None,
        routing=None,
        source="baseline",
        round_index=0,
    )
    backend = LocalScoringBackend(
        skill_dir, skill=meta["name"], api_key=api_key, drive_path=drive_path
    )
    config = SearchConfig(
        max_candidates=max_candidates,
        max_rounds=max_rounds,
        minibatch_size=minibatch_size,
        repeat=repeat,
        improvement_threshold=improvement_threshold,
    )
    allowed = sorted(MODELS.keys())

    async def _propose_fn(*, n: int, parent: Candidate, round_index: int) -> list[Candidate]:
        report = backend.reports.get(seed.id)
        base_score = _report_to_score(report) if report else ScoreResult()
        pin = ProposerInput(
            skill_name=meta["name"],
            skill_description=meta["description"],
            input_schema=meta["input_schema"],
            output_schema=meta["output_schema"],
            current_prompt=parent.system_prompt,
            current_model=parent.model or meta["model"],
            current_routing=parent.routing if parent.routing is not None else meta["routing"],
            baseline_pass_rate=base_score.pass_rate,
            baseline_mean_score=base_score.mean_score,
            per_grader=base_score.per_grader,
            failing_cases=_failing_cases(report, meta["cases"]),
            allowed_models=allowed,
        )
        sink("optimize_proposing", {"round": round_index})
        return await asyncio.to_thread(
            propose_candidates,
            pin,
            n=n,
            parent=parent,
            round_index=round_index,
            proposer_model=proposer_model,
        )

    def _on_round(log: RoundLog) -> None:
        sink("optimize_round", {"round": log.round_index, "improved": log.improved,
                                "survivors": len(log.survivors)})
        _print_round(log)

    result = asyncio.run(
        run_search(
            backend,
            seed=seed,
            propose_fn=_propose_fn,
            config=config,
            all_case_ids=meta["case_ids"],
            on_round=_on_round,
        )
    )

    def _cand_view(c: Candidate) -> dict:
        sr = result.scored[c.id][1]
        return {
            "id": c.id,
            "source": c.source,
            "round_index": c.round_index,
            "model": c.model,
            "routing": c.routing,
            "rationale": c.rationale,
            "mean_score": sr.mean_score,
            "pass_rate": sr.pass_rate,
            "error": sr.error,
        }

    artifact = _build_artifact(result.baseline, result.winner, result.scored)
    _print_summary(result, artifact)
    return {
        "skill": meta["name"],
        "baseline": _cand_view(result.baseline),
        "winner": _cand_view(result.winner),
        "candidates": [_cand_view(c) for c, _ in result.scored.values()],
        "rounds": [vars(r) for r in result.rounds],
        "stop_reason": result.stop_reason,
        "artifact": artifact,
    }


def _print_round(log: RoundLog) -> None:
    mark = "↑ improved" if log.improved else "= no gain"
    print(f"  round {log.round_index}: {log.proposed} proposed, "
          f"{len(log.survivors)} survived → {mark}")


def _print_summary(result, artifact: dict) -> None:
    base = result.scored[result.baseline.id][1]
    win = result.scored[result.winner.id][1]
    print("")
    print(f"baseline mean_score {base.mean_score} (pass_rate {base.pass_rate})")
    if artifact["changed"]:
        print(f"winner   mean_score {win.mean_score} (pass_rate {win.pass_rate}) "
              f"  Δscore {artifact['deltas']['mean_score']:+}")
        if artifact["skill_yaml_patch"]:
            print(f"  skill.yaml: {artifact['skill_yaml_patch']}")
        print("  → review the proposed SKILL.md and apply with `puras deploy`")
    else:
        print(f"no improvement found ({result.stop_reason}) — keeping the current prompt")
