"""Run a skill's declared evals against a finished job's output and score it.

Evals are to a skill what unit tests are to code (see manifest.EvalDecl). After a
job succeeds, every grader the skill declared scores THAT run's output in [0,1];
the weighted mean (×100) becomes the job's `eval_score`, surfaced on the job
cards/tables. Two grader kinds:

  - "check"  — a deterministic Python grader, run in the same subprocess sandbox
    as a tool (function_runner). Called as `fn(inputs=..., output=...)` and must
    return `{score: 0..1, passed: bool, detail: str}`. The objective layer.
  - "rubric" — an LLM-as-judge grader. The criteria (+ anchored levels) and the
    run's inputs/output are handed to the skill's text model, which returns a
    JSON `{score: 0..1, reasoning}`. The qualitative layer.

This runs best-effort: any grader that errors scores 0 and is recorded with its
error, but never fails the job. A skill with no `evals:` produces no score
(returns (None, None)), so the feature is fully opt-in and backward-compatible.

Judge calls bill the workspace exactly like the agent's own model turns
(record_usage → jobs.cost_micros), so the eval cost is finalized with the run.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from uuid import UUID  # annotations only (PEP 563)
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .config import get_settings
from .function_runner import run_function
from .llm_models import resolve as resolve_model
from .pricing import with_margin
from .providers import make_provider
from .run_context import RunContext
from .schema_dialect import prune_extras, to_output_jsonschema
from .skill_loader import LoadedEval, LoadedSkill

log = structlog.get_logger()

# A rubric grader passes (the green/amber/red badge cutover) at or above this
# normalized score; checks define their own pass/fail. Display-only — the
# numeric eval_score is the weighted mean either way.
_RUBRIC_PASS = 0.7

_JUDGE_SYSTEM = (
    "You are a strict, fair evaluator (an autorater) scoring the OUTPUT of an AI "
    "skill against ONE criterion. You are given the run's inputs, the run's "
    "output, the criterion, and — when provided — anchored score levels. Judge "
    "ONLY the given criterion; ignore everything else. Be calibrated: reserve 1.0 "
    "for output that fully satisfies the criterion and 0.0 for output that "
    "ignores it. Reply with a SINGLE JSON object and nothing else: "
    '{"score": <number 0..1>, "reasoning": "<one or two sentences>"}.'
)


def _weighted_eval_score(rows: list[dict]) -> int | None:
    """Weighted mean (×100, rounded) of the grader rows' [0,1] scores. `skipped`
    graders carry no weight — dropped from both numerator and denominator so they
    don't drag the score toward 0. Returns None when every grader skipped (nothing
    to score)."""
    scored = [r for r in rows if not r.get("skipped")]
    if not scored:
        return None
    total_w = sum(r["weight"] for r in scored) or 1.0
    score01 = sum(r["score"] * r["weight"] for r in scored) / total_w
    return int(round(score01 * 100))


def _clamp01(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


async def run_evals(
    ctx: RunContext,
    skill: LoadedSkill,
    inputs: dict,
    output: Any,
    *,
    deployment_root: Path,
    workdir: Path,
    python_exe: str,
    secrets: dict[str, str] | None,
    model_slug: str,
    expected: Any = None,
    has_expected: bool = False,
) -> tuple[int | None, dict | None]:
    """Score a finished run. Returns (eval_score 0..100, results) or (None, None)
    when the skill declares no evals. Never raises — a grader failure is recorded
    in its row and scores 0.

    The eval's I/O (the `evals` timeline event, the judge-call usage) goes through
    `ctx` — so this runs identically hosted (DbRunContext → Postgres) and offline
    (LocalRunContext → console/tally, `puras eval --local`). The rubric judge uses
    `make_provider` directly, so it works offline on the user's BYO key too.

    `expected` (with `has_expected` set) is the eval-dataset case's expected
    output — used by `exact_match` graders (and available to future expected-aware
    graders). On a live run there is no case, so `has_expected` is False and any
    grader that needs it is recorded as `skipped` and excluded from the weighted
    mean (it neither helps nor hurts the live `eval_score`)."""
    graders = skill.evals or []
    if not skill.is_agentic or not graders:
        return None, None

    job_id = ctx.job_id
    workspace_id = ctx.workspace_id

    # Build the judge provider once, lazily, and only if a rubric grader exists.
    judge: dict[str, Any] | None = None
    if any(g.kind == "rubric" for g in graders):
        try:
            info = resolve_model(model_slug)
            family, _, variant = model_slug.partition("/")
            judge = {
                "provider": make_provider(info.upstream_provider, info.upstream_id),
                "family": family,
                "variant": variant,
            }
        except Exception:
            log.warning("eval_judge_unavailable", job_id=str(job_id), model=model_slug)

    rows: list[dict] = []
    for g in graders:
        if g.kind == "check":
            row = await _run_check(
                g, inputs, output,
                deployment_root=deployment_root, workdir=workdir,
                python_exe=python_exe, secrets=secrets,
                workspace_id=workspace_id, job_id=job_id,
            )
        elif g.kind == "rubric":
            row = await _run_rubric(ctx, g, inputs, output, judge)
        elif g.kind == "exact_match":
            row = _run_exact_match(g, output, expected, has_expected)
        else:  # schema
            row = _run_schema(g, output, skill.output_schema)
        rows.append(row)

    eval_score = _weighted_eval_score(rows)
    if eval_score is None:
        # Every grader skipped (e.g. exact_match on a live run with no `expected`,
        # schema with no schema) — there's nothing to score.
        return None, None

    results = {
        "score": eval_score,
        "model": judge["variant"] if judge else None,
        "graders": rows,
    }
    # Timeline breadcrumb so the run detail shows the score + per-grader breakdown.
    await ctx.emit_event("evals", {"score": eval_score, "graders": rows})
    return eval_score, results


async def _run_check(
    g: LoadedEval,
    inputs: dict,
    output: Any,
    *,
    deployment_root: Path,
    workdir: Path,
    python_exe: str,
    secrets: dict[str, str] | None,
    workspace_id: UUID,
    job_id: UUID,
) -> dict:
    """Run one deterministic grader in the subprocess sandbox. The grader is
    called as `fn(inputs=<job inputs>, output=<job output>)` and must return
    `{score, passed, detail}`."""
    base = {"name": g.name, "kind": "check", "weight": g.weight}
    if not g.module or not g.func:
        return {**base, "score": 0.0, "passed": False, "detail": "", "error": "unresolved entrypoint"}
    try:
        out = await asyncio.to_thread(
            run_function,
            f"{g.module}:{g.func}",
            {"inputs": inputs, "output": output},
            workdir,
            deployment_root,
            python_exe,
            secrets,
            str(workspace_id),
            str(job_id),
        )
    except Exception as e:  # subprocess plumbing failure — soft-fail the grader
        return {**base, "score": 0.0, "passed": False, "detail": "", "error": str(e)[:300]}

    if not out.get("ok"):
        return {
            **base, "score": 0.0, "passed": False, "detail": "",
            "error": str(out.get("error") or "grader failed")[:300],
        }
    res = out.get("result")
    if not isinstance(res, dict):
        return {**base, "score": 0.0, "passed": False, "detail": "",
                "error": "grader did not return an object"}
    score = _clamp01(res.get("score"))
    passed = bool(res.get("passed", score >= 0.999))
    return {**base, "score": score, "passed": passed,
            "detail": str(res.get("detail") or "")[:500]}


async def _run_rubric(
    ctx: RunContext,
    g: LoadedEval,
    inputs: dict,
    output: Any,
    judge: dict[str, Any] | None,
) -> dict:
    """Run one LLM-as-judge grader. The judge call is metered through `ctx` like
    any other model turn (billed hosted; tallied offline)."""
    base = {"name": g.name, "kind": "rubric", "weight": g.weight}
    if judge is None:
        return {**base, "score": 0.0, "passed": False, "detail": "",
                "error": "judge model unavailable"}

    levels = ""
    if g.levels:
        anchored = "\n".join(f"  {k}: {v}" for k, v in sorted(g.levels.items()))
        levels = f"\nAnchored score levels:\n{anchored}"
    user = (
        f"CRITERION:\n{g.criteria}{levels}\n\n"
        f"RUN INPUTS (JSON):\n{_dump(inputs)}\n\n"
        f"RUN OUTPUT (JSON):\n{_dump(output)}\n\n"
        "Score how well the OUTPUT satisfies the CRITERION. Reply with the JSON "
        "object only."
    )
    try:
        resp = await asyncio.to_thread(
            judge["provider"].messages_create,
            _JUDGE_SYSTEM,
            [{"role": "user", "content": user}],
            None,
            1024,
            cache_messages=False,
        )
    except Exception as e:
        return {**base, "score": 0.0, "passed": False, "detail": "",
                "error": str(e)[:300]}

    # Meter the judge call exactly like an agent model turn (billed hosted via
    # DbRunContext; tallied for info offline via LocalRunContext).
    try:
        billed = with_margin(resp.upstream_cost_micros)
        await ctx.record_usage(
            provider=judge["family"], model=judge["variant"],
            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            upstream_micros=resp.upstream_cost_micros, billed_micros=billed,
            meta={"eval": g.name},
        )
    except Exception:
        log.debug("eval_usage_record_failed", job_id=str(ctx.job_id), grader=g.name)

    parsed = _parse_judge_json("\n".join(resp.text_blocks))
    if parsed is None:
        return {**base, "score": 0.0, "passed": False, "detail": "",
                "error": "judge returned non-JSON"}
    score = _clamp01(parsed.get("score"))
    return {**base, "score": score, "passed": score >= _RUBRIC_PASS,
            "detail": str(parsed.get("reasoning") or "")[:500]}


_MISSING = object()


def _extract_path(value: Any, dotted: str) -> Any:
    """Walk a dotted path (`a.b.0.c`) into nested dicts/lists. Returns `_MISSING`
    if any segment doesn't resolve, so a missing field is distinguishable from a
    field whose value is literally None."""
    cur = value
    for seg in dotted.split("."):
        if isinstance(cur, dict):
            if seg not in cur:
                return _MISSING
            cur = cur[seg]
        elif isinstance(cur, list):
            try:
                idx = int(seg)
            except ValueError:
                return _MISSING
            if idx < 0 or idx >= len(cur):
                return _MISSING
            cur = cur[idx]
        else:
            return _MISSING
    return cur


def _canonical(v: Any) -> str:
    """Order-insensitive canonical JSON for deep equality (sorted keys)."""
    try:
        return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(v)


def _run_exact_match(
    g: LoadedEval, output: Any, expected: Any, has_expected: bool
) -> dict:
    """Deterministic, free grader: the output (or `g.field` of it) must deep-equal
    the case's `expected` (or that same `field` of the expected). Skipped when the
    run has no `expected` (a live run), so it never drags a live score down."""
    base = {"name": g.name, "kind": "exact_match", "weight": g.weight}
    if not has_expected:
        return {
            **base, "score": 0.0, "passed": False, "skipped": True,
            "detail": "no `expected` for this run (live run — exact_match only "
                      "scores in an eval suite)",
        }
    if g.field:
        got = _extract_path(output, g.field)
        want = _extract_path(expected, g.field)
        label = f"`{g.field}`"
    else:
        got, want, label = output, expected, "output"
    if want is _MISSING:
        return {**base, "score": 0.0, "passed": False,
                "detail": f"{label} missing from the case's `expected`"}
    if got is _MISSING:
        return {**base, "score": 0.0, "passed": False,
                "detail": f"{label} missing from the output"}
    equal = _canonical(got) == _canonical(want)
    if equal:
        return {**base, "score": 1.0, "passed": True, "detail": f"{label} matches"}
    return {
        **base, "score": 0.0, "passed": False,
        "detail": f"{label} mismatch — expected {_canonical(want)[:200]}, "
                  f"got {_canonical(got)[:200]}",
    }


def _run_schema(g: LoadedEval, output: Any, skill_output_schema: Any) -> dict:
    """Deterministic, free grader: the output must validate against a JSON Schema.
    Uses the grader's explicit `schema` (Puras dialect) when given, else the
    skill's own `output_schema`. Extra keys are pruned first (same leniency as the
    platform's post-run output validation)."""
    base = {"name": g.name, "kind": "schema", "weight": g.weight}
    schema = g.schema or skill_output_schema
    if not isinstance(schema, dict) or not schema:
        return {**base, "score": 0.0, "passed": False, "skipped": True,
                "detail": "no schema to validate against (grader has no `schema` "
                          "and the skill declares no output_schema)"}
    try:
        pruned = prune_extras(schema, output)
        Draft202012Validator(to_output_jsonschema(schema)).validate(pruned)
    except ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) or "<root>"
        return {**base, "score": 0.0, "passed": False,
                "detail": f"schema validation failed at `{path}`: {e.message[:300]}"}
    except Exception as e:  # malformed schema → soft-fail the grader, never the job
        return {**base, "score": 0.0, "passed": False, "skipped": True,
                "detail": f"schema grader could not run: {str(e)[:200]}"}
    return {**base, "score": 1.0, "passed": True, "detail": "output validates"}


def _dump(v: Any) -> str:
    """Compact JSON for the judge prompt, capped so a giant output can't blow the
    judge's context (the grader sees a representative head, not megabytes)."""
    try:
        s = json.dumps(v, ensure_ascii=False, default=str, indent=2)
    except (TypeError, ValueError):
        s = str(v)
    if len(s) > 12000:
        s = s[:12000] + "\n… [truncated]"
    return s


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(text: str) -> dict | None:
    """Pull the JSON verdict out of the judge's reply — tolerant of ```json
    fences or a stray sentence around the object."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None
