"""Offline eval suite — `puras eval --local` (open-core).

Runs a skill's declared eval suite (its `evals.dataset` cases + graders) entirely
on the local runner: each case is executed offline via the same agent loop as
`puras run --local`, then graded by the SAME `eval_runner.run_evals` the hosted
platform uses — routed through a LocalRunContext, so exact_match / schema / check
graders run free and a `rubric` (LLM-judge) grader runs on the user's BYO key.
No Postgres, no platform API.

`run_eval_local()` is the programmatic entry; the CLI's `puras eval --local`
calls it. Returns an aggregate report (pass-rate, mean score, per-case rows) and
a `gate_passed` flag for CI use with `--threshold`.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable

from .local_run import (
    _LOCAL_WORKSPACE_ID,
    LocalRunError,
    _pick_skill,
    _prepare_env,
)


def parse_cases_jsonl(text: str) -> list[dict]:
    """Parse an eval dataset (.jsonl): one JSON object per line, blank lines and
    `#` comments ignored. Each case needs an `inputs` object; `expected` (for
    exact_match graders), `id`, and `tags` are optional. `has_expected` records
    whether the case carried an `expected` key (None is a valid expected value)."""
    cases: list[dict] = []
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except ValueError as e:
            raise LocalRunError(f"dataset line {n}: invalid JSON ({e})") from None
        if not isinstance(obj, dict) or not isinstance(obj.get("inputs"), dict):
            raise LocalRunError(f"dataset line {n}: each case needs an `inputs` object")
        cases.append(
            {
                "id": str(obj.get("id") or f"case-{len(cases) + 1}"),
                "inputs": obj["inputs"],
                "expected": obj.get("expected"),
                "has_expected": "expected" in obj,
                "tags": obj.get("tags"),
            }
        )
    return cases


def run_eval_local(
    skill_dir: str | Path,
    *,
    skill: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    drive_path: str | None = None,
    case_ids: list[str] | None = None,
    repeat: int = 1,
    threshold: int | None = None,
    overlay: dict[str, Any] | None = None,
    on_event: Callable[[str, dict], None] | None = None,
) -> dict[str, Any]:
    """Run a skill's eval suite offline and return an aggregate report:
    `{total, passed, pass_rate_pct, mean_score, gate_passed, threshold, cases}`.

    A case `passed` when every non-skipped grader passed; `pass_rate_pct` is the
    share of (case × repeat) runs that passed. `gate_passed` is True unless a
    `threshold` is given and the pass-rate falls below it (the CI gate)."""
    root = Path(skill_dir).expanduser().resolve()
    if not root.is_dir():
        raise LocalRunError(f"bundle dir not found: {root}")

    _prepare_env(api_key, drive_path)
    from .config import get_settings

    get_settings.cache_clear()

    import asyncio

    from .agent_runner import run_agent
    from .deployment import ResolvedDeployment, build_skill_python
    from .drive import setup_drive
    from .eval_runner import run_evals
    from .manifest import ManifestError, parse_bundle_dir
    from .run_context import LocalRunContext
    from .skill_loader import apply_prompt_override, load as load_skill
    from .workdir import attach_skill, cleanup_workdir, create_workdir

    setup_drive()

    try:
        manifest = parse_bundle_dir(root)
    except ManifestError as e:
        raise LocalRunError(f"invalid bundle: {e}") from e
    deployment = ResolvedDeployment(root=root, manifest=manifest, deployment_id=None)
    loaded = load_skill(manifest, root, _pick_skill(manifest, skill))
    # Optimizer candidate injection (local mirror of jobs.prompt_override): swap the
    # candidate's prompt / model / routing onto the skill so a candidate is scored
    # offline against the same tool code + dataset. None = score the skill as-is.
    loaded = apply_prompt_override(loaded, overlay)

    if not loaded.is_agentic:
        raise LocalRunError(f"`{loaded.name}` is a deterministic skill — no agent eval")
    if not loaded.evals:
        raise LocalRunError(f"`{loaded.name}` declares no `evals:` graders")
    if not loaded.eval_dataset:
        raise LocalRunError(f"`{loaded.name}` declares no `evals.dataset` to run")
    dataset = loaded.root / loaded.eval_dataset
    if not dataset.is_file():
        raise LocalRunError(f"eval dataset not found: {dataset}")

    cases = parse_cases_jsonl(dataset.read_text("utf-8"))
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        raise LocalRunError("no eval cases to run (after --case filtering)")

    s = get_settings()
    model_slug = model or loaded.model or s.default_model_slug
    python_exe, venv_dir = build_skill_python(loaded.root)
    workspace_id = _LOCAL_WORKSPACE_ID
    reps = max(1, int(repeat or 1))
    # Per-event sink stays quiet by default (a suite is many runs); we print a
    # one-line verdict per case instead. A caller can pass on_event to stream.
    sink = on_event or (lambda _t, _p: None)

    rows: list[dict] = []

    async def _run_one(case: dict, rep: int) -> dict:
        job_id = uuid.uuid4()
        ctx = LocalRunContext(job_id, workspace_id, on_event=sink)
        workdir = create_workdir(str(job_id), workspace_id, case["inputs"])
        try:
            attach_skill(workdir, loaded.root)
            try:
                run = await run_agent(
                    None, job_id, workspace_id, deployment, loaded,
                    case["inputs"], workdir, None,
                    python_exe=python_exe, venv_dir=venv_dir,
                    model_override=model, use_cache=False, ctx=ctx,
                )
                score, graded = await run_evals(
                    ctx, loaded, case["inputs"], run.get("output"),
                    deployment_root=root, workdir=workdir,
                    python_exe=python_exe, secrets=None, model_slug=model_slug,
                    expected=case["expected"], has_expected=case["has_expected"],
                )
                graders = (graded or {}).get("graders", [])
                scored = [g for g in graders if not g.get("skipped")]
                passed = bool(scored) and all(g.get("passed") for g in scored)
                return {
                    "id": case["id"], "repeat": rep, "score": score,
                    "passed": passed, "graders": graders, "error": None,
                }
            except Exception as e:
                return {
                    "id": case["id"], "repeat": rep, "score": None,
                    "passed": False, "graders": [], "error": f"{type(e).__name__}: {e}",
                }
        finally:
            cleanup_workdir(str(job_id))

    async def _run_all():
        for case in cases:
            for rep in range(reps):
                row = await _run_one(case, rep)
                rows.append(row)
                _print_case(row, reps)

    asyncio.run(_run_all())

    total = len(rows)
    passed_n = sum(1 for r in rows if r["passed"])
    scored = [r["score"] for r in rows if r["score"] is not None]
    pass_rate = round(100.0 * passed_n / total, 1) if total else 0.0
    mean_score = round(sum(scored) / len(scored), 1) if scored else None
    gate_passed = threshold is None or pass_rate >= threshold
    return {
        "total": total,
        "passed": passed_n,
        "pass_rate_pct": pass_rate,
        "mean_score": mean_score,
        "gate_passed": gate_passed,
        "threshold": threshold,
        "cases": rows,
    }


def _print_case(row: dict, reps: int) -> None:
    tag = f"{row['id']}#{row['repeat']}" if reps > 1 else row["id"]
    if row["error"]:
        print(f"  ✗ {tag}: error — {row['error'][:160]}")
        return
    mark = "✓" if row["passed"] else "✗"
    print(f"  {mark} {tag}: score {row['score']}")
