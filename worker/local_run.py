"""Offline runner — `puras run --local` (open-core PR 3/3).

Runs a skill bundle straight off local disk with NO platform: no Postgres, no
bucket, no platform API. The SAME `agent_runner.run_agent` loop drives it (that's
the whole point of the RunContext seam — one loop, two environments), here on a
`LocalRunContext` with `platform_enabled=False`, so the hosted-only tools
(memory / media / web / cross-skillpack subagents) are switched off and the
free local surface (text + bash + the file tools + deterministic skill tools +
in-process subagents) runs on the user's OWN LLM key (BYO).

`run_local()` is the programmatic entry; the CLI's `puras run --local` calls it.
The worker runtime isn't pip-installable (it carries the hosted DB/storage
stack), so the CLI imports this lazily and degrades with a clear message when the
runner package isn't importable.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable


class LocalRunError(RuntimeError):
    """A local run could not be set up or completed (bad bundle, missing key…)."""


# A stable synthetic workspace so a run's drive (and anything a prior run wrote)
# persists across invocations under one predictable folder, instead of a fresh
# random dir each time. There is no real workspace/tenant offline.
_LOCAL_WORKSPACE_ID = "00000000-0000-0000-0000-000000000000"


def _prepare_env(api_key: str | None, drive_path: str | None) -> None:
    """Make the worker config loadable offline.

    `WorkerSettings` hard-requires the hosted env (DATABASE_URL, SUPABASE_*,
    PURAS_SERVICE_TOKEN) so a misconfigured PROD worker fails fast at startup —
    valuable there, useless here. We fill harmless placeholders for the bits a
    local run never touches, flip on PURAS_LOCAL_MODE, and point the drive at a
    real local dir. The user's own LLM key is the one thing that must be real.
    `setdefault` never clobbers a value the user already exported."""
    os.environ.setdefault("DATABASE_URL", "")
    os.environ.setdefault("SUPABASE_URL", "")
    os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "")
    os.environ.setdefault("PURAS_SERVICE_TOKEN", "")
    os.environ["PURAS_LOCAL_MODE"] = "1"
    os.environ["LOCAL_DRIVE_PATH"] = drive_path or str(
        Path(tempfile.gettempdir()) / "puras-local" / "drive"
    )
    os.environ.setdefault(
        "WORKDIR_ROOT", str(Path(tempfile.gettempdir()) / "puras-local" / "jobs")
    )
    # Per-skill venvs + extracted bundles cache under deployments_root. Its
    # hosted default (/var/puras/deployments) is root-owned and unwritable for a
    # `pip install` user, so redirect it under the same local scratch folder as
    # the drive/workdir above — otherwise build_skill_python() dies on mkdir.
    os.environ.setdefault(
        "DEPLOYMENTS_ROOT",
        str(Path(tempfile.gettempdir()) / "puras-local" / "deployments"),
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LocalRunError(
            "no LLM key — set ANTHROPIC_API_KEY (or pass --api-key). A local run "
            "is BYO key: you call the provider directly and pay your own bill."
        )


def _pick_skill(manifest, skill: str | None) -> str:
    """Resolve which skill to run: the named one, or the sole top-level skill."""
    top = [s for s in manifest.skills if s.parent_skill is None]
    if skill:
        for s in manifest.skills:
            if s.name == skill:
                return s.name
        avail = ", ".join(s.name for s in top) or "(none)"
        raise LocalRunError(f"skill `{skill}` not found in bundle. available: {avail}")
    if not top:
        raise LocalRunError("no skills found in this bundle (no `<skill>/skill.yaml`)")
    if len(top) > 1:
        names = ", ".join(s.name for s in top)
        raise LocalRunError(
            f"bundle has several skills ({names}) — pass one as the skill argument"
        )
    return top[0].name


def run_local(
    skill_dir: str | Path,
    inputs: dict,
    *,
    skill: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    drive_path: str | None = None,
    on_event: Callable[[str, dict], None] | None = None,
) -> dict[str, Any]:
    """Run one skill from a local bundle dir, offline, on the user's own key.

    Returns `{output, steps, usage}` where `usage` is the informational token /
    cost tally the LocalRunContext accumulated (the user paid their provider
    directly; nothing was billed by a platform).

    `skill_dir` is a skillpack bundle root (a dir of `<skill>/skill.yaml`).
    `skill` selects one when the bundle has several; with one it's optional.
    `on_event(event_type, payload)` streams the run's events (defaults to a
    compact console print)."""
    root = Path(skill_dir).expanduser().resolve()
    if not root.is_dir():
        raise LocalRunError(f"bundle dir not found: {root}")

    _prepare_env(api_key, drive_path)
    # Settings may have been cached by an earlier call in this process; rebuild
    # them so our env edits take effect.
    from .config import get_settings

    get_settings.cache_clear()

    # Imported here (after env is ready) so module import never trips on the
    # hosted env the worker package otherwise assumes.
    import asyncio

    from .agent_runner import run_agent
    from .deployment import ResolvedDeployment, build_skill_python
    from .drive import setup_drive
    from .manifest import ManifestError, parse_bundle_dir
    from .run_context import LocalRunContext
    from .skill_loader import load as load_skill, load_adhoc
    from .workdir import attach_skill, cleanup_workdir, create_workdir

    setup_drive()

    try:
        manifest = parse_bundle_dir(root)
    except ManifestError as e:
        raise LocalRunError(f"invalid bundle: {e}") from e
    deployment = ResolvedDeployment(root=root, manifest=manifest, deployment_id=None)

    # A `references/*.md` path runs as an ad-hoc subagent; otherwise it's a
    # declared skill (resolved by name, or the bundle's sole top-level skill).
    if skill and str(skill).endswith(".md"):
        loaded = load_adhoc(root, skill)
    else:
        loaded = load_skill(manifest, root, _pick_skill(manifest, skill))

    if not loaded.is_agentic:
        raise LocalRunError(
            f"`{loaded.name}` is a deterministic (Python) skill — the local runner "
            f"drives the agent loop; run its function directly instead"
        )

    job_id = uuid.uuid4()
    workspace_id = _LOCAL_WORKSPACE_ID
    sink = on_event or _print_event
    ctx = LocalRunContext(job_id, workspace_id, on_event=sink)

    workdir = create_workdir(str(job_id), str(workspace_id), inputs)
    try:
        attach_skill(workdir, loaded.root)
        python_exe, venv_dir = build_skill_python(loaded.root)

        async def _go():
            return await run_agent(
                None, job_id, workspace_id, deployment, loaded,
                inputs, workdir, None,
                python_exe=python_exe, venv_dir=venv_dir,
                model_override=model,
                use_cache=False,
                ctx=ctx,
            )

        result = asyncio.run(_go())
    finally:
        cleanup_workdir(str(job_id))

    return {
        "output": result.get("output"),
        "steps": result.get("steps"),
        "usage": {
            "input_tokens": ctx.input_tokens,
            "output_tokens": ctx.output_tokens,
            "cost_micros": ctx.total_cost_micros,
        },
        # OTel-style trace spans (P0-3): run → step → model/tool, with
        # parent_span_id + duration_ms. Lets a local run show the same
        # latency waterfall the hosted job timeline does.
        "spans": ctx.spans,
    }


def _print_event(event_type: str, payload: dict) -> None:
    """Default console sink: one compact line per event, with a little shape for
    the events a watching human cares about most."""
    import json

    if event_type == "tool_use":
        label = payload.get("label") or payload.get("name")
        print(f"  → {payload.get('name')}: {label}")
        return
    if event_type == "tool_result":
        mark = "ok" if payload.get("ok") else "ERR"
        print(f"  ← [{mark}] {str(payload.get('preview', ''))[:160]}")
        return
    if event_type == "model_response":
        print(
            f"· step {payload.get('step')} "
            f"(in {payload.get('input_tokens')}, out {payload.get('output_tokens')})"
        )
        return
    try:
        extra = json.dumps(payload, default=str)
    except (TypeError, ValueError):
        extra = str(payload)
    print(f"· {event_type}: {extra[:200]}")
