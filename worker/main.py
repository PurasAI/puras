"""Worker entrypoint. Polls the queue, claims one job, runs it, repeats.

A job targets a single skill. The skill's entrypoint decides how it runs:
  - `*.md`            → agentic loop (Claude/OpenRouter), system prompt = file
  - `<file.py>:<fn>`  → deterministic Python in a subprocess

Both paths validate inputs against the skill's `input_schema` and outputs
against `output_schema` (jsonschema). For agentic skills, the output_schema is
enforced via the auto-injected `set_output` tool (see agent_runner).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import time

import sentry_sdk
import structlog
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from sqlalchemy import text

from . import analytics
from .agent_runner import run_agent, validate_input_files
from .config import get_settings
from .db import session
from .deployment import (
    ResolvedDeployment,
    build_skill_python,
    resolve_deployment,
    resolve_local,
)
from .drive import resolve_output_dir, setup_drive, teardown_drive
from .eval_runner import run_evals
from .function_runner import run_function
from . import health
from . import resources
from .queue import (
    MICROS_PER_DOLLAR,
    claim_one,
    deregister_worker,
    emit_event,
    finalize_charges,
    get_job_cost,
    lookup_deployment,
    lookup_eval_case_expected,
    lookup_secrets,
    mark_failed,
    mark_succeeded,
    read_billing_snapshot,
    record_eval_case_result,
    register_worker,
    requeue_job,
    touch_job_heartbeat,
    touch_worker,
)
from .skill_loader import LoadedSkill, load as load_skill, load_adhoc, load_inline
from .storage import (
    build_output_manifest,
    ensure_input_files,
    relocate_outputs_to_run_dir,
    sync_output_files,
)
from .workdir import attach_skill, cleanup_workdir, create_workdir

log = structlog.get_logger()
_stopping = False
# The jobs currently being processed across all concurrent slots (each
# _process_one adds its id on claim, discards it on finish), so the background
# heartbeat task can stamp jobs.heartbeat_at for every in-flight job and report
# the machine's live load to the worker registry. Empty when fully idle.
# A plain set mutated from one event loop — no lock needed (single-threaded).
_current_job_ids: set[str] = set()


def _install_signals() -> None:
    def _stop(*_):
        global _stopping
        _stopping = True
        log.info("shutdown_signal_received")

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)


async def _resolve_deployment_for_job(sess, job: dict) -> ResolvedDeployment:
    """Resolve the deployment bundle this job's skill code lives in.

    In dev (no deployment_id), fall back to LOCAL_PROJECT_PATH. In prod,
    the job carries a concrete deployment_id (set by the API at submit
    time from the skillpack's active_deployment_id).
    """
    s = get_settings()
    if job["deployment_id"] is None:
        if not s.local_project_path:
            raise RuntimeError(
                "job has no deployment_id and worker has no LOCAL_PROJECT_PATH set"
            )
        return resolve_local()
    dep = await lookup_deployment(sess, job["deployment_id"])
    if dep is None:
        raise RuntimeError(f"deployment {job['deployment_id']} not found in DB")
    return resolve_deployment(str(dep["id"]), dep["storage_path"])


async def _lookup_skillpack_meta(sess, skillpack_id) -> tuple[str | None, str | None, str | None]:
    """(owner_workspace_id, skillpack_slug, publisher_workspace_slug) for a
    skillpack. owner_workspace_id drives the `cross_workspace` log/metric flag
    (drive and secrets paths don't branch on it); slug + publisher slug label
    the job analytics events ("which skillpack / which publisher"). Any field is
    None if the row vanished (publisher slug is None when the owner hasn't
    claimed a workspace slug)."""
    row = (
        await sess.execute(
            text(
                "select sp.owner_id, sp.slug as skillpack_slug, "
                "       pr.workspace_slug as publisher_slug "
                "from skillpacks sp "
                "left join profiles pr on pr.id = sp.owner_id "
                "where sp.id = :id"
            ).bindparams(id=skillpack_id)
        )
    ).first()
    if row is None:
        return None, None, None
    return str(row.owner_id), row.skillpack_slug, row.publisher_slug


def _inputs_equal(a: object, b: object) -> bool:
    """Order-insensitive deep equality for two input dicts."""
    try:
        return json.dumps(a, sort_keys=True, default=str) == json.dumps(
            b, sort_keys=True, default=str
        )
    except Exception:
        return a == b


def _is_default_example(manifest, skill_name: str, inputs: dict) -> bool | None:
    """Whether `inputs` match the skill's DEFAULT example (examples[0] — the one
    the playground seeds). True = the user ran the canned default unchanged,
    False = their own (or an edited) inputs. None when the skill declares no
    examples (e.g. inline/@adhoc subagents), where the distinction is undefined.

    Caveat: media inputs are materialized to drive refs before submit, so an
    image-bearing example won't match its manifest form here and reads as
    custom — fine for text/param skills, lossy for media-heavy ones."""
    decl = manifest.skill(skill_name)
    if decl is None or not decl.examples:
        return None
    return _inputs_equal(inputs, decl.examples[0].inputs)


def _validate(schema: dict, value, label: str) -> None:
    from .schema_dialect import to_jsonschema

    try:
        Draft202012Validator(to_jsonschema(schema)).validate(value)
    except ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) or "<root>"
        raise ValueError(f"{label} validation failed at `{path}`: {e.message}") from e


def _validate_output(schema: dict, value, label: str = "output"):
    """Same as `_validate` but lenient on extra object properties.

    Drops any properties not declared in the schema before validating, so
    the agent can echo `drive_path` next to an `image`-typed field (or
    similar scratch keys) without failing the run. Missing required fields
    and type mismatches still error.

    Returns the pruned value — the caller should persist this so the
    extras are stripped from the stored result, not just from validation.
    """
    from .schema_dialect import prune_extras, to_output_jsonschema

    pruned = prune_extras(schema, value)
    try:
        Draft202012Validator(to_output_jsonschema(schema)).validate(pruned)
    except ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) or "<root>"
        raise ValueError(f"{label} validation failed at `{path}`: {e.message}") from e
    return pruned


async def _score_evals(
    sess, job_id, workspace_id, skill, inputs, output,
    *, deployment_root, workdir, python_exe, secrets,
    expected=None, has_expected=False,
) -> tuple[int | None, dict | None]:
    """Run the skill's evals against a finished run and return (score, results).
    Fully best-effort: any failure logs and yields (None, None) so scoring can
    never turn a succeeded job into a failed one. The judge uses the skill's
    declared model (or the worker default), independent of any per-run override
    so the score stays comparable across runs.

    `expected`/`has_expected` carry an eval-suite case's expected output (None on
    a live run) so `exact_match` graders can score against it."""
    if not getattr(skill, "evals", None):
        return None, None
    s = get_settings()
    model_slug = skill.model or s.default_model_slug
    try:
        # Route the eval's I/O (the `evals` event, judge-call usage) through the
        # same RunContext seam the agent loop uses — behavior-identical hosted.
        from .run_context import DbRunContext

        ctx = DbRunContext(sess, job_id, workspace_id)
        return await run_evals(
            ctx, skill, inputs, output,
            deployment_root=deployment_root, workdir=workdir,
            python_exe=python_exe, secrets=secrets, model_slug=model_slug,
            expected=expected, has_expected=has_expected,
        )
    except Exception:
        log.warning("evals_failed", job_id=str(job_id), exc_info=True)
        return None, None


async def _finalize_eval_case(
    sess, job_id, *, succeeded: bool, eval_score, eval_results, t0: float,
    error: str | None = None,
) -> None:
    """Write an eval-suite case's grade onto its eval_case_results row. `passed`
    = there was at least one scored (non-skipped) grader and all of them passed;
    cost is read from the job, latency from the claim clock. Best-effort — a
    bookkeeping failure here must never change the job's own outcome."""
    cost = 0
    try:
        cost = await get_job_cost(sess, job_id)
    except Exception:
        pass
    latency_ms = int((time.monotonic() - t0) * 1000)
    passed: bool | None
    if not succeeded:
        passed = False
    elif eval_results:
        scored = [g for g in (eval_results.get("graders") or []) if not g.get("skipped")]
        passed = bool(scored) and all(bool(g.get("passed")) for g in scored)
    else:
        passed = None  # nothing scored — can't say pass/fail
    try:
        await record_eval_case_result(
            sess, job_id,
            status="succeeded" if succeeded else "failed",
            passed=passed,
            score=(eval_score if succeeded else 0),
            grader_results=(eval_results if succeeded else None),
            error=error,
            cost_micros=cost,
            latency_ms=latency_ms,
        )
    except Exception:
        log.warning("eval_case_record_failed", job_id=str(job_id), exc_info=True)


async def _process_one() -> bool:
    """Returns True if a job was processed, False if queue empty."""
    # Memory-aware admission: don't STACK a new job onto a machine that's
    # already busy and low on RAM — one in-flight ffmpeg encode can push N
    # concurrent jobs past the VM's memory. An idle machine (no active jobs)
    # always accepts work, so the queue never stalls; the proc_limits OOM-victim
    # hint is the backstop if we admit under pressure anyway.
    if _current_job_ids:
        s = get_settings()
        used_pct = resources.mem_used_pct()
        if used_pct is not None and used_pct >= s.claim_mem_ceiling_pct:
            log.info(
                "claim_skipped_mem_pressure",
                used_pct=used_pct,
                ceiling_pct=s.claim_mem_ceiling_pct,
                active_jobs=len(_current_job_ids),
            )
            return False

    async with session() as sess:
        job = await claim_one(sess)
        if job is None:
            return False

    job_id = job["id"]
    workspace_id = job["workspace_id"]
    skillpack_id = job["skillpack_id"]
    t0 = time.monotonic()
    log.info(
        "job_claimed",
        job_id=str(job_id),
        skill=job["skill_name"],
        workspace_id=str(workspace_id),
        skillpack_id=str(skillpack_id),
        deployment_id=str(job["deployment_id"]) if job["deployment_id"] else "local",
    )
    # Outcome accumulators for the single `job_completed` analytics event emitted
    # in `finally` (always fires, exactly once). The descriptive fields are
    # filled in once the deployment/skill resolves; they stay at these defaults
    # for a job that fails before then (e.g. missing deployment).
    source = job.get("source")
    status = "failed"  # flipped to "succeeded" on the happy path
    error_type: str | None = None
    is_agentic: bool | None = None
    steps: int | None = None
    resolved_skill_name = job["skill_name"]
    skillpack_slug: str | None = None
    publisher_slug: str | None = None
    is_public_run = False
    was_default_example: bool | None = None

    try:
        # Register with the heartbeat task only once we're INSIDE the try, so the
        # finally always discards the id and the except always marks the job —
        # even if create_workdir (or anything below) raises. Otherwise the id
        # would leak (heartbeat stamps a dead job forever, the registry
        # over-reports busy slots) and the already-'running' row would zombie
        # until the reaper times it out.
        _current_job_ids.add(str(job_id))
        workdir = create_workdir(str(job_id), str(workspace_id), job["inputs"])
        # Inputs are uploaded browser→Supabase directly (no bytes through the
        # worker), so pull each declared input from the bucket onto the worker's
        # local drive before the skill reads it. See ensure_input_files.
        await asyncio.to_thread(
            ensure_input_files, str(workspace_id), job["inputs"]
        )
        async with session() as sess:
            deployment = await _resolve_deployment_for_job(sess, job)
            # Secrets live on the skillpack (the code's publisher provides the
            # upstream-provider keys the skill needs). Drive + billing live on
            # the workspace; no branching either way.
            secrets = await lookup_secrets(sess, skillpack_id)
            owner_workspace_id, skillpack_slug, publisher_slug = (
                await _lookup_skillpack_meta(sess, skillpack_id)
            )
            # `cross_workspace` is for logging/metrics only — drive, secrets,
            # and billing all already resolved through workspace_id /
            # skillpack_id with no branching.
            cross_workspace = (
                owner_workspace_id is not None
                and owner_workspace_id != str(workspace_id)
            )
            is_public_run = cross_workspace

            # Three load paths for a subagent run (spawned by run_subagent /
            # subagent.run), all schema-less free-form agents that don't resolve
            # in the manifest:
            #   - skill_name == '@inline' + inline_prompt → run the raw prompt
            #     string in this bundle's context.
            #   - a `*.md` skill_name → run that bundle prompt file.
            # Anything else is a declared manifest skill.
            if job.get("inline_prompt") and str(job["skill_name"]) == "@inline":
                skill: LoadedSkill = load_inline(deployment.root, job["inline_prompt"])
            elif str(job["skill_name"]).endswith(".md"):
                skill = load_adhoc(deployment.root, job["skill_name"])
            else:
                skill = load_skill(
                    deployment.manifest, deployment.root, job["skill_name"]
                )
            attach_skill(workdir, skill.root)
            inputs = job["inputs"] if isinstance(job["inputs"], dict) else {}
            _validate(skill.input_schema, inputs, "input")
            # Shape-valid is not content-valid: a well-formed `{drive_path}` can
            # point at an empty (0-byte) or non-image file that only blows up deep
            # in the run (an empty image block 400s the model call). Check the
            # declared file inputs' actual bytes now — BEFORE any model spend — so
            # a bad upload fails fast with a message the end user can act on.
            await asyncio.to_thread(
                validate_input_files, skill.input_schema, inputs, str(workspace_id)
            )

            resolved_skill_name = skill.name
            is_agentic = skill.is_agentic
            # Per-run deliverables folder (`<skill>/<jobshort>`). The platform
            # files this run's outputs here at end-of-job regardless of where the
            # skill wrote them; the media/screenshot defaults also target it.
            out_dir = resolve_output_dir(str(workspace_id), skill.name, str(job_id))
            was_default_example = _is_default_example(
                deployment.manifest, job["skill_name"], inputs
            )

            await emit_event(
                sess, job_id, "skill_resolved",
                {
                    "skill": skill.name,
                    "cross_workspace": cross_workspace,
                    "skillpack_owner": owner_workspace_id,
                },
            )
            # Job start signal — emitted once the skill resolves so it carries
            # the full descriptive context (source / publisher / skillpack /
            # default-vs-custom). Its completion counterpart is `job_completed`
            # in `finally`. Cost/balance/duration are intentionally absent here
            # (unknown at start) — they live on job_completed.
            analytics.capture(
                str(workspace_id),
                "job_started",
                {
                    "job_id": str(job_id),
                    "source": source,
                    "skill_name": resolved_skill_name,
                    "skillpack_id": str(skillpack_id),
                    "skillpack_slug": skillpack_slug,
                    "publisher": publisher_slug,
                    "is_public_run": is_public_run,
                    "is_local": job["deployment_id"] is None,
                    "is_agentic": is_agentic,
                    "was_default_example": was_default_example,
                },
            )

            # Build (or reuse) the skill's own venv from skills/<name>/requirements.txt
            python_exe, venv_dir = await asyncio.to_thread(build_skill_python, skill.root)

            # Per-run agentic model override, set from the playground's model
            # picker (jobs.media_overrides["text"]). run_agent validates it and
            # falls back to the skill's own model if it isn't a known slug, so a
            # stale option never breaks a run.
            overrides = job.get("media_overrides")
            model_override = (
                overrides.get("text") if isinstance(overrides, dict) else None
            )

            if skill.is_agentic:
                # An eval-suite case bypasses the exact-match prompt cache: a
                # cached deterministic re-run replays one trace, which would zero
                # out the N-run variance the suite measures (`--repeat`). Normal
                # runs keep the cache (the step-0 saving is the whole point).
                use_cache = not bool(job.get("eval_suite_id"))
                result = await run_agent(
                    sess, job_id, workspace_id, deployment, skill,
                    inputs, workdir, secrets,
                    python_exe=python_exe, venv_dir=venv_dir,
                    model_override=model_override,
                    out_dir=out_dir,
                    use_cache=use_cache,
                )
                # run_agent already enforces output_schema via set_output;
                # but double-check on the wire just in case. Ad-hoc subagents
                # have no declared schema (free-form set_output) — skip the
                # check and store whatever they recorded.
                if not skill.is_adhoc:
                    result["output"] = _validate_output(
                        skill.output_schema, result.get("output")
                    )
                # Platform-enforced output organization: file the deliverable into
                # the run folder no matter where the skill wrote it (the skill is an
                # LLM — we don't trust it to choose the folder), rewriting result
                # paths to the new locations before they're served/persisted.
                result["output"] = await asyncio.to_thread(
                    relocate_outputs_to_run_dir,
                    str(workspace_id), result.get("output"), out_dir,
                )
                # Push any output media the skill produced to the bucket before
                # marking done — the API serves outputs only from the bucket.
                await asyncio.to_thread(sync_output_files, str(workspace_id), result)
                # Record the deliverable's files on jobs.outputs so the dashboard's
                # Outputs view can list this run's results (grouped by skill) without
                # scanning the bucket. Scoped to result["output"] (the set_output
                # deliverable), not the whole run record.
                outputs = await asyncio.to_thread(
                    build_output_manifest, str(workspace_id), result.get("output")
                )
                # For an eval-suite case run, pull the case's `expected` so
                # exact_match graders can score against it (live runs have none).
                has_expected, expected = (False, None)
                if job.get("eval_suite_id"):
                    has_expected, expected = await lookup_eval_case_expected(
                        sess, job_id
                    )
                # Score this run against the skill's declared evals (if any). Runs
                # BEFORE finalize_charges so any judge-model cost is debited with
                # the run; best-effort so a grader failure never fails the job.
                eval_score, eval_results = await _score_evals(
                    sess, job_id, workspace_id, skill, inputs, result.get("output"),
                    deployment_root=deployment.root, workdir=workdir,
                    python_exe=python_exe, secrets=secrets,
                    expected=expected, has_expected=has_expected,
                )
                await finalize_charges(sess, job_id)
                await mark_succeeded(
                    sess, job_id, result,
                    eval_score=eval_score, eval_results=eval_results,
                    outputs=outputs,
                )
                # An eval-suite case folds its grade into eval_case_results so the
                # suite can aggregate pass-rate / cost / latency / variance.
                if job.get("eval_suite_id"):
                    await _finalize_eval_case(
                        sess, job_id, succeeded=True,
                        eval_score=eval_score, eval_results=eval_results, t0=t0,
                    )
                await emit_event(
                    sess, job_id, "succeeded",
                    {"steps": result.get("steps"), "eval_score": eval_score},
                )
                status = "succeeded"
                steps = result.get("steps")
            else:
                await emit_event(
                    sess, job_id, "function_start",
                    {"name": skill.name, "module": skill.py_module, "func": skill.py_func},
                )
                out = await asyncio.to_thread(
                    run_function,
                    f"{skill.py_module}:{skill.py_func}",
                    inputs,
                    workdir,
                    deployment.root,
                    python_exe,
                    secrets,
                    str(workspace_id),
                    str(job_id),
                )
                if not out.get("ok"):
                    await mark_failed(sess, job_id, out.get("error") or "function failed")
                    if job.get("eval_suite_id"):
                        await _finalize_eval_case(
                            sess, job_id, succeeded=False,
                            eval_score=None, eval_results=None, t0=t0,
                            error=str(out.get("error") or "function failed")[:600],
                        )
                    await emit_event(sess, job_id, "failed", {"error": out.get("error")})
                    status = "failed"
                    error_type = "function_error"
                else:
                    result_value = out.get("result")
                    # Same lenient/all-required output contract as the agent path:
                    # prune undeclared keys, then require every declared property.
                    result_value = _validate_output(skill.output_schema, result_value)
                    # Same platform-enforced organization for deterministic skills.
                    result_value = await asyncio.to_thread(
                        relocate_outputs_to_run_dir,
                        str(workspace_id), result_value, out_dir,
                    )
                    await asyncio.to_thread(
                        sync_output_files, str(workspace_id), {"result": result_value}
                    )
                    outputs = await asyncio.to_thread(
                        build_output_manifest, str(workspace_id), result_value
                    )
                    await finalize_charges(sess, job_id)
                    await mark_succeeded(
                        sess, job_id, {"result": result_value}, outputs=outputs
                    )
                    # Deterministic skills declare no evals (agentic-only), so an
                    # eval-suite case here records a graderless 'succeeded' row —
                    # keeps the suite from hanging on a non-agentic target.
                    if job.get("eval_suite_id"):
                        await _finalize_eval_case(
                            sess, job_id, succeeded=True,
                            eval_score=None, eval_results=None, t0=t0,
                        )
                    await emit_event(sess, job_id, "succeeded", {})
                    status = "succeeded"

    except asyncio.CancelledError:
        # Shutdown drain (_drain_watchdog): the watchdog cancels this slot and
        # then requeues the job for a replacement machine. Don't mark the row
        # failed — the requeue flips it back to 'queued' right after we unwind.
        # The finally below still fires (id discard, workdir cleanup, analytics
        # with status='requeued') before the cancellation propagates.
        status = "requeued"
        error_type = "worker_drain"
        raise
    except Exception as e:
        log.exception("job_failed", job_id=str(job_id))
        # Job failures are persisted and never propagate past this handler, so
        # without an explicit capture they would never reach Sentry. No-op when
        # the SDK isn't initialized (SENTRY_DSN unset).
        with sentry_sdk.new_scope() as scope:
            scope.set_tag("job_id", str(job_id))
            scope.set_tag("skill", resolved_skill_name)
            scope.set_tag("workspace_id", str(workspace_id))
            sentry_sdk.capture_exception(e)
        status = "failed"
        error_type = type(e).__name__
        try:
            # Persist a concise, user-facing message only — the full traceback
            # (with local file paths) is already in the worker logs via
            # log.exception above. Dumping it into jobs.error leaked internals
            # to the playground and broke the frontend's headline/details split.
            err_msg = f"{type(e).__name__}: {e}".strip()[:600] or type(e).__name__
            async with session() as sess:
                await mark_failed(sess, job_id, err_msg)
                if job.get("eval_suite_id"):
                    await _finalize_eval_case(
                        sess, job_id, succeeded=False,
                        eval_score=None, eval_results=None, t0=t0, error=err_msg,
                    )
                await emit_event(sess, job_id, "failed", {"error": str(e)})
        except Exception:
            log.exception("job_finalize_failed", job_id=str(job_id))
    finally:
        _current_job_ids.discard(str(job_id))
        cleanup_workdir(str(job_id))
        # Single completion signal for the whole run — succeeded OR failed, with
        # cost/balance/duration and the full descriptive context. Read the
        # billing snapshot best-effort (its own session) so a DB hiccup here
        # never masks the real outcome. On failure cost is the wasted upstream
        # spend (the user wasn't debited) and balance is unchanged.
        cost_micros, balance_micros = 0, 0
        try:
            async with session() as sess:
                cost_micros, balance_micros = await read_billing_snapshot(sess, job_id)
        except Exception:
            log.warning("job_billing_snapshot_failed", job_id=str(job_id), exc_info=True)
        analytics.capture(
            str(workspace_id),
            "job_completed",
            {
                "job_id": str(job_id),
                "status": status,
                "source": source,
                "skill_name": resolved_skill_name,
                "skillpack_id": str(skillpack_id),
                "skillpack_slug": skillpack_slug,
                "publisher": publisher_slug,
                "is_public_run": is_public_run,
                "is_agentic": is_agentic,
                "steps": steps,
                "error_type": error_type,
                "duration_ms": int((time.monotonic() - t0) * 1000),
                "cost_micros": cost_micros,
                "cost_usd": round(cost_micros / MICROS_PER_DOLLAR, 6),
                "balance_after_micros": balance_micros,
                "balance_after_usd": round(balance_micros / MICROS_PER_DOLLAR, 6),
                "was_default_example": was_default_example,
            },
        )

    return True


async def _heartbeat_loop() -> None:
    """Refresh liveness on a fixed interval, decoupled from job progress.

    The Fly health check (health.py) reports healthy only while the heartbeat is
    fresh. Recording it once per poll iteration (the old behavior) meant any
    single job longer than the staleness window let the heartbeat go stale →
    /health 503 → Fly restarts the machine MID-JOB, stranding the run. Ticking
    here from a background task keeps the heartbeat fresh for the whole life of a
    long job — the event loop is free during the job's awaited `to_thread` calls
    (LLM round-trips, media polling), so this task keeps firing.

    The same tick best-effort stamps jobs.heartbeat_at for every in-flight job
    (so the API reaper can tell a live-but-slow job from one whose worker has
    died) and refreshes this machine's row in the worker registry with its live
    slot load (so the admin fleet view shows idle/busy slots in real time).
    """
    s = get_settings()
    interval = max(1.0, s.heartbeat_interval_s)
    # Keep beating while EITHER we're still running OR jobs are still draining
    # after SIGTERM. If we stopped at `_stopping`, an in-flight job that takes
    # longer than the reaper window (REAP_AFTER_S) during a graceful drain would
    # stop heartbeating and the API reaper would spuriously fail it mid-run on
    # every rolling deploy. main() cancels this task after the slots drain.
    while not _stopping or _current_job_ids:
        health.record_heartbeat()
        ids = list(_current_job_ids)
        # Job heartbeats (migration 022) and the worker registry (024) are
        # independent features — keep them in SEPARATE transactions so a missing
        # table for one (pre-migration) can't roll back the other. The in-memory
        # Fly heartbeat above keeps the machine alive regardless of either.
        if ids:
            try:
                async with session() as sess:
                    for jid in ids:
                        await touch_job_heartbeat(sess, jid)
            except Exception:
                log.debug("job_heartbeat_failed", count=len(ids))
        cpu_pct, mem_used_pct, mem_total_mb = resources.sample()
        try:
            async with session() as sess:
                await touch_worker(
                    sess, s.worker_id,
                    lane=s.job_lane,
                    concurrency=max(1, s.worker_concurrency),
                    active_jobs=len(ids),
                    region=s.fly_region,
                    cpu_pct=cpu_pct,
                    mem_used_pct=mem_used_pct,
                    mem_total_mb=mem_total_mb,
                )
        except Exception:
            log.debug("worker_touch_failed", worker_id=s.worker_id)
        await asyncio.sleep(interval)


async def _drain_watchdog(slot_tasks: list[asyncio.Task]) -> None:
    """Requeue whatever is still running when a shutdown drain is about to be
    hard-killed.

    SIGTERM starts a graceful drain: the claim loops stop taking work and
    in-flight jobs run to completion. But Fly SIGKILLs the VM kill_timeout
    (300s — the platform maximum) after the signal, so any job longer than
    that — a 15-minute video pipeline — used to die with the process and
    surface ~15 minutes later as a reaper "worker stopped responding" failure
    on every worker deploy (jobs d9edd295/1b919e94, killed mid-run by deploy
    v203). So: shortly BEFORE the hard kill, cancel the still-busy slots and
    hand their jobs back to the queue — replacement machines re-run them from
    scratch and the user gets a finished job instead of a spurious failure.
    The first attempt's upstream spend is sunk either way; the re-run strictly
    dominates the failure.

    Jobs that finish within drain_requeue_after_s drain normally — this task
    wakes up, sees nothing in flight, and exits without touching anything.
    """
    s = get_settings()
    while not _stopping:
        await asyncio.sleep(1.0)
    deadline = time.monotonic() + max(10.0, s.drain_requeue_after_s)
    while _current_job_ids and time.monotonic() < deadline:
        await asyncio.sleep(1.0)
    ids = sorted(_current_job_ids)
    if not ids:
        return
    log.warning("drain_requeueing", count=len(ids), job_ids=ids)
    for t in slot_tasks:
        t.cancel()
    # Let every _process_one unwind first (its CancelledError path skips the
    # mark-failed write), so a slot can't stamp 'failed' over a row we're
    # about to flip back to 'queued'.
    await asyncio.gather(*slot_tasks, return_exceptions=True)
    requeued = 0
    for jid in ids:
        try:
            async with session() as sess:
                if await requeue_job(
                    sess, jid,
                    reason="worker shut down before the job could finish "
                           "(deploy drain) — requeued for a fresh machine",
                ):
                    requeued += 1
        except Exception:
            log.exception("drain_requeue_failed", job_id=jid)
            sentry_sdk.capture_exception()
    log.warning("drain_requeued", requeued=requeued, of=len(ids))


async def _claim_loop(slot: int) -> None:
    """One concurrent job slot: claim a job, run it to completion, repeat.

    `worker_concurrency` of these run in the same event loop (asyncio.gather in
    main). Because a job spends almost all its wall time awaiting upstream I/O
    (LLM + fal via asyncio.to_thread), the slots genuinely overlap — one machine
    processes up to N jobs at once. Each claim is its own skip-locked
    transaction, so slots (and other machines) never double-claim the same job.

    _process_one already persists any job failure and never propagates; the
    guard here is purely so a claim-time DB hiccup can't kill the slot.
    """
    s = get_settings()
    while not _stopping:
        try:
            did_work = await _process_one()
        except Exception:
            log.exception("claim_loop_iteration_failed", slot=slot)
            sentry_sdk.capture_exception()
            did_work = False
        if not did_work:
            await asyncio.sleep(s.poll_interval_seconds)


async def main() -> None:
    s = get_settings()
    if s.sentry_dsn:
        sentry_sdk.init(dsn=s.sentry_dsn, environment=s.environment)
        # The Sentry project is shared with the sibling puras app — this tag is
        # how purasbackend events are told apart there.
        sentry_sdk.set_tag("service", "purasbackend-worker")
    _install_signals()

    # Tiny HTTP /health endpoint so the platform (fly, k8s) can detect a
    # stuck or dead loop. Heartbeat is recorded after every poll iteration
    # below; if it goes stale past s.health_staleness_window_s seconds, the
    # endpoint returns 503 and fly's check fails → alert.
    health.configure(s.health_staleness_window_s)
    health.start(port=s.health_port)

    drive_root = setup_drive()
    concurrency = max(1, s.worker_concurrency)
    log.info(
        "worker_started",
        worker_id=s.worker_id,
        concurrency=concurrency,
        model=s.default_model_slug,
        drive_root=str(drive_root),
        workdir_root=s.workdir_root,
        local_project_path=s.local_project_path,
        deployments_root=s.deployments_root,
    )

    # Announce this machine to the worker registry (best-effort; the table may
    # not exist pre-migration 024). Resets started_at for THIS process run so
    # the admin fleet view shows accurate uptime across restarts.
    try:
        async with session() as sess:
            await register_worker(
                sess, s.worker_id, lane=s.job_lane,
                concurrency=concurrency, region=s.fly_region,
            )
    except Exception:
        log.debug("worker_register_failed", worker_id=s.worker_id)

    # Heartbeat runs in its own task so liveness stays fresh DURING a long job,
    # not just between jobs (see _heartbeat_loop). Without this, any job longer
    # than the staleness window self-kills the worker mid-run.
    heartbeat_task = asyncio.create_task(_heartbeat_loop())
    # N concurrent claim loops in one event loop. They overlap their I/O waits,
    # so this machine runs up to `concurrency` jobs at once. The watchdog
    # requeues any job that outlives the post-SIGTERM drain window instead of
    # letting Fly's kill_timeout SIGKILL strand it (see _drain_watchdog).
    slot_tasks = [asyncio.create_task(_claim_loop(i)) for i in range(concurrency)]
    drain_task = asyncio.create_task(_drain_watchdog(slot_tasks))
    try:
        # return_exceptions: a drain-cancelled slot resolves here instead of
        # blowing CancelledError through main() and skipping the cleanup below.
        # Returns once every slot has drained (each exits when _stopping) or
        # the watchdog cancelled the stragglers.
        await asyncio.gather(*slot_tasks, return_exceptions=True)
    finally:
        # The watchdog may still be mid-requeue when the slots resolve — give
        # it time to commit; the timeout is only the something-went-wrong out.
        try:
            await asyncio.wait_for(drain_task, timeout=60)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task
        # Deregister so a scaled-down / stopped machine drops out of the admin
        # fleet view immediately instead of lingering until its row goes stale.
        try:
            async with session() as sess:
                await deregister_worker(sess, s.worker_id)
        except Exception:
            log.debug("worker_deregister_failed", worker_id=s.worker_id)
        teardown_drive()
        analytics.shutdown()
        log.info("worker_stopped")


if __name__ == "__main__":
    asyncio.run(main())
