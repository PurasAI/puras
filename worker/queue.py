"""Postgres-backed job queue. Claim one job at a time, atomically."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from . import analytics
from .config import get_settings

MICROS_PER_DOLLAR = 1_000_000

# The event-attribution ContextVar now lives in `event_ctx` (a light module the
# agent loop can import without pulling sqlalchemy via this module). Re-exported
# here so hosted callers that reference `queue.event_ctx` keep working.
from .event_ctx import event_ctx  # noqa: E402,F401

CLAIM_SQL = text(
    """
    with picked as (
      select j.id from jobs j
      -- The workspace's owning profile is always the wallet that pays.
      -- workspace_id IS the profile id (profiles.id == workspace_id).
      join profiles pr on pr.id = j.workspace_id
      where j.status = 'queued'
        and j.lane = :lane                    -- only claim jobs on our lane
        and pr.credit_balance_micros > 0      -- skip wallets with no credits
      -- Child-first ordering. Same-deployment subagents now run IN-PROCESS
      -- (agent_runner._dispatch_subagent) and never create a queued child job,
      -- so the common pipeline case can't deadlock at all. Only CROSS-skillpack
      -- subagents still fall back to a queued child whose parent blocks a slot
      -- waiting on it — for those, claiming children before new top-level jobs
      -- drains in-flight trees first and reduces (not eliminates) starvation.
      -- FIFO (created_at) within each class.
      order by (j.parent_job_id is not null) desc, j.created_at
      for update of j skip locked
      limit 1
    )
    update jobs j
    set status = 'running',
        started_at = now(),
        worker_id = :worker_id
    from picked
    where j.id = picked.id
    returning j.id, j.workspace_id, j.skillpack_id, j.deployment_id,
             j.type, j.skill_name, j.inputs, j.inline_prompt, j.source,
             j.media_overrides, j.eval_suite_id;
    """
)

LOOKUP_DEPLOYMENT_SQL = text(
    """
    select id, storage_path, manifest, version
    from deployments
    where id = :id
    """
)


async def claim_one(session: AsyncSession) -> dict[str, Any] | None:
    s = get_settings()
    row = (
        await session.execute(
            CLAIM_SQL.bindparams(worker_id=s.worker_id, lane=s.job_lane)
        )
    ).first()
    if row is None:
        return None
    return {
        "id": row.id,
        "workspace_id": row.workspace_id,
        "skillpack_id": row.skillpack_id,
        "deployment_id": row.deployment_id,
        "type": row.type,
        "skill_name": row.skill_name,
        "inputs": row.inputs,
        "inline_prompt": row.inline_prompt,
        "source": row.source,
        "media_overrides": row.media_overrides,
        "eval_suite_id": row.eval_suite_id,
    }


# CLAIM_SQL's inverse, for the shutdown drain (main._drain_watchdog): hand a
# still-running job back to the queue so a replacement machine re-runs it,
# instead of letting the platform's SIGKILL strand it as a reaper failure.
# Guarded on status AND worker_id so it can never yank a job that finished, was
# failed, or was re-claimed by another machine in the meantime. Clearing
# heartbeat_at/started_at makes the row look freshly queued to both CLAIM_SQL
# and the API reaper (which only considers status='running').
REQUEUE_SQL = text(
    """
    update jobs
       set status = 'queued',
           worker_id = null,
           started_at = null,
           heartbeat_at = null,
           resume_count = resume_count + 1
     where id = :job_id
       and status = 'running'
       and worker_id = :worker_id
    returning id
    """
)


async def requeue_job(session: AsyncSession, job_id, *, reason: str) -> bool:
    """Requeue one of THIS worker's running jobs; True if the row was flipped.

    The `requeued` event lands in the same transaction as the status flip, so
    the job timeline explains the hand-off instead of showing a silent gap
    between two runs."""
    s = get_settings()
    row = (
        await session.execute(
            REQUEUE_SQL.bindparams(job_id=job_id, worker_id=s.worker_id)
        )
    ).first()
    if row is None:
        return False
    await emit_event(session, job_id, "requeued", {"reason": reason})
    return True


async def lookup_secrets(session: AsyncSession, skillpack_id: UUID) -> dict[str, str]:
    """Return all secrets for a skillpack as a {NAME: VALUE} dict.

    Secrets are owned by the skillpack (its publisher provides them) — when
    a workspace runs someone else's public skillpack, those secrets travel
    with the code so the upstream-provider credentials the skill depends on
    are available regardless of which workspace is calling.
    """
    rows = (
        await session.execute(
            text("select name, value from skillpack_secrets where skillpack_id=:p")
            .bindparams(p=skillpack_id)
        )
    ).all()
    # Decrypt at-rest values (P1-5). A legacy plaintext row (no `enc:` prefix)
    # passes through unchanged, so this is safe before any backfill.
    from .secret_crypto import decrypt

    return {r.name: decrypt(r.value) for r in rows}


async def lookup_deployment(session: AsyncSession, deployment_id: UUID) -> dict | None:
    row = (await session.execute(LOOKUP_DEPLOYMENT_SQL.bindparams(id=deployment_id))).first()
    if row is None:
        return None
    return {
        "id": row.id,
        "storage_path": row.storage_path,
        "manifest": row.manifest,
        "version": row.version,
    }


async def emit_event(
    session: AsyncSession, job_id: UUID, event_type: str, payload: dict
) -> None:
    # Stamp the emitting agent's context (None for the root run) so the UI can
    # attribute interleaved events from concurrent subagents to the right node.
    ctx = event_ctx.get()
    if ctx is not None and "ctx" not in payload:
        payload = {**payload, "ctx": ctx}
    await session.execute(
        text(
            "insert into job_events (job_id, type, payload) "
            "values (:job_id, :type, cast(:payload as jsonb))"
        ).bindparams(job_id=job_id, type=event_type, payload=json.dumps(payload))
    )
    # Wake up any /jobs/{id}/stream listener immediately. Payload is just the
    # job_id; the listener fetches new rows from job_events by id > last_id.
    # Channel is per-job so each SSE consumer only sees its own traffic.
    await session.execute(
        text("select pg_notify(:chan, :payload)").bindparams(
            chan=f"puras_job_events:{job_id}", payload=str(job_id),
        )
    )


async def record_span(
    session: AsyncSession,
    job_id: UUID,
    *,
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    kind: str,
    name: str,
    duration_ms: int,
    attributes: dict | None = None,
) -> None:
    """Persist one trace span (P0-3). Insert-only, no commit — mirrors emit_event:
    the agent loop's frequent commits (and the terminal job commit) flush it. The
    parallel-tool path runs inside a `db.session()` that commits on exit."""
    await session.execute(
        text(
            "insert into job_spans "
            "(job_id, trace_id, span_id, parent_span_id, kind, name, duration_ms, attributes) "
            "values (:job_id, :trace_id, :span_id, :parent_span_id, :kind, :name, "
            ":duration_ms, cast(:attributes as jsonb))"
        ).bindparams(
            job_id=job_id,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            kind=kind,
            name=name,
            duration_ms=duration_ms,
            attributes=json.dumps(attributes or {}, default=str),
        )
    )


async def mark_succeeded(
    session: AsyncSession,
    job_id: UUID,
    result: dict,
    eval_score: int | None = None,
    eval_results: dict | None = None,
    outputs: list | None = None,
) -> None:
    # Guard on status='running' so we never overwrite a terminal state another
    # actor already set: the API reaper failing a presumed-dead worker's job, or
    # a user cancelling it. Whoever reaches terminal first wins; this write
    # no-ops otherwise instead of racing.
    #
    # eval_score / eval_results are written atomically with the result so the
    # score lands on the same row the UI reads — NULL when the skill declares no
    # evals (the common case), so this stays backward-compatible.
    await session.execute(
        text(
            "update jobs set status='succeeded', result=cast(:result as jsonb), "
            "outputs=cast(:outputs as jsonb), "
            "eval_score=:eval_score, eval_results=cast(:eval_results as jsonb), "
            "finished_at=:finished_at where id=:id and status='running'"
        ).bindparams(
            result=json.dumps(result, default=str),
            outputs=(json.dumps(outputs, default=str) if outputs is not None else None),
            eval_score=eval_score,
            eval_results=(
                json.dumps(eval_results, default=str) if eval_results is not None else None
            ),
            finished_at=datetime.utcnow(),
            id=job_id,
        )
    )


async def mark_failed(session: AsyncSession, job_id: UUID, error: str) -> None:
    # Same status='running' guard as mark_succeeded: don't clobber a job the
    # reaper already failed or the user already cancelled (a cancelled job stays
    # 'cancelled' instead of being downgraded to 'failed').
    await session.execute(
        text(
            "update jobs set status='failed', error=:error, finished_at=:finished_at "
            "where id=:id and status='running'"
        ).bindparams(error=error[:8000], finished_at=datetime.utcnow(), id=job_id)
    )


async def record_usage(
    session: AsyncSession,
    job_id: UUID,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    upstream_micros: int,
    billed_micros: int,
    meta: dict | None = None,
    cache_hit: bool = False,
) -> None:
    """Insert a usage_event and accrue the charge onto jobs.cost_micros. The
    caller's wallet is NOT debited here — wallet debit is deferred to
    finalize_charges() on successful job completion, so failed/cancelled jobs
    cost the user nothing. Balance is checked only at claim time (CLAIM_SQL),
    so mid-job dips below zero are accepted.

    `cache_hit` marks a call served from the exact-match prompt cache (migration
    037): billed_micros is 0 on a hit, and the flag makes hit-rate / saved spend
    queryable.

    usage_events.workspace_id mirrors jobs.workspace_id (always non-NULL —
    the caller's workspace owns the bill), so it's pulled from the job
    directly."""
    await session.execute(
        text(
            "insert into usage_events "
            "(workspace_id, job_id, provider, model, input_tokens, output_tokens, "
            " upstream_cost_micros, billed_micros, cache_hit, meta) "
            "select j.workspace_id, :jid, :prov, :model, :in_tok, :out_tok, :up, :bill, "
            "       :cache_hit, cast(:meta as jsonb) "
            "from jobs j where j.id = :jid"
        ).bindparams(
            jid=job_id, prov=provider, model=model,
            in_tok=input_tokens, out_tok=output_tokens,
            up=upstream_micros, bill=billed_micros,
            cache_hit=cache_hit,
            meta=json.dumps(meta or {}),
        )
    )
    await session.execute(
        text(
            "update jobs set cost_micros = cost_micros + :bill where id = :jid"
        ).bindparams(bill=billed_micros, jid=job_id)
    )


async def finalize_charges(session: AsyncSession, job_id: UUID) -> int:
    """Debit the caller's wallet by jobs.cost_micros — the total accrued during
    the run. Call once on successful completion. Failed/cancelled jobs skip this
    entirely, so the user pays nothing for them. Returns the new balance (can be
    negative; we accept that since the balance was positive at claim time).

    Wallet is always the job's workspace (workspace_id IS the profile id);
    cross-workspace skillpack runs still bill the caller's workspace, not the
    skillpack publisher."""
    cost_row = (
        await session.execute(
            text("select cost_micros, workspace_id from jobs where id = :jid")
            .bindparams(jid=job_id)
        )
    ).first()
    cost = int(cost_row.cost_micros) if cost_row else 0
    if cost <= 0:
        return await get_balance(session, job_id)
    row = (
        await session.execute(
            text(
                "update profiles set credit_balance_micros = credit_balance_micros - :bill "
                "where id = (select workspace_id from jobs where id = :jid) "
                "returning credit_balance_micros"
            ).bindparams(bill=cost, jid=job_id)
        )
    ).first()
    balance = int(row.credit_balance_micros) if row else 0
    # Spend/value signal — the only place real credit consumption is recorded.
    # Fires once per billable job (cost > 0). See docs/analytics.md on ROAS.
    analytics.capture(
        str(cost_row.workspace_id) if cost_row else None,
        "credits_charged",
        {
            "job_id": str(job_id),
            "amount_micros": cost,
            "amount_usd": round(cost / MICROS_PER_DOLLAR, 6),
            "balance_after_micros": balance,
        },
    )
    return balance


async def get_balance(session: AsyncSession, job_id: UUID) -> int:
    """Caller's current wallet balance, resolved via the job's workspace."""
    row = (
        await session.execute(
            text(
                "select credit_balance_micros from profiles "
                "where id = (select workspace_id from jobs where id = :jid)"
            ).bindparams(jid=job_id)
        )
    ).first()
    return int(row.credit_balance_micros) if row else 0


async def read_billing_snapshot(session: AsyncSession, job_id: UUID) -> tuple[int, int]:
    """(cost_micros accrued on the job, caller wallet balance) in one query —
    read at job completion to stamp the job_completed analytics event. On
    success the cost equals what finalize_charges debited and the balance is
    post-debit; on failure nothing was debited (cost is the wasted upstream
    spend, balance unchanged). Returns (0, 0) if the job row vanished."""
    row = (
        await session.execute(
            text(
                "select j.cost_micros, pr.credit_balance_micros "
                "from jobs j join profiles pr on pr.id = j.workspace_id "
                "where j.id = :jid"
            ).bindparams(jid=job_id)
        )
    ).first()
    if row is None:
        return 0, 0
    return int(row.cost_micros), int(row.credit_balance_micros)


async def get_job_cost(session: AsyncSession, job_id: UUID) -> int:
    """Total micros accrued on this job so far (LLM + media). Read by the agent
    loop to enforce a per-job spend cap mid-run."""
    row = (
        await session.execute(
            text("select cost_micros from jobs where id=:id").bindparams(id=job_id)
        )
    ).first()
    return int(row.cost_micros) if row else 0


async def lookup_eval_case_expected(
    session: AsyncSession, child_job_id: UUID
) -> tuple[bool, Any]:
    """For an eval-suite case job, return (has_expected, expected) — the case's
    expected output, handed to exact_match graders. Returns (False, None) when
    there's no case row or the case carries no `expected` (treats a JSON null the
    same as absent)."""
    row = (
        await session.execute(
            text(
                "select expected from eval_case_results "
                "where child_job_id = :jid limit 1"
            ).bindparams(jid=child_job_id)
        )
    ).first()
    if row is None or row.expected is None:
        return False, None
    return True, row.expected


async def record_eval_case_result(
    session: AsyncSession,
    child_job_id: UUID,
    *,
    status: str,
    passed: bool | None,
    score: int | None,
    grader_results: dict | None,
    error: str | None,
    cost_micros: int,
    latency_ms: int | None,
) -> None:
    """Write an eval suite case's grade back onto its pre-inserted
    eval_case_results row (keyed by the case job's id). The suite aggregate is
    computed from these rows on read (api/app/routers/evals.py)."""
    await session.execute(
        text(
            "update eval_case_results set "
            "  status = :status, passed = :passed, score = :score, "
            "  grader_results = cast(:grader_results as jsonb), error = :error, "
            "  cost_micros = :cost_micros, latency_ms = :latency_ms, "
            "  finished_at = now() "
            "where child_job_id = :jid"
        ).bindparams(
            jid=child_job_id,
            status=status,
            passed=passed,
            score=score,
            grader_results=(
                json.dumps(grader_results, default=str)
                if grader_results is not None
                else None
            ),
            error=(error[:2000] if error else None),
            cost_micros=int(cost_micros or 0),
            latency_ms=latency_ms,
        )
    )


async def touch_job_heartbeat(session: AsyncSession, job_id) -> None:
    """Stamp jobs.heartbeat_at = now() for the in-flight job. The worker calls
    this on a fixed interval (main._heartbeat_loop) so the API reaper can tell a
    live-but-slow job from one whose worker has died and requeue/fail the latter.
    Raises if the column is missing (pre-migration); callers treat it as
    best-effort.

    `id` is cast to uuid: the heartbeat loop passes job ids as STRINGS (from the
    in-flight `_current_job_ids` set), and asyncpg won't bind a str to a uuid
    column — without the cast every heartbeat silently failed and jobs.heartbeat_at
    stayed NULL, so the reaper fell back to started_at and could falsely reap any
    job running longer than its window."""
    await session.execute(
        text("update jobs set heartbeat_at = now() where id = cast(:id as uuid)")
        .bindparams(id=str(job_id))
    )


async def register_worker(
    session: AsyncSession,
    worker_id: str,
    lane: str,
    concurrency: int,
    region: str | None,
) -> None:
    """Upsert this machine's row in the worker registry at process startup.

    Sets started_at = now() for THIS process run (so uptime resets across
    restarts of the same FLY_MACHINE_ID) and zeroes the active-job count. The
    admin fleet view reads these rows; the periodic touch_worker() keeps
    last_seen_at fresh so a dead machine ages out of the view."""
    await session.execute(
        text(
            """
            insert into worker_heartbeats
              (worker_id, lane, concurrency, active_jobs, region, started_at, last_seen_at)
            values (:wid, :lane, :conc, 0, :region, now(), now())
            on conflict (worker_id) do update set
              lane = excluded.lane,
              concurrency = excluded.concurrency,
              active_jobs = 0,
              region = excluded.region,
              started_at = now(),
              last_seen_at = now()
            """
        ).bindparams(wid=worker_id, lane=lane, conc=concurrency, region=region)
    )


async def touch_worker(
    session: AsyncSession,
    worker_id: str,
    lane: str,
    concurrency: int,
    active_jobs: int,
    region: str | None,
    cpu_pct: float | None = None,
    mem_used_pct: float | None = None,
    mem_total_mb: int | None = None,
) -> None:
    """Refresh this machine's registry row each heartbeat: bump last_seen_at,
    record its current live slot load and self-sampled CPU/memory health (so the
    admin view shows idle/busy slots + resource headroom in real time).
    Self-healing upsert — if the row is missing (registration raced/failed or the
    row was deleted) it is recreated, so a live worker always appears in the
    fleet view. started_at is only set on (re)insert; ON CONFLICT preserves it,
    so register_worker remains the authoritative per-process reset."""
    await session.execute(
        text(
            """
            insert into worker_heartbeats
              (worker_id, lane, concurrency, active_jobs, region,
               cpu_pct, mem_used_pct, mem_total_mb, started_at, last_seen_at)
            values (:wid, :lane, :conc, :n, :region,
                    :cpu, :mem, :memtot, now(), now())
            on conflict (worker_id) do update set
              lane = excluded.lane,
              concurrency = excluded.concurrency,
              active_jobs = excluded.active_jobs,
              region = excluded.region,
              cpu_pct = excluded.cpu_pct,
              mem_used_pct = excluded.mem_used_pct,
              mem_total_mb = excluded.mem_total_mb,
              last_seen_at = now()
            """
        ).bindparams(
            wid=worker_id, lane=lane, conc=concurrency, n=active_jobs, region=region,
            cpu=cpu_pct, mem=mem_used_pct, memtot=mem_total_mb,
        )
    )


async def deregister_worker(session: AsyncSession, worker_id: str) -> None:
    """Drop this machine's registry row on graceful shutdown so a scaled-down or
    redeploying machine leaves the admin fleet view at once (a hard-killed
    machine instead ages out via its stale last_seen_at)."""
    await session.execute(
        text("delete from worker_heartbeats where worker_id = :wid")
        .bindparams(wid=worker_id)
    )


async def is_cancelled(session: AsyncSession, job_id: UUID) -> bool:
    row = (
        await session.execute(
            text("select status from jobs where id=:id").bindparams(id=job_id)
        )
    ).first()
    return row is not None and row.status == "cancelled"


# Workspace memory moved to worker/worker/memory_store.py (memory v2 — hybrid
# retrieval, soft-delete/supersedence, provenance). Import from there.
