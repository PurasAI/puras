"""Per-step agent checkpointing for resumable execution (P1-4).

The agentic loop holds its whole conversation (`messages`) in memory and grows it
each step. If the worker dies mid-run (deploy drain, OOM, reaper timeout) that
state is lost and the requeued job restarts from scratch — re-billing and
re-running every step. This persists the conversation at the end of every CLEAN
turn (a `user(tool_results)` boundary, so the next model call is always valid),
so a re-claimed job RESUMES from the last good step. At most the single step that
was in flight when the worker died re-runs.

Checkpoints are written only at top level (depth 1); a subagent is part of a
parent step and re-runs with it. Everything here is best-effort — a checkpoint
error never breaks (or slows materially) a run, it just forfeits resumability for
that step.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

log = structlog.get_logger()

# Don't persist a pathologically large state (e.g. a run that inlined many big
# base64 images into messages) — a multi-MB upsert every step isn't worth it.
# Such a run simply isn't resumable; it falls back to the prior restart behavior.
MAX_CHECKPOINT_BYTES = 6_000_000


async def save_checkpoint(
    session: AsyncSession,
    job_id: UUID,
    *,
    step: int,
    messages: list[dict],
    final_text_parts: list[str],
    structured_output: Any,
) -> None:
    """Upsert the job's checkpoint to resume at `step` (steps [0, step) are
    already captured in `messages`). Best-effort: serialization or DB errors are
    swallowed so checkpointing can never fail a run."""
    try:
        state = json.dumps(
            {
                "messages": messages,
                "final_text_parts": final_text_parts,
                "structured_output": structured_output,
            },
            default=str,
        )
    except (TypeError, ValueError):
        log.debug("checkpoint_serialize_failed", job_id=str(job_id))
        return
    if len(state) > MAX_CHECKPOINT_BYTES:
        log.debug("checkpoint_too_large", job_id=str(job_id), bytes=len(state))
        return
    try:
        await session.execute(
            text(
                "insert into job_checkpoints (job_id, step, state, updated_at) "
                "values (:jid, :step, cast(:state as jsonb), now()) "
                "on conflict (job_id) do update set "
                "  step = excluded.step, state = excluded.state, updated_at = now()"
            ).bindparams(jid=job_id, step=step, state=state)
        )
    except Exception:
        log.debug("checkpoint_save_failed", job_id=str(job_id), exc_info=True)


async def load_checkpoint(session: AsyncSession, job_id: UUID) -> dict | None:
    """Return the resume state `{step, messages, final_text_parts,
    structured_output}` for a job, or None when there's no checkpoint (a fresh
    run) or it can't be read. Best-effort — a load failure just means a fresh
    start."""
    try:
        row = (
            await session.execute(
                text("select step, state from job_checkpoints where job_id = :jid")
                .bindparams(jid=job_id)
            )
        ).first()
    except Exception:
        log.debug("checkpoint_load_failed", job_id=str(job_id), exc_info=True)
        return None
    if row is None:
        return None
    state = row.state
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except ValueError:
            return None
    if not isinstance(state, dict):
        return None
    messages = state.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    return {
        "step": int(row.step or 0),
        "messages": messages,
        "final_text_parts": state.get("final_text_parts") or [],
        "structured_output": state.get("structured_output"),
    }
