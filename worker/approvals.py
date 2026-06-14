"""Human-in-the-loop confirmation gates (P1-5 security).

When the agent calls a tool a skill marked `confirm: true`, the worker records a
PENDING approval, emits an `approval_required` event, then BLOCKS the run until a
human approves or denies it from the dashboard
(api/app/routers/approvals.py) — or the wait times out. The gate is enforced here
at the dispatcher off the deploy-time manifest flag, so neither the model nor any
content it fetched can talk its way past it.

Failure mode is fail-CLOSED: a timeout, a cancel, or any error auto-DENIES the
call (a missed approval must never silently run a side effect).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .queue import emit_event, is_cancelled

log = structlog.get_logger()

_TERMINAL = {"approved", "denied", "expired"}
# Tool input is echoed into the approval row + event so the reviewer sees what
# they're approving — capped so a huge payload can't bloat the row/timeline.
_MAX_INPUT_CHARS = 4000


def _preview_input(tool_input: Any) -> Any:
    try:
        s = json.dumps(tool_input, default=str)
    except (TypeError, ValueError):
        return None
    if len(s) > _MAX_INPUT_CHARS:
        return {"_truncated": True, "preview": s[:_MAX_INPUT_CHARS]}
    return tool_input


async def request_approval(
    session: AsyncSession, job_id: UUID, tool_name: str, tool_input: Any
) -> str | None:
    """Insert a pending approval + emit `approval_required`. Returns the approval
    id, or None if the row couldn't be created (the caller then fails closed)."""
    preview = _preview_input(tool_input)
    try:
        row = (
            await session.execute(
                text(
                    "insert into job_approvals (job_id, tool_name, tool_input) "
                    "values (:jid, :tool, cast(:inp as jsonb)) returning id"
                ).bindparams(
                    jid=job_id, tool=tool_name, inp=json.dumps(preview, default=str)
                )
            )
        ).first()
    except Exception:
        log.warning("approval_create_failed", job_id=str(job_id), tool=tool_name, exc_info=True)
        return None
    approval_id = str(row.id)
    await emit_event(
        session, job_id, "approval_required",
        {"approval_id": approval_id, "tool": tool_name, "input": preview},
    )
    await session.commit()
    return approval_id


async def _decision(session: AsyncSession, approval_id: str) -> tuple[str, str | None]:
    row = (
        await session.execute(
            text("select status, reason from job_approvals where id = cast(:id as uuid)")
            .bindparams(id=approval_id)
        )
    ).first()
    if row is None:
        return "expired", None
    return row.status, row.reason


async def _expire(session: AsyncSession, approval_id: str) -> None:
    # Only flip a still-pending row, so a decision that lands in the same tick wins.
    await session.execute(
        text(
            "update job_approvals set status='expired', decided_at=now() "
            "where id = cast(:id as uuid) and status='pending'"
        ).bindparams(id=approval_id)
    )
    await session.commit()


async def wait_for_decision(
    session: AsyncSession, job_id: UUID, approval_id: str
) -> tuple[str, str | None]:
    """Block until the approval is decided, the run is cancelled, or the timeout
    elapses. Returns (decision, reason) where decision is approved|denied|expired.
    Fail-closed: cancel or timeout → 'expired' (treated as a deny by the caller)."""
    s = get_settings()
    poll = max(0.5, s.approval_poll_interval_s)
    waited = 0.0
    while waited < s.approval_timeout_s:
        status, reason = await _decision(session, approval_id)
        if status in _TERMINAL:
            if status != "expired":
                await emit_event(
                    session, job_id, "approval_decided",
                    {"approval_id": approval_id, "decision": status, "reason": reason},
                )
                await session.commit()
            return status, reason
        # A cancel while we wait aborts the gate (the loop's own is_cancelled
        # check will then end the run). Treat it as fail-closed.
        if await is_cancelled(session, job_id):
            await _expire(session, approval_id)
            return "expired", "run cancelled while awaiting approval"
        await asyncio.sleep(poll)
        waited += poll
    await _expire(session, approval_id)
    await emit_event(
        session, job_id, "approval_decided",
        {"approval_id": approval_id, "decision": "expired", "reason": "timed out"},
    )
    await session.commit()
    return "expired", "approval timed out"
