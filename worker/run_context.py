"""RunContext — the seam between the agent runtime and the platform (P1-6).

`agent_runner.run_agent` is the crown-jewel loop and we want the SAME loop to run
two ways:
  - hosted: events/usage/cancel/cost/checkpoints go to Postgres, media/web/memory
    and cross-skillpack subagents go to the API (DbRunContext);
  - local (`puras run --local`, BYO LLM key): the same loop with no platform —
    events print to the console, usage is tallied for info, cancellation/cost
    caps are no-ops, checkpoints are skipped, and the hosted-only capabilities
    (memory, media, web, cross-skillpack subagents) are switched OFF
    (LocalRunContext).

Keeping the loop identical is what makes "runs identically locally and in prod"
true instead of a second, drifting implementation. This module defines that
contract and the two implementations. `agent_runner.run_agent` now routes ALL of
its loop I/O — events, usage, cancellation, the cost cap, checkpoints — through
`ctx`, including the per-task contexts the concurrent tool dispatch and the
subagent dispatch run on (`with_session` rebinds one to a fresh DB session). The
platform-only tools (memory, approvals, cross-skillpack subagents) still reach
the raw `ctx.session` directly; PR3 gates those on `platform_enabled` for the
local runner.

`platform_enabled` is the open-core line: True hosted (memory/media/web are the
paid value-adds), False local (text + bash + file + deterministic tools + in-proc
subagents run free on the user's own key).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

import structlog

log = structlog.get_logger()


class RunContext(ABC):
    """Everything the agent loop needs from its environment. Both the hosted and
    the local runtimes implement this; the loop only ever talks to it."""

    job_id: Any
    workspace_id: Any

    # The open-core switch: hosted media / web / cross-skillpack subagents.
    # Local runs flip this off so those tools are cleanly disabled rather than
    # erroring against an absent platform.
    platform_enabled: bool = True

    # Whether workspace memory (memory_search/get/put/forget + job-start
    # injection) is available. Hosted backs it with Postgres; a local run backs
    # it with a SQLite file (memory_store_sqlite) so it stays ON offline too.
    # The agent loop selects the backend via `memory_backend()`.
    memory_enabled: bool = True

    def memory_backend(self) -> tuple[Any, Any] | None:
        """Return `(module, handle)` for the active memory store, or None when
        memory is disabled. `module` exposes memory_search/get/put/forget/
        context taking `handle` as their first argument (the DB session hosted,
        a LocalMemoryStore locally)."""
        return None

    @abstractmethod
    async def emit_event(self, event_type: str, payload: dict) -> None: ...

    @abstractmethod
    async def record_usage(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        upstream_micros: int,
        billed_micros: int,
        meta: dict | None = None,
        cache_hit: bool = False,
    ) -> None: ...

    @abstractmethod
    async def record_span(
        self,
        *,
        span_id: str,
        parent_span_id: str | None,
        kind: str,
        name: str,
        duration_ms: int,
        attributes: dict | None = None,
    ) -> None:
        """Record one OTel-style trace span for this run (P0-3). The trace id is
        derived from the run's job id; an in-proc subagent shares it."""
        ...

    @abstractmethod
    async def is_cancelled(self) -> bool: ...

    @abstractmethod
    async def get_job_cost(self) -> int: ...

    @abstractmethod
    async def save_checkpoint(
        self, *, step: int, messages: list[dict],
        final_text_parts: list[str], structured_output: Any,
    ) -> None: ...

    @abstractmethod
    async def load_checkpoint(self) -> dict | None: ...

    @abstractmethod
    async def commit(self) -> None: ...


class DbRunContext(RunContext):
    """Hosted runtime: delegate to the existing Postgres-backed helpers, exactly
    as the loop calls them today (behavior-preserving). `with_session` rebinds the
    context to a fresh session for a concurrent tool task (the loop opens one per
    parallel dispatch)."""

    platform_enabled = True

    def __init__(self, session, job_id: UUID, workspace_id: UUID):
        self.session = session
        self.job_id = job_id
        self.workspace_id = workspace_id

    def with_session(self, session) -> "DbRunContext":
        return DbRunContext(session, self.job_id, self.workspace_id)

    def memory_backend(self):
        # Hosted: the Postgres store operates on the run's DB session.
        from . import memory_store
        return memory_store, self.session

    async def emit_event(self, event_type: str, payload: dict) -> None:
        from .queue import emit_event
        await emit_event(self.session, self.job_id, event_type, payload)

    async def record_usage(
        self, *, provider, model, input_tokens, output_tokens,
        upstream_micros, billed_micros, meta=None, cache_hit=False,
    ) -> None:
        from .queue import record_usage
        await record_usage(
            self.session, job_id=self.job_id, provider=provider, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            upstream_micros=upstream_micros, billed_micros=billed_micros,
            meta=meta, cache_hit=cache_hit,
        )

    async def record_span(
        self, *, span_id, parent_span_id, kind, name, duration_ms, attributes=None
    ) -> None:
        from .queue import record_span
        # trace_id == the run's job id (an in-proc subagent shares this job_id, so
        # its spans stitch into the same trace). Best-effort: never break the run.
        try:
            await record_span(
                self.session, self.job_id,
                trace_id=str(self.job_id), span_id=span_id,
                parent_span_id=parent_span_id, kind=kind, name=name,
                duration_ms=duration_ms, attributes=attributes,
            )
        except Exception:
            log.warning("record_span_failed", job_id=str(self.job_id), exc_info=True)

    async def is_cancelled(self) -> bool:
        from .queue import is_cancelled
        return await is_cancelled(self.session, self.job_id)

    async def get_job_cost(self) -> int:
        from .queue import get_job_cost
        return await get_job_cost(self.session, self.job_id)

    async def save_checkpoint(
        self, *, step, messages, final_text_parts, structured_output
    ) -> None:
        from .checkpoint import save_checkpoint
        await save_checkpoint(
            self.session, self.job_id, step=step, messages=messages,
            final_text_parts=final_text_parts, structured_output=structured_output,
        )

    async def load_checkpoint(self) -> dict | None:
        from .checkpoint import load_checkpoint
        return await load_checkpoint(self.session, self.job_id)

    async def commit(self) -> None:
        await self.session.commit()


class LocalRunContext(RunContext):
    """Offline runtime (`puras run --local`): the same loop with no platform.
    Events stream to a sink (the console by default), usage is tallied for info,
    cancellation/cost are no-ops, checkpoints are skipped, and the hosted-only
    tools are switched off. There is no DB session — `session` is None, so any
    not-yet-abstracted DB call must be guarded by `platform_enabled`.

    Workspace memory and web search are the exceptions that stay ON offline:
    memory is backed by a local SQLite file (memory_store_sqlite), and
    `web_search` is backed by Anthropic's native server-side web search on the
    user's BYO key (see `_anthropic_web_search`). `memory_backend()` hands the
    agent loop that store so memory_search/put/etc behave the same as hosted."""

    platform_enabled = False
    memory_enabled = True
    session = None

    def __init__(self, job_id, workspace_id, *, on_event=None):
        self.job_id = job_id
        self.workspace_id = workspace_id
        self._mem_store = None  # lazily-built LocalMemoryStore
        # Default sink: a compact one-line console print. The CLI can pass a
        # richer renderer.
        self._on_event = on_event or self._print_event
        # Informational running tallies (the user pays their own LLM bill).
        self.total_cost_micros = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.events: list[dict] = []
        self.spans: list[dict] = []

    def with_session(self, session) -> "LocalRunContext":
        # No platform DB → concurrent tasks share this context (and its one
        # SQLite memory store, which serializes its own access).
        return self

    def _resolve_memory_path(self):
        """The SQLite memory file: explicit PURAS_LOCAL_MEMORY_PATH, else a
        `memory.db` alongside the local drive root so it persists across runs."""
        from pathlib import Path

        from .config import get_settings

        s = get_settings()
        if s.local_memory_path:
            return Path(s.local_memory_path).expanduser()
        try:
            from .drive import get_drive_root
            return get_drive_root().parent / "memory.db"
        except Exception:
            import tempfile
            return Path(tempfile.gettempdir()) / "puras-local" / "memory.db"

    def memory_backend(self):
        if not self.memory_enabled:
            return None
        if self._mem_store is None:
            from .memory_store_sqlite import LocalMemoryStore
            self._mem_store = LocalMemoryStore(self._resolve_memory_path())
        from . import memory_store_sqlite
        return memory_store_sqlite, self._mem_store

    @staticmethod
    def _print_event(event_type: str, payload: dict) -> None:
        try:
            extra = json.dumps(payload, default=str)
        except (TypeError, ValueError):
            extra = str(payload)
        print(f"· {event_type}: {extra[:300]}")

    async def emit_event(self, event_type: str, payload: dict) -> None:
        self.events.append({"type": event_type, "payload": payload})
        self._on_event(event_type, payload)

    async def record_usage(
        self, *, provider, model, input_tokens, output_tokens,
        upstream_micros, billed_micros, meta=None, cache_hit=False,
    ) -> None:
        self.total_cost_micros += int(upstream_micros or 0)
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)

    async def record_span(
        self, *, span_id, parent_span_id, kind, name, duration_ms, attributes=None
    ) -> None:
        self.spans.append(
            {
                "trace_id": str(self.job_id),
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "kind": kind,
                "name": name,
                "duration_ms": duration_ms,
                "attributes": attributes or {},
            }
        )

    async def is_cancelled(self) -> bool:
        return False

    async def get_job_cost(self) -> int:
        # No platform cost cap locally — the user runs on their own key.
        return 0

    async def save_checkpoint(self, *, step, messages, final_text_parts, structured_output) -> None:
        return None

    async def load_checkpoint(self) -> dict | None:
        return None

    async def commit(self) -> None:
        return None
