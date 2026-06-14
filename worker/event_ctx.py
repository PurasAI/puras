"""The event-attribution ContextVar, factored out of `queue` so the agent loop
can read it WITHOUT importing `queue` (which builds module-level SQL and so drags
in sqlalchemy). The offline runner imports `event_ctx` from here; `queue`
re-exports it for the hosted code that still reaches it as `queue.event_ctx`.

Which agent emitted the event currently being recorded: the root run leaves it
None; a nested subagent sets it to its own id (== the run_subagent tool_use id)
for the duration of its run, so emit_event stamps every event with `ctx`. This
lets the UI rebuild the agent tree by id even when concurrent subagents
interleave their events. It's a ContextVar so each asyncio task (one per
concurrent subagent) carries its own value independently.
"""

from __future__ import annotations

import contextvars

event_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "puras_event_ctx", default=None
)
