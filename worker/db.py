import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from .config import get_settings

_engine: AsyncEngine | None = None
_sm: async_sessionmaker[AsyncSession] | None = None


def _normalize(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    return url


def get_engine() -> AsyncEngine:
    global _engine, _sm
    if _engine is None:
        s = get_settings()
        # Pool is configured via DB_POOL_SIZE / DB_MAX_OVERFLOW (kept small to
        # bound the shared Supabase session-mode pool across machines), BUT it
        # must never be smaller than the worker's concurrency: each of the N
        # concurrent jobs holds one session open for its whole run, and on top of
        # that the heartbeat loop and a slot mid-claim each briefly need one. A
        # pool below N+2 could stall those on a sibling-held connection. So we
        # floor pool_size at concurrency+2 regardless of the configured value.
        conc = max(1, s.worker_concurrency)
        # Each running job holds one session for its whole run; on top of that a
        # turn that fans out tool calls (agent_runner._dispatch_one_tool) opens up
        # to PARALLEL_TOOL_LIMIT extra sessions concurrently — each held for that
        # tool's duration (a subagent can be minutes). Floor the pool at the worst
        # case so a fan-out can't stall waiting on a sibling-held connection.
        # (Connections are opened lazily, so a high floor costs nothing when idle.)
        parallel = max(1, int(os.getenv("PARALLEL_TOOL_LIMIT", "3")))
        pool_size = max(s.db_pool_size, conc * (1 + parallel) + 2)
        _engine = create_async_engine(
            _normalize(s.database_url),
            pool_size=pool_size,
            max_overflow=s.db_max_overflow,
            pool_timeout=s.db_pool_timeout,
            pool_recycle=s.db_pool_recycle,
            # A job session is held open for minutes; if the pooler dropped the
            # underlying server conn between queries, pre_ping reconnects on the
            # next checkout instead of failing the job on a dead socket.
            pool_pre_ping=True,
        )
        _sm = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


@asynccontextmanager
async def session() -> AsyncIterator[AsyncSession]:
    if _sm is None:
        get_engine()
    assert _sm is not None
    async with _sm() as s:
        try:
            yield s
            await s.commit()
        except Exception:
            await s.rollback()
            raise
