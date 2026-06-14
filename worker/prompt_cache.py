"""Exact-match cross-request LLM cache (P0-2a cost control).

An agent re-running the same skill on the same inputs issues an identical first
model call (full system prompt + every tool definition + the first user turn) —
usually the single biggest call of the run. This caches the model RESPONSE keyed
on a hash of the normalized request, so an identical later request is served from
the store with NO upstream call and billed 0.

Why later steps still mostly miss (and that's correct): step K's messages embed
the prior turns' tool_use ids + tool results. A deterministic re-run replays the
cached earlier responses, so it reproduces the SAME ids/results and keeps hitting
all the way down; a run whose tools return something new (a fresh media URL, a
live web result) diverges at that step and misses from there on — i.e. the cache
collapses exactly the deterministic prefix and no more. Because a deterministic
re-run collapses to one cached trace, eval suites that measure N-run VARIANCE
pass `use_cache=False` to keep their repeats independent.

Distinct from Anthropic's in-call prompt caching (cheap re-reads WITHIN one run):
this skips the request entirely across runs.

Best-effort throughout: any cache error (miss, DB hiccup, serialization) falls
back to a normal upstream call — the cache can never break or slow a run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

# `sqlalchemy` (text) is imported lazily inside _get/_put so the OFFLINE runner
# can import this module — and call cached_messages_create with use_cache=False
# (no cache lookup, so _get/_put never run) — without sqlalchemy installed. The
# AsyncSession annotations are strings (PEP 563), so a TYPE_CHECKING import is
# enough for them.
if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .providers.base import NormalizedResponse, NormalizedToolUse

log = structlog.get_logger()

# Bump when the key inputs or the serialized response shape change, so a deploy
# can't serve a stale-shaped entry written by an older worker.
_CACHE_VERSION = 1


def cache_key(
    *,
    model_slug: str,
    system: str,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
    workspace_id: str,
) -> str:
    """sha256 over the canonicalized request. workspace_id is in the key so an
    identical prompt never crosses a tenant boundary; model_slug + system + tools
    mean any skill or model change misses automatically."""
    payload = {
        "v": _CACHE_VERSION,
        "model": model_slug,
        "ws": workspace_id,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
        "tools": tools,
    }
    blob = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _serialize(resp: NormalizedResponse) -> dict[str, Any]:
    return {
        "stop_reason": resp.stop_reason,
        "text_blocks": list(resp.text_blocks),
        "tool_uses": [
            {"id": tu.id, "name": tu.name, "input": tu.input} for tu in resp.tool_uses
        ],
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "upstream_cost_micros": resp.upstream_cost_micros,
    }


def _deserialize(d: dict[str, Any]) -> NormalizedResponse:
    return NormalizedResponse(
        stop_reason=d.get("stop_reason") or "end_turn",
        text_blocks=list(d.get("text_blocks") or []),
        tool_uses=[
            NormalizedToolUse(
                id=tu.get("id", ""),
                name=tu.get("name", ""),
                input=tu.get("input") if isinstance(tu.get("input"), dict) else {},
            )
            for tu in (d.get("tool_uses") or [])
        ],
        input_tokens=int(d.get("input_tokens") or 0),
        output_tokens=int(d.get("output_tokens") or 0),
        # A cache hit costs nothing — caching counters and context-mgmt stats are
        # per-call and not meaningful on replay.
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        upstream_cost_micros=int(d.get("upstream_cost_micros") or 0),
        context_management_applied=None,
    )


async def _get(session: AsyncSession, key: str) -> NormalizedResponse | None:
    """Look up + bump a cache entry, respecting the TTL. Returns None on miss or
    any error (best-effort)."""
    from sqlalchemy import text

    s = get_settings()
    ttl = s.prompt_cache_ttl_s
    try:
        ttl_clause = (
            "and created_at > now() - make_interval(secs => :ttl)" if ttl and ttl > 0 else ""
        )
        row = (
            await session.execute(
                text(
                    f"update prompt_cache set hit_count = hit_count + 1, "
                    f"  last_hit_at = now() "
                    f"where cache_key = :k {ttl_clause} "
                    f"returning response"
                ).bindparams(**({"k": key, "ttl": ttl} if ttl_clause else {"k": key}))
            )
        ).first()
    except Exception:
        log.debug("prompt_cache_get_failed", exc_info=True)
        return None
    if row is None:
        return None
    resp_json = row.response
    if isinstance(resp_json, str):
        try:
            resp_json = json.loads(resp_json)
        except ValueError:
            return None
    if not isinstance(resp_json, dict):
        return None
    try:
        return _deserialize(resp_json)
    except Exception:
        return None


async def _put(
    session: AsyncSession,
    key: str,
    *,
    workspace_id: UUID | str,
    model_slug: str,
    resp: NormalizedResponse,
) -> None:
    """Store a fresh response. Best-effort; a concurrent identical write is a
    harmless no-op (ON CONFLICT DO NOTHING)."""
    from sqlalchemy import text

    try:
        await session.execute(
            text(
                "insert into prompt_cache "
                "(cache_key, workspace_id, model, response, input_tokens, "
                " output_tokens, upstream_cost_micros) "
                "values (:k, cast(:ws as uuid), :model, cast(:resp as jsonb), "
                "        :in_tok, :out_tok, :cost) "
                "on conflict (cache_key) do nothing"
            ).bindparams(
                k=key,
                ws=str(workspace_id),
                model=model_slug,
                resp=json.dumps(_serialize(resp), default=str),
                in_tok=resp.input_tokens,
                out_tok=resp.output_tokens,
                cost=resp.upstream_cost_micros,
            )
        )
    except Exception:
        log.debug("prompt_cache_put_failed", exc_info=True)


async def cached_messages_create(
    session: AsyncSession,
    provider,
    *,
    model_slug: str,
    workspace_id: UUID | str,
    system: str,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
    cache_messages: bool,
    use_cache: bool,
) -> tuple[NormalizedResponse, bool]:
    """Run one inference, serving from the exact-match cache when possible.

    Returns (response, cache_hit). On a hit the response is replayed from the
    store (its `upstream_cost_micros` is the ORIGINAL call's cost, so the caller
    can report the savings) and no upstream call is made. On a miss the fresh
    response is stored. Caching is skipped entirely when `use_cache` is False or
    the feature is disabled."""
    s = get_settings()
    enabled = use_cache and s.prompt_cache_enabled

    key: str | None = None
    if enabled:
        try:
            key = cache_key(
                model_slug=model_slug, system=system, messages=messages,
                tools=tools, max_tokens=max_tokens, workspace_id=str(workspace_id),
            )
        except Exception:
            key = None
    if key is not None:
        hit = await _get(session, key)
        if hit is not None:
            return hit, True

    resp = await asyncio.to_thread(
        provider.messages_create, system, messages, tools, max_tokens,
        cache_messages=cache_messages,
    )
    if key is not None:
        await _put(session, key, workspace_id=workspace_id, model_slug=model_slug, resp=resp)
    return resp, False
