"""Memory v2 store — all DB access for the workspace "shared brain"
(workspace_memory table, migrations 029 + 032).

Retrieval is HYBRID, the consensus design of production agent-memory systems
(Mem0 / Zep / Letta): up to three candidate signals are fused with Reciprocal
Rank Fusion and the fused score is shaped by read-time heuristics —

    score = RRF(exact, fts, vector) × recency_decay(mtype) × importance × pin

  * exact   — entity_key / alias / content_hash equality (weight 1.0; identity
              matches must dominate).
  * fts     — Postgres full-text + ILIKE over title/summary/kind/entity_key
              (weight 0.7). The to_tsvector expression matches the GIN
              expression index from migration 032 verbatim.
  * vector  — pgvector cosine over `embedding` (weight 0.9). Only when the
              column exists (migration 032's best-effort block) AND an
              embedding endpoint is configured — probed once per process and
              degraded silently otherwise.

Writes are LLM-at-write: the agent supplies `summary` / `tags` / `importance`
once, so reads never re-reason. Rows are soft-deleted (`deleted_at` +
`superseded_by`), never destroyed by agents; the nightly
`workspace_memory_maintenance()` SQL function does the actual forgetting.

Tenancy: the worker connects with a privileged role that bypasses RLS, so the
`workspace_id = cast(:ws as uuid)` predicate on EVERY statement here IS the
tenant boundary — it must never be omitted (and the cast must stay: a str bind
is ::VARCHAR and Postgres has no `uuid = varchar` comparison operator). All
helpers take an explicit `session` (never a module global) so they're safe
under the agent loop's same-turn parallel tool fan-out; the dispatcher commits
after each tool, so nothing here commits.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .embeddings import embed_text, embeddings_enabled, to_pgvector_literal
from .memory import DECAY_HALF_LIFE_DAYS, RRF_K, SIGNAL_WEIGHTS

log = structlog.get_logger()

# Column list shared by the read paths. `not_stale` is computed server-side so
# freshness doesn't depend on the worker's clock vs the DB's.
_MEM_SELECT = (
    "id, mtype, scope, kind, entity_key, title, summary, record, aliases, tags, "
    "content_hash, source_url, source_type, pinned, importance, version, "
    "hit_count, source_job_id, skillpack_id, stale_at, deleted_at, "
    "deleted_reason, superseded_by, last_used_at, created_at, updated_at, "
    "(stale_at is null or stale_at > now()) as not_stale"
)

_MEM_INSERT_COLS = (
    "(workspace_id, mtype, scope, kind, entity_key, title, summary, record, "
    "aliases, tags, content_hash, source_url, source_type, pinned, importance, "
    "source_job_id, skillpack_id, stale_at, last_used_at)"
)
# uuid columns are cast explicitly: SQLAlchemy's asyncpg dialect tags a Python
# str bind as ::VARCHAR, which a bare `uuid` column rejects
# (DatatypeMismatchError). Casting lets us bind ids as strings uniformly.
_MEM_INSERT_VALS = (
    "(cast(:ws as uuid), :mtype, :scope, :kind, :ekey, :title, :summary, "
    "cast(:record as jsonb), :aliases, :tags, :chash, :surl, :stype, :pinned, "
    "coalesce(cast(:importance as real), 0.5), cast(:sjob as uuid), "
    "cast(:spack as uuid), cast(:stale as timestamptz), now())"
)

# The FTS expression must stay byte-identical to workspace_memory_fts_idx
# (migration 032) or the GIN index is dead weight.
_FTS_EXPR = (
    "to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(summary, '') "
    "|| ' ' || coalesce(kind, '') || ' ' || coalesce(entity_key, ''))"
)

# Recency half-life per memory type (days) — events fade fastest, rules
# slowest. The floor keeps an old-but-only match retrievable (decay shapes
# ranking; the maintenance pass does the actual forgetting).
_DECAY_CASE = (
    "(case mtype "
    + " ".join(f"when '{k}' then {v}" for k, v in DECAY_HALF_LIFE_DAYS.items())
    + " else 60.0 end)"
)
_SCORE_EXPR = (
    "fused.rrf"
    " * (0.35 + 0.65 * exp(-extract(epoch from (now() - greatest(m.updated_at,"
    " coalesce(m.last_used_at, m.updated_at)))) / 86400.0 / " + _DECAY_CASE.replace("mtype", "m.mtype") + "))"
    " * (0.5 + coalesce(m.importance, 0.5))"
    " * (case when m.pinned then 1.15 else 1.0 end)"
)


def _memory_row(r) -> dict:
    """Map a workspace_memory row (RowMapping) to a plain JSON-safe dict.
    `record` comes back as a dict (SQLAlchemy/asyncpg decode jsonb) but we
    tolerate a str just in case."""
    rec = r["record"]
    if isinstance(rec, str):
        try:
            rec = json.loads(rec)
        except (ValueError, TypeError):
            pass
    keys = r.keys()
    not_stale = r["not_stale"] if "not_stale" in keys else None

    def _iso(col):
        v = r[col] if col in keys else None
        return v.isoformat() if v is not None else None

    out = {
        "id": str(r["id"]),
        "mtype": r["mtype"],
        "scope": r["scope"],
        "kind": r["kind"],
        "entity_key": r["entity_key"],
        "title": r["title"],
        "summary": r["summary"] if "summary" in keys else None,
        "record": rec,
        "aliases": list(r["aliases"] or []),
        "tags": list(r["tags"] or []) if "tags" in keys else [],
        "content_hash": r["content_hash"],
        "source_url": r["source_url"],
        "source_type": r["source_type"] if "source_type" in keys else None,
        "pinned": r["pinned"],
        "importance": r["importance"],
        "version": r["version"],
        "hit_count": r["hit_count"],
        "source_job_id": str(r["source_job_id"]) if r["source_job_id"] else None,
        "skillpack_id": str(r["skillpack_id"]) if r["skillpack_id"] else None,
        "stale_at": _iso("stale_at"),
        "superseded_by": (
            str(r["superseded_by"])
            if "superseded_by" in keys and r["superseded_by"]
            else None
        ),
        "last_used_at": _iso("last_used_at"),
        "created_at": _iso("created_at"),
        "updated_at": _iso("updated_at"),
        "fresh": bool(not_stale) if not_stale is not None else None,
    }
    if "score" in keys and r["score"] is not None:
        out["score"] = round(float(r["score"]), 6)
    return out


# Probed once per process: migration 032 adds `embedding` only where pgvector
# exists, so the semantic branch must check before referencing the column.
_HAS_EMBEDDING_COL: bool | None = None


async def _has_embedding_column(session: AsyncSession) -> bool:
    global _HAS_EMBEDDING_COL
    if _HAS_EMBEDDING_COL is None:
        try:
            # Savepoint so a failed probe can't leave the caller's transaction
            # aborted (a swallowed SQL error poisons every later statement).
            async with session.begin_nested():
                row = (
                    await session.execute(
                        text(
                            "select 1 from information_schema.columns "
                            "where table_schema = 'public' "
                            "and table_name = 'workspace_memory' "
                            "and column_name = 'embedding'"
                        )
                    )
                ).first()
            _HAS_EMBEDDING_COL = row is not None
        except Exception:
            return False  # don't cache a transient failure
    return _HAS_EMBEDDING_COL


async def memory_search(
    session: AsyncSession,
    workspace_id,
    *,
    kind: str | None = None,
    entity_key: str | None = None,
    query: str | None = None,
    content_hash: str | None = None,
    mtype: str | None = None,
    scope: str | None = None,
    limit: int = 8,
) -> list[dict]:
    """Hybrid search over active (non-deleted) memory.

    Builds only the candidate CTEs whose inputs are present, RRF-fuses them and
    applies the recency × importance shaping. With no identity/query inputs at
    all it falls back to a freshness-ordered browse of the (filtered) store.
    A result's `fresh` flag is True iff it isn't past `stale_at` AND — when a
    `content_hash` was supplied — the stored hash matches (inputs unchanged).
    """
    params: dict[str, Any] = {
        "ws": str(workspace_id),
        "lim": max(1, min(int(limit or 8), 50)),
    }
    # `cast(:ws as uuid)` everywhere a uuid column is COMPARED: asyncpg tags a
    # str bind ::VARCHAR and Postgres has no `uuid = varchar` operator (it only
    # implicit-casts in assignment context, i.e. INSERT values). A bare compare
    # works or breaks depending on whether the caller passed UUID or str —
    # casting in SQL makes the module type-agnostic.
    base = ["workspace_id = cast(:ws as uuid)", "deleted_at is null"]
    if kind:
        base.append("kind = :kind")
        params["kind"] = kind
    if mtype:
        base.append("mtype = :mtype")
        params["mtype"] = mtype
    if scope:
        base.append("scope = :scope")
        params["scope"] = scope
    base_where = " and ".join(base)

    ctes: list[str] = []
    arms: list[str] = []

    if entity_key or content_hash:
        ident: list[str] = []
        if entity_key:
            ident += ["entity_key = :ekey", ":ekey = any(aliases)"]
            params["ekey"] = entity_key
        if content_hash:
            ident.append("content_hash = :chash")
            params["chash"] = content_hash
        ctes.append(
            "exact_hits as ("
            f"select id, row_number() over (order by (stale_at is null or stale_at > now()) desc, updated_at desc) as rnk "
            f"from workspace_memory where {base_where} and ({' or '.join(ident)}) "
            "limit 50)"
        )
        arms.append(
            f"select id, rnk, {SIGNAL_WEIGHTS['exact']} as w from exact_hits"
        )

    if query and query.strip():
        params["q"] = query.strip()
        params["likeq"] = f"%{query.strip()}%"
        ctes.append(
            "fts_hits as ("
            f"select id, row_number() over (order by ts_rank({_FTS_EXPR}, websearch_to_tsquery('simple', :q)) desc, updated_at desc) as rnk "
            f"from workspace_memory where {base_where} and "
            f"({_FTS_EXPR} @@ websearch_to_tsquery('simple', :q) "
            "or title ilike :likeq or summary ilike :likeq or :likeq = any(tags)) "
            "limit 50)"
        )
        arms.append(f"select id, rnk, {SIGNAL_WEIGHTS['fts']} as w from fts_hits")

        # Semantic branch — only when the column exists and a provider is
        # configured; embed failures silently drop the arm.
        if embeddings_enabled() and await _has_embedding_column(session):
            qvec = await embed_text(query.strip())
            if qvec is not None:
                params["qvec"] = to_pgvector_literal(qvec)
                ctes.append(
                    "vec_hits as ("
                    "select id, row_number() over (order by embedding <=> cast(:qvec as vector)) as rnk "
                    f"from workspace_memory where {base_where} and embedding is not null "
                    "order by embedding <=> cast(:qvec as vector) limit 50)"
                )
                arms.append(
                    f"select id, rnk, {SIGNAL_WEIGHTS['vector']} as w from vec_hits"
                )

    if not arms:
        # Browse mode: no identity and no query — freshest, weightiest first.
        sql = (
            f"select {_MEM_SELECT}, null::float as score from workspace_memory "
            f"where {base_where} "
            "order by (stale_at is null or stale_at > now()) desc, "
            "coalesce(importance, 0.5) desc, updated_at desc limit :lim"
        )
    else:
        sql = (
            "with " + ", ".join(ctes) + ", "
            f"fused as (select id, sum(w / ({RRF_K} + rnk)) as rrf "
            f"from ({' union all '.join(arms)}) signals group by id) "
            f"select {_sel_m()}, {_SCORE_EXPR} as score "
            "from fused join workspace_memory m on m.id = fused.id "
            "order by score desc limit :lim"
        )

    rows = (await session.execute(text(sql).bindparams(**params))).mappings().all()
    out = [_memory_row(r) for r in rows]
    # Refine `fresh` against the queried content_hash (inputs-changed check).
    if content_hash:
        for d in out:
            d["fresh"] = bool(d.get("fresh")) and (
                d.get("content_hash") is None or d.get("content_hash") == content_hash
            )
    return out


def _sel_m() -> str:
    """_MEM_SELECT with every column qualified `m.` for the fused join."""
    cols = [c.strip() for c in _MEM_SELECT.rsplit(",", 1)[0].split(",")]
    qualified = ", ".join(f"m.{c}" for c in cols)
    return qualified + ", (m.stale_at is null or m.stale_at > now()) as not_stale"


async def memory_get(session: AsyncSession, workspace_id, mem_id) -> dict | None:
    """Fetch one memory by id (workspace-scoped, soft-deleted rows included so
    an injected/cited id never 404s mid-job) and bump its hit counter."""
    row = (
        await session.execute(
            text(
                f"select {_MEM_SELECT} from workspace_memory "
                "where id = cast(:id as uuid) and workspace_id = cast(:ws as uuid)"
            ).bindparams(id=str(mem_id), ws=str(workspace_id))
        )
    ).mappings().first()
    if row is None:
        return None
    await session.execute(
        text(
            "update workspace_memory set hit_count = hit_count + 1, "
            "last_used_at = now() where id = cast(:id as uuid) and workspace_id = cast(:ws as uuid)"
        ).bindparams(id=str(mem_id), ws=str(workspace_id))
    )
    return _memory_row(row)


async def memory_put(
    session: AsyncSession,
    workspace_id,
    *,
    kind: str,
    record: dict,
    entity_key: str | None = None,
    mtype: str = "semantic",
    scope: str = "entity",
    title: str | None = None,
    summary: str | None = None,
    tags: list[str] | None = None,
    content_hash: str | None = None,
    source_url: str | None = None,
    source_type: str = "agent",
    aliases: list[str] | None = None,
    pinned: bool = False,
    importance: float | None = None,
    supersedes=None,
    stale_at=None,
    source_job_id=None,
    skillpack_id=None,
) -> dict:
    """Upsert a keyed (semantic/procedural) record by (workspace, kind,
    entity_key) — bumping `version` — or append a new row for episodic /
    keyless writes. On upsert, omitted summary/tags/importance keep their
    stored values (so a partial update can't wipe write-time signals).

    `supersedes` soft-deletes another row (sets superseded_by → new id) — the
    conflict-resolution path when a record REPLACES one stored under a
    different key/kind, rather than upserting over the same key.

    Returns {id, version, created}. Embedding (when configured) is refreshed
    best-effort after the write.
    """
    if importance is not None:
        try:
            importance = min(1.0, max(0.0, float(importance)))
        except (TypeError, ValueError):
            importance = None
    if source_type not in ("agent", "user", "system"):
        source_type = "agent"
    params: dict[str, Any] = {
        "ws": str(workspace_id),
        "mtype": mtype,
        "scope": scope,
        "kind": kind,
        "ekey": entity_key,
        "title": title,
        "summary": (summary or None),
        "record": json.dumps(record, default=str),
        "aliases": list(aliases or []),
        "tags": list(tags or []),
        "chash": content_hash,
        "surl": source_url,
        "stype": source_type,
        "pinned": bool(pinned),
        "importance": importance,
        "sjob": str(source_job_id) if source_job_id else None,
        "spack": str(skillpack_id) if skillpack_id else None,
        "stale": stale_at,
    }
    keyed = mtype in ("semantic", "procedural") and bool(entity_key)
    if keyed:
        sql = (
            f"insert into workspace_memory {_MEM_INSERT_COLS} values {_MEM_INSERT_VALS} "
            "on conflict (workspace_id, kind, entity_key) "
            "where mtype in ('semantic','procedural') and entity_key is not null "
            "do update set "
            "  mtype = excluded.mtype, scope = excluded.scope, "
            "  title = coalesce(excluded.title, workspace_memory.title), "
            "  summary = coalesce(:summary, workspace_memory.summary), "
            "  record = excluded.record, "
            # Non-empty wins: a partial re-put can't wipe aliases/tags.
            "  aliases = case when cardinality(excluded.aliases) > 0 "
            "    then excluded.aliases else workspace_memory.aliases end, "
            "  tags = case when cardinality(excluded.tags) > 0 "
            "    then excluded.tags else workspace_memory.tags end, "
            "  content_hash = excluded.content_hash, "
            "  source_url = coalesce(excluded.source_url, workspace_memory.source_url), "
            "  source_type = excluded.source_type, "
            "  pinned = excluded.pinned, "
            "  importance = coalesce(cast(:importance as real), workspace_memory.importance), "
            "  source_job_id = coalesce(excluded.source_job_id, workspace_memory.source_job_id), "
            "  skillpack_id = excluded.skillpack_id, stale_at = excluded.stale_at, "
            # Re-putting a key resurrects a soft-deleted row: the new write IS
            # the current truth.
            "  deleted_at = null, deleted_reason = null, superseded_by = null, "
            "  last_used_at = now(), version = workspace_memory.version + 1, "
            "  updated_at = now() "
            "returning id, version, (xmax = 0) as created"
        )
    else:
        sql = (
            f"insert into workspace_memory {_MEM_INSERT_COLS} values {_MEM_INSERT_VALS} "
            "returning id, version, true as created"
        )
    row = (await session.execute(text(sql).bindparams(**params))).mappings().first()
    new_id = str(row["id"])

    # Conflict resolution: the new record replaces an old one stored under a
    # different key — soft-delete it, keeping the audit pointer (Zep pattern).
    # Best-effort statements below run inside a SAVEPOINT (begin_nested): a
    # plain try/except swallows the Python exception but leaves the Postgres
    # transaction ABORTED — every later statement on the session (the runner's
    # job_events insert, the commit itself) then fails with
    # InFailedSQLTransactionError and the whole put is rolled back. The
    # savepoint confines the failure to the optional statement.
    if supersedes:
        try:
            async with session.begin_nested():
                await session.execute(
                    text(
                        "update workspace_memory set deleted_at = now(), "
                        "deleted_reason = 'superseded', "
                        "superseded_by = cast(:new as uuid), updated_at = now() "
                        "where id = cast(:old as uuid) and workspace_id = cast(:ws as uuid) "
                        "and id <> cast(:new as uuid) and deleted_at is null"
                    ).bindparams(new=new_id, old=str(supersedes), ws=str(workspace_id))
                )
        except Exception as e:
            log.warning("memory_supersede_failed", error=repr(e), old=str(supersedes))

    # Best-effort embedding refresh — the row is fully usable without it.
    if embeddings_enabled() and await _has_embedding_column(session):
        emb_text = " — ".join(
            x for x in (title, summary or json.dumps(record, default=str)[:2000]) if x
        )
        vec = await embed_text(emb_text)
        if vec is not None:
            try:
                async with session.begin_nested():
                    await session.execute(
                        text(
                            "update workspace_memory set embedding = cast(:v as vector) "
                            "where id = cast(:id as uuid) and workspace_id = cast(:ws as uuid)"
                        ).bindparams(
                            v=to_pgvector_literal(vec), id=new_id, ws=str(workspace_id)
                        )
                    )
            except Exception as e:
                log.warning("memory_embed_write_failed", error=repr(e))

    return {"id": new_id, "version": row["version"], "created": bool(row["created"])}


async def memory_forget(
    session: AsyncSession, workspace_id, mem_id, *, reason: str | None = None
) -> bool:
    """Soft-delete one memory (agent-facing DELETE). The row survives for the
    audit window and is purged by the nightly maintenance pass. Returns True
    when an active row was marked."""
    row = (
        await session.execute(
            text(
                "update workspace_memory set deleted_at = now(), "
                "deleted_reason = :reason, updated_at = now() "
                "where id = cast(:id as uuid) and workspace_id = cast(:ws as uuid) "
                "and deleted_at is null returning id"
            ).bindparams(
                id=str(mem_id),
                ws=str(workspace_id),
                reason=(reason or "forgotten by agent")[:500],
            )
        )
    ).first()
    return row is not None


async def memory_context(
    session: AsyncSession,
    workspace_id,
    *,
    skillpack_id=None,
    entity_keys: list[str] | None = None,
    content_hashes: list[str] | None = None,
    pinned_limit: int = 8,
    entity_limit: int = 6,
) -> dict:
    """Selective retrieval for job-start injection — NOT the whole store:
      - pinned: workspace/skillpack `pinned` rows (preferences, brand kit),
        loaded regardless of inputs; procedural-first, importance-ranked.
      - entity: fresh rows whose entity_key / alias / content_hash matches the
        job's derived identity (semantic briefs first, then episodic events
        about the same subject).
    Bumps hit_count on entity matches. Returns {"pinned": [...], "entity": [...]}.
    """
    pinned_rows = (
        await session.execute(
            text(
                f"select {_MEM_SELECT} from workspace_memory "
                "where workspace_id = cast(:ws as uuid) and pinned and deleted_at is null "
                "  and (skillpack_id is null or skillpack_id = cast(:spack as uuid)) "
                "  and (stale_at is null or stale_at > now()) "
                "order by (mtype = 'procedural') desc, "
                "importance desc nulls last, updated_at desc limit :lim"
            ).bindparams(
                ws=str(workspace_id),
                spack=(str(skillpack_id) if skillpack_id else None),
                lim=pinned_limit,
            )
        )
    ).mappings().all()

    ekeys = [k for k in (entity_keys or []) if k]
    chashes = [h for h in (content_hashes or []) if h]
    entity_rows: list = []
    if ekeys or chashes:
        # Array params are passed as JSON and expanded via jsonb_array_elements_text
        # (the proven `cast(:x as jsonb)` bind), avoiding any asyncpg list-encoding
        # ambiguity. Empty arrays simply match nothing.
        entity_rows = (
            await session.execute(
                text(
                    f"select {_MEM_SELECT} from workspace_memory "
                    "where workspace_id = cast(:ws as uuid) and deleted_at is null "
                    "  and mtype in ('semantic', 'episodic') "
                    "  and (entity_key = any(select jsonb_array_elements_text(cast(:ekeys as jsonb))) "
                    "       or aliases && array(select jsonb_array_elements_text(cast(:ekeys as jsonb))) "
                    "       or content_hash = any(select jsonb_array_elements_text(cast(:chashes as jsonb)))) "
                    "  and (stale_at is null or stale_at > now()) "
                    "order by (mtype = 'semantic') desc, "
                    "importance desc nulls last, updated_at desc limit :lim"
                ).bindparams(
                    ws=str(workspace_id),
                    ekeys=json.dumps(ekeys),
                    chashes=json.dumps(chashes),
                    lim=entity_limit,
                )
            )
        ).mappings().all()
        hit_ids = [str(r["id"]) for r in entity_rows]
        if hit_ids:
            await session.execute(
                text(
                    "update workspace_memory set hit_count = hit_count + 1, "
                    "last_used_at = now() where workspace_id = cast(:ws as uuid) "
                    "and id::text = any(select jsonb_array_elements_text(cast(:ids as jsonb)))"
                ).bindparams(ws=str(workspace_id), ids=json.dumps(hit_ids))
            )

    return {
        "pinned": [_memory_row(r) for r in pinned_rows],
        "entity": [_memory_row(r) for r in entity_rows],
    }
