"""Local (offline) memory store — the SQLite mirror of `memory_store.py`.

`puras run --local` has no Postgres, so the hosted workspace "shared brain"
(`memory_store.py`, Postgres + pgvector) can't back it. This module is the
open-core local twin: the SAME agent-facing contract (`memory_search` /
`memory_get` / `memory_put` / `memory_forget` / `memory_context`) over a plain
on-disk **SQLite** file, so a skill that reads and writes workspace memory
behaves identically whether it runs hosted or on the user's own machine.

Design parity with the hosted store, with the two pieces that can't exist
offline cleanly dropped:

  * HYBRID RETRIEVAL — exact key/alias/content_hash + lexical (title / summary /
    kind / entity_key / tags) candidates, fused with the SAME Reciprocal Rank
    Fusion + recency-decay + importance shaping the hosted store uses. The
    ranking KNOBS (`SIGNAL_WEIGHTS`, `RRF_K`, `DECAY_HALF_LIFE_DAYS` via
    `recency_decay`) are imported from `memory.py` — the one source of truth both
    halves share — so local and hosted rank the same way. The pgvector semantic
    arm is the only thing missing: local runs are exact+lexical, exactly the
    documented "no embedding provider configured" hosted fallback.
  * SOFT DELETE / SUPERSEDENCE — rows are never hard-deleted by the agent
    (`deleted_at` + `superseded_by` + `deleted_reason`); reads filter
    `deleted_at IS NULL`. `maintenance()` is the offline twin of
    `workspace_memory_maintenance()` (expire stale episodic, decay unused
    importance, purge long-dead rows, cap episodic volume).
  * WRITE-TIME QUALITY — `summary` / `tags` / `importance` are written once by
    the agent; reads stay cheap.

Dependency discipline: this is on the OFFLINE import path, which
`tests/dry/test_local_import_isolation.py` proves carries none of the hosted
heavy stack (sqlalchemy / asyncpg / …). So this module uses ONLY the stdlib
`sqlite3` — never SQLAlchemy. Blocking calls run under `asyncio.to_thread` and
a single `asyncio.Lock` serializes access (one local user, same-turn parallel
tool fan-out shares one connection), mirroring the async signatures of the
hosted store so `agent_runner` can call either with the same `await`.

`workspace_id` is still threaded through and filtered on every statement so the
contract matches the hosted store exactly, even though a local run only ever
uses the one synthetic offline workspace.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from .memory import RRF_K, SIGNAL_WEIGHTS, recency_decay

log = structlog.get_logger()

# All persisted columns, in a fixed order so row→dict mapping is positional and
# matches the hosted `_memory_row` output keys verbatim.
_COLUMNS = (
    "id", "workspace_id", "mtype", "scope", "kind", "entity_key", "title",
    "summary", "record", "aliases", "tags", "content_hash", "source_url",
    "source_type", "pinned", "importance", "version", "hit_count",
    "source_job_id", "skillpack_id", "stale_at", "deleted_at", "deleted_reason",
    "superseded_by", "last_used_at", "created_at", "updated_at",
)

_CREATE_SQL = """
create table if not exists workspace_memory (
  id            text primary key,
  workspace_id  text not null,
  mtype         text not null default 'semantic',
  scope         text not null default 'entity',
  kind          text not null,
  entity_key    text,
  title         text,
  summary       text,
  record        text not null,            -- JSON object
  aliases       text not null default '[]',  -- JSON array
  tags          text not null default '[]',  -- JSON array
  content_hash  text,
  source_url    text,
  source_type   text not null default 'agent',
  pinned        integer not null default 0,
  importance    real,
  version       integer not null default 1,
  hit_count     integer not null default 0,
  source_job_id text,
  skillpack_id  text,
  stale_at      text,
  deleted_at    text,
  deleted_reason text,
  superseded_by text,
  last_used_at  text,
  created_at    text not null,
  updated_at    text not null
);
-- Upsert target for keyed records, partial so episodic/keyless rows don't
-- collide — mirrors workspace_memory_key_uq (migration 029).
create unique index if not exists workspace_memory_key_uq
  on workspace_memory (workspace_id, kind, entity_key)
  where mtype in ('semantic','procedural') and entity_key is not null;
create index if not exists workspace_memory_browse_idx
  on workspace_memory (workspace_id, updated_at desc);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt is not None else None


def _parse_dt(v: Any) -> datetime | None:
    if not v:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class LocalMemoryStore:
    """Connection + schema holder for one SQLite memory file. One instance per
    process (the offline runner builds it lazily on the LocalRunContext). All
    access is serialized through `_lock` and run off the event loop via
    `asyncio.to_thread`, so the same store is safe under the agent loop's
    same-turn parallel tool fan-out."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("pragma journal_mode=WAL")
            conn.execute("pragma foreign_keys=on")
            conn.executescript(_CREATE_SQL)
            conn.commit()
            self._conn = conn
            log.info("local_memory_db", path=str(self.db_path))
        return self._conn

    async def _run(self, fn):
        """Run a blocking `fn(conn)` under the lock on a worker thread."""
        async with self._lock:
            return await asyncio.to_thread(self._call, fn)

    def _call(self, fn):
        conn = self._connect()
        try:
            out = fn(conn)
            conn.commit()
            return out
        except Exception:
            conn.rollback()
            raise


# ── row mapping (output shape identical to memory_store._memory_row) ─────────


def _row_to_dict(r: sqlite3.Row, *, now: datetime, query_hash: str | None = None,
                 score: float | None = None) -> dict:
    stale = _parse_dt(r["stale_at"])
    not_stale = stale is None or stale > now
    fresh = bool(not_stale)
    if query_hash is not None:
        fresh = fresh and (r["content_hash"] is None or r["content_hash"] == query_hash)

    def _load(col: str) -> list:
        try:
            v = json.loads(r[col]) if r[col] else []
        except (ValueError, TypeError):
            v = []
        return list(v) if isinstance(v, list) else []

    rec = r["record"]
    if isinstance(rec, str):
        try:
            rec = json.loads(rec)
        except (ValueError, TypeError):
            pass

    out = {
        "id": r["id"],
        "mtype": r["mtype"],
        "scope": r["scope"],
        "kind": r["kind"],
        "entity_key": r["entity_key"],
        "title": r["title"],
        "summary": r["summary"],
        "record": rec,
        "aliases": _load("aliases"),
        "tags": _load("tags"),
        "content_hash": r["content_hash"],
        "source_url": r["source_url"],
        "source_type": r["source_type"],
        "pinned": bool(r["pinned"]),
        "importance": r["importance"],
        "version": r["version"],
        "hit_count": r["hit_count"],
        "source_job_id": r["source_job_id"],
        "skillpack_id": r["skillpack_id"],
        "stale_at": _iso(stale),
        "superseded_by": r["superseded_by"],
        "last_used_at": _iso(_parse_dt(r["last_used_at"])),
        "created_at": _iso(_parse_dt(r["created_at"])),
        "updated_at": _iso(_parse_dt(r["updated_at"])),
        "fresh": fresh,
    }
    if score is not None:
        out["score"] = round(float(score), 6)
    return out


def _age_days(r: sqlite3.Row, now: datetime) -> float:
    updated = _parse_dt(r["updated_at"]) or now
    last_used = _parse_dt(r["last_used_at"])
    ref = max(updated, last_used) if last_used else updated
    return max(0.0, (now - ref).total_seconds() / 86400.0)


def _haystack(r: sqlite3.Row) -> str:
    bits = [r["title"] or "", r["summary"] or "", r["kind"] or "", r["entity_key"] or ""]
    try:
        bits.extend(json.loads(r["tags"]) if r["tags"] else [])
    except (ValueError, TypeError):
        pass
    return " ".join(str(b) for b in bits).lower()


def _base_rows(conn: sqlite3.Connection, workspace_id, *, kind=None, mtype=None,
               scope=None) -> list[sqlite3.Row]:
    sql = (
        f"select {', '.join(_COLUMNS)} from workspace_memory "
        "where workspace_id = ? and deleted_at is null"
    )
    params: list[Any] = [str(workspace_id)]
    if kind:
        sql += " and kind = ?"
        params.append(kind)
    if mtype:
        sql += " and mtype = ?"
        params.append(mtype)
    if scope:
        sql += " and scope = ?"
        params.append(scope)
    sql += " order by updated_at desc limit 2000"
    return conn.execute(sql, params).fetchall()


# ── public API (mirrors memory_store.py signatures; `store` ↔ `session`) ─────


async def memory_search(
    store: LocalMemoryStore,
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
    """Hybrid search over active memory — exact identity + lexical candidates
    RRF-fused and shaped by recency × importance × pin, identical to the hosted
    store minus the pgvector arm. With no identity/query inputs it falls back to
    a freshness-ordered browse."""
    lim = max(1, min(int(limit or 8), 50))
    now = _now()

    def _do(conn: sqlite3.Connection) -> list[dict]:
        rows = _base_rows(conn, workspace_id, kind=kind, mtype=mtype, scope=scope)
        by_id = {r["id"]: r for r in rows}

        q = (query or "").strip()
        # Whether any candidate ARM was requested. Browse mode is for the
        # no-identity / no-query case only — a requested-but-unmatched arm must
        # return empty (the subject simply isn't in memory), never the whole
        # store, matching the hosted store's `not arms` fallback.
        has_arm = bool(entity_key or content_hash or q)

        signals: list[tuple[str, float]] = []  # (id, weight/(K+rank)) contributions

        # exact identity arm (weight 1.0) — entity_key / alias / content_hash.
        if entity_key or content_hash:
            def _is_exact(r: sqlite3.Row) -> bool:
                if entity_key and (r["entity_key"] == entity_key):
                    return True
                if entity_key:
                    try:
                        if entity_key in (json.loads(r["aliases"]) if r["aliases"] else []):
                            return True
                    except (ValueError, TypeError):
                        pass
                if content_hash and r["content_hash"] == content_hash:
                    return True
                return False

            exact = [r for r in rows if _is_exact(r)]
            exact.sort(
                key=lambda r: (
                    (_parse_dt(r["stale_at"]) is None or _parse_dt(r["stale_at"]) > now),
                    _parse_dt(r["updated_at"]) or now,
                ),
                reverse=True,
            )
            for rank, r in enumerate(exact[:50], start=1):
                signals.append((r["id"], SIGNAL_WEIGHTS["exact"] / (RRF_K + rank)))

        # lexical arm (weight 0.7) — tokens over title/summary/kind/entity_key/tags.
        if q:
            tokens = [t for t in q.lower().split() if t]
            ql = q.lower()
            scored: list[tuple[int, sqlite3.Row]] = []
            for r in rows:
                hay = _haystack(r)
                hits = sum(1 for t in tokens if t in hay)
                # whole-phrase match in title/summary, or a tag equal to the query
                if ql in (r["title"] or "").lower() or ql in (r["summary"] or "").lower():
                    hits += 1
                try:
                    if ql in [str(t).lower() for t in (json.loads(r["tags"]) if r["tags"] else [])]:
                        hits += 1
                except (ValueError, TypeError):
                    pass
                if hits:
                    scored.append((hits, r))
            scored.sort(
                key=lambda hr: (hr[0], _parse_dt(hr[1]["updated_at"]) or now),
                reverse=True,
            )
            for rank, (_, r) in enumerate(scored[:50], start=1):
                signals.append((r["id"], SIGNAL_WEIGHTS["fts"] / (RRF_K + rank)))

        if not has_arm:
            # Browse mode: no identity and no query — freshest, weightiest first.
            browse = sorted(
                rows,
                key=lambda r: (
                    (_parse_dt(r["stale_at"]) is None or _parse_dt(r["stale_at"]) > now),
                    (r["importance"] if r["importance"] is not None else 0.5),
                    _parse_dt(r["updated_at"]) or now,
                ),
                reverse=True,
            )[:lim]
            return [_row_to_dict(r, now=now, query_hash=content_hash) for r in browse]

        # Fuse the arms (sum of contributions per id), then shape.
        rrf: dict[str, float] = {}
        for mid, contrib in signals:
            rrf[mid] = rrf.get(mid, 0.0) + contrib

        ranked: list[tuple[float, sqlite3.Row]] = []
        for mid, base in rrf.items():
            r = by_id[mid]
            imp = r["importance"] if r["importance"] is not None else 0.5
            score = (
                base
                * recency_decay(_age_days(r, now), r["mtype"])
                * (0.5 + imp)
                * (1.15 if r["pinned"] else 1.0)
            )
            ranked.append((score, r))
        ranked.sort(key=lambda sr: sr[0], reverse=True)
        return [
            _row_to_dict(r, now=now, query_hash=content_hash, score=score)
            for score, r in ranked[:lim]
        ]

    return await store._run(_do)


async def memory_get(store: LocalMemoryStore, workspace_id, mem_id) -> dict | None:
    """Fetch one memory by id (workspace-scoped, soft-deleted rows included so a
    cited id never 404s mid-job) and bump its hit counter."""
    now = _now()

    def _do(conn: sqlite3.Connection) -> dict | None:
        r = conn.execute(
            f"select {', '.join(_COLUMNS)} from workspace_memory "
            "where id = ? and workspace_id = ?",
            (str(mem_id), str(workspace_id)),
        ).fetchone()
        if r is None:
            return None
        conn.execute(
            "update workspace_memory set hit_count = hit_count + 1, last_used_at = ? "
            "where id = ? and workspace_id = ?",
            (_iso(now), str(mem_id), str(workspace_id)),
        )
        return _row_to_dict(r, now=now)

    return await store._run(_do)


async def memory_put(
    store: LocalMemoryStore,
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
    entity_key) — bumping `version` — or append a new row for episodic/keyless
    writes. On upsert, omitted summary/tags/importance keep their stored values.
    `supersedes` soft-deletes another row. Returns {id, version, created}."""
    if importance is not None:
        try:
            importance = min(1.0, max(0.0, float(importance)))
        except (TypeError, ValueError):
            importance = None
    if source_type not in ("agent", "user", "system"):
        source_type = "agent"
    now = _now()
    now_iso = _iso(now)
    keyed = mtype in ("semantic", "procedural") and bool(entity_key)
    aliases = list(aliases or [])
    tags = list(tags or [])
    stale_iso = _iso(_parse_dt(stale_at))

    def _do(conn: sqlite3.Connection) -> dict:
        existing = None
        if keyed:
            existing = conn.execute(
                "select id, version, aliases, tags, title, summary, source_url, "
                "importance from workspace_memory where workspace_id = ? and kind = ? "
                "and entity_key = ? and mtype in ('semantic','procedural')",
                (str(workspace_id), kind, entity_key),
            ).fetchone()

        if existing is not None:
            # Non-empty wins: a partial re-put can't wipe aliases/tags; omitted
            # summary/title/source_url/importance keep their stored values.
            try:
                old_aliases = json.loads(existing["aliases"]) if existing["aliases"] else []
            except (ValueError, TypeError):
                old_aliases = []
            try:
                old_tags = json.loads(existing["tags"]) if existing["tags"] else []
            except (ValueError, TypeError):
                old_tags = []
            new_id = existing["id"]
            new_version = int(existing["version"]) + 1
            conn.execute(
                "update workspace_memory set mtype = ?, scope = ?, "
                "title = ?, summary = ?, record = ?, aliases = ?, tags = ?, "
                "content_hash = ?, source_url = ?, source_type = ?, pinned = ?, "
                "importance = ?, source_job_id = coalesce(?, source_job_id), "
                "skillpack_id = ?, stale_at = ?, deleted_at = null, "
                "deleted_reason = null, superseded_by = null, last_used_at = ?, "
                "version = ?, updated_at = ? where id = ?",
                (
                    mtype, scope,
                    (title if title is not None else existing["title"]),
                    (summary if summary is not None else existing["summary"]),
                    json.dumps(record, default=str),
                    json.dumps(aliases if aliases else old_aliases),
                    json.dumps(tags if tags else old_tags),
                    content_hash,
                    (source_url if source_url is not None else existing["source_url"]),
                    source_type, 1 if pinned else 0,
                    (importance if importance is not None else existing["importance"]),
                    (str(source_job_id) if source_job_id else None),
                    (str(skillpack_id) if skillpack_id else None),
                    stale_iso, now_iso, new_version, now_iso, new_id,
                ),
            )
            created = False
        else:
            new_id = str(uuid.uuid4())
            new_version = 1
            conn.execute(
                f"insert into workspace_memory ({', '.join(_COLUMNS)}) "
                f"values ({', '.join('?' for _ in _COLUMNS)})",
                (
                    new_id, str(workspace_id), mtype, scope, kind, entity_key,
                    title, summary, json.dumps(record, default=str),
                    json.dumps(aliases), json.dumps(tags), content_hash, source_url,
                    source_type, 1 if pinned else 0,
                    (importance if importance is not None else 0.5),
                    new_version, 0,
                    (str(source_job_id) if source_job_id else None),
                    (str(skillpack_id) if skillpack_id else None),
                    stale_iso, None, None, None, now_iso, now_iso, now_iso,
                ),
            )
            created = True

        # Conflict resolution: the new record replaces one stored elsewhere.
        if supersedes:
            conn.execute(
                "update workspace_memory set deleted_at = ?, "
                "deleted_reason = 'superseded', superseded_by = ?, updated_at = ? "
                "where id = ? and workspace_id = ? and id <> ? and deleted_at is null",
                (now_iso, new_id, now_iso, str(supersedes), str(workspace_id), new_id),
            )

        return {"id": new_id, "version": new_version, "created": created}

    return await store._run(_do)


async def memory_forget(
    store: LocalMemoryStore, workspace_id, mem_id, *, reason: str | None = None
) -> bool:
    """Soft-delete one memory (agent-facing DELETE). Returns True when an active
    row was marked."""
    now_iso = _iso(_now())

    def _do(conn: sqlite3.Connection) -> bool:
        cur = conn.execute(
            "update workspace_memory set deleted_at = ?, deleted_reason = ?, "
            "updated_at = ? where id = ? and workspace_id = ? and deleted_at is null",
            (now_iso, (reason or "forgotten by agent")[:500], now_iso,
             str(mem_id), str(workspace_id)),
        )
        return cur.rowcount > 0

    return await store._run(_do)


async def memory_context(
    store: LocalMemoryStore,
    workspace_id,
    *,
    skillpack_id=None,
    entity_keys: list[str] | None = None,
    content_hashes: list[str] | None = None,
    pinned_limit: int = 8,
    entity_limit: int = 6,
) -> dict:
    """Selective retrieval for job-start injection — pinned preferences (always)
    + fresh entity matches for the job's derived identity. Bumps hit_count on
    entity matches. Returns {"pinned": [...], "entity": [...]}."""
    now = _now()
    ekeys = [k for k in (entity_keys or []) if k]
    chashes = [h for h in (content_hashes or []) if h]

    def _do(conn: sqlite3.Connection) -> dict:
        rows = _base_rows(conn, workspace_id)

        def _fresh(r: sqlite3.Row) -> bool:
            stale = _parse_dt(r["stale_at"])
            return stale is None or stale > now

        # pinned: workspace/skillpack pinned rows; procedural-first, importance-ranked.
        pinned = [
            r for r in rows
            if r["pinned"] and _fresh(r)
            and (r["skillpack_id"] is None
                 or (skillpack_id and r["skillpack_id"] == str(skillpack_id)))
        ]
        pinned.sort(
            key=lambda r: (
                r["mtype"] == "procedural",
                (r["importance"] if r["importance"] is not None else -1),
                _parse_dt(r["updated_at"]) or now,
            ),
            reverse=True,
        )
        pinned = pinned[:pinned_limit]

        entity: list[sqlite3.Row] = []
        if ekeys or chashes:
            ekset = set(ekeys)
            chset = set(chashes)

            def _matches(r: sqlite3.Row) -> bool:
                if r["mtype"] not in ("semantic", "episodic"):
                    return False
                if not _fresh(r):
                    return False
                if r["entity_key"] and r["entity_key"] in ekset:
                    return True
                if r["content_hash"] and r["content_hash"] in chset:
                    return True
                try:
                    al = json.loads(r["aliases"]) if r["aliases"] else []
                except (ValueError, TypeError):
                    al = []
                return bool(ekset.intersection(al))

            entity = [r for r in rows if _matches(r)]
            entity.sort(
                key=lambda r: (
                    r["mtype"] == "semantic",
                    (r["importance"] if r["importance"] is not None else -1),
                    _parse_dt(r["updated_at"]) or now,
                ),
                reverse=True,
            )
            entity = entity[:entity_limit]
            hit_ids = [r["id"] for r in entity]
            if hit_ids:
                conn.executemany(
                    "update workspace_memory set hit_count = hit_count + 1, "
                    "last_used_at = ? where workspace_id = ? and id = ?",
                    [(_iso(now), str(workspace_id), mid) for mid in hit_ids],
                )

        return {
            "pinned": [_row_to_dict(r, now=now) for r in pinned],
            "entity": [_row_to_dict(r, now=now) for r in entity],
        }

    return await store._run(_do)


async def maintenance(store: LocalMemoryStore, workspace_id=None) -> dict:
    """Offline twin of `workspace_memory_maintenance()` — the cheap forgetting
    pass. Expires old unused episodic rows, decays their importance, purges
    long-dead soft-deleted rows, and caps unpinned episodic volume. Not
    scheduled offline (there is no cron); callable from a tool or a test."""
    now = _now()

    def _do(conn: sqlite3.Connection) -> dict:
        ws_clause = " and workspace_id = ?" if workspace_id else ""
        ws_params: tuple = (str(workspace_id),) if workspace_id else ()
        # ISO timestamps sort lexically, so a string compare is a date compare.
        from datetime import timedelta
        d90 = _iso(now - timedelta(days=90))
        d30 = _iso(now - timedelta(days=30))

        expired = conn.execute(
            "update workspace_memory set deleted_at = ?, "
            "deleted_reason = 'expired: episodic ttl' "
            "where mtype = 'episodic' and pinned = 0 and deleted_at is null "
            "and created_at < ? and coalesce(last_used_at, created_at) < ?"
            + ws_clause,
            (_iso(now), d90, d30, *ws_params),
        ).rowcount

        decayed = conn.execute(
            "update workspace_memory set importance = "
            "max(coalesce(importance, 0.5) * 0.98, 0.05) "
            "where mtype = 'episodic' and pinned = 0 and deleted_at is null "
            "and coalesce(last_used_at, created_at) < ?" + ws_clause,
            (d30, *ws_params),
        ).rowcount

        purged = conn.execute(
            "delete from workspace_memory where deleted_at is not null "
            "and deleted_at < ?" + ws_clause,
            (d30, *ws_params),
        ).rowcount

        return {"expired_episodic": expired, "decayed": decayed, "purged": purged}

    return await store._run(_do)
