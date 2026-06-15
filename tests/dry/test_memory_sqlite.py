"""Dry tests for the LOCAL (offline) workspace-memory store — the SQLite mirror
of the hosted Postgres `memory_store.py` that backs `puras run --local`.

DB-free in the sense that it touches no platform: a throwaway on-disk SQLite
file in tmp_path, the stdlib `sqlite3` only. Covers the agent-facing contract
(put/get/search/forget/context), the hybrid-retrieval shaping, soft-delete /
supersede, and the tool-gating that makes memory the ONE hosted capability that
stays on offline.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from worker import agent_runner
from worker import memory_store_sqlite as store


WS = "00000000-0000-0000-0000-000000000000"


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def mem(tmp_path):
    return store.LocalMemoryStore(tmp_path / "memory.db")


# ── put / get ───────────────────────────────────────────────────────────────
def test_keyed_put_upserts_and_bumps_version(mem):
    async def go():
        r1 = await store.memory_put(
            mem, WS, kind="product_brief", record={"color": "black"},
            entity_key="url:acme.com/mug", summary="Acme mug", tags=["mug"],
            importance=0.7,
        )
        assert r1["created"] and r1["version"] == 1
        # Same kind+key upserts in place: version bumps, id stable, created False.
        r2 = await store.memory_put(
            mem, WS, kind="product_brief", record={"color": "navy"},
            entity_key="url:acme.com/mug",
        )
        assert not r2["created"] and r2["version"] == 2 and r2["id"] == r1["id"]
        got = await store.memory_get(mem, WS, r1["id"])
        # Partial re-put keeps the stored summary/tags (write-time signals).
        assert got["summary"] == "Acme mug" and got["tags"] == ["mug"]
        assert got["record"]["color"] == "navy"
    _run(go())


def test_episodic_put_appends_not_upserts(mem):
    async def go():
        a = await store.memory_put(mem, WS, kind="event", record={"n": 1}, mtype="episodic")
        b = await store.memory_put(mem, WS, kind="event", record={"n": 2}, mtype="episodic")
        assert a["id"] != b["id"] and a["created"] and b["created"]
    _run(go())


def test_get_bumps_hit_count(mem):
    async def go():
        r = await store.memory_put(mem, WS, kind="k", record={"x": 1}, entity_key="e")
        first = await store.memory_get(mem, WS, r["id"])
        assert first["hit_count"] == 0  # value as read, before this call's bump
        second = await store.memory_get(mem, WS, r["id"])
        assert second["hit_count"] == 1
    _run(go())


def test_importance_clamped_to_unit_interval(mem):
    async def go():
        r = await store.memory_put(mem, WS, kind="k", record={"x": 1}, entity_key="e", importance=9.0)
        assert (await store.memory_get(mem, WS, r["id"]))["importance"] == 1.0
    _run(go())


# ── search ───────────────────────────────────────────────────────────────────
def test_exact_search_freshness_tracks_content_hash(mem):
    async def go():
        await store.memory_put(
            mem, WS, kind="product_brief", record={"x": 1},
            entity_key="e1", content_hash="fp:new",
        )
        hit = (await store.memory_search(mem, WS, entity_key="e1", content_hash="fp:new"))[0]
        assert hit["fresh"] is True
        # A different fingerprint = inputs changed = not fresh.
        stale = (await store.memory_search(mem, WS, entity_key="e1", content_hash="fp:old"))[0]
        assert stale["fresh"] is False
    _run(go())


def test_search_matches_alias_and_lexical(mem):
    async def go():
        r = await store.memory_put(
            mem, WS, kind="product_brief", record={"x": 1}, entity_key="e1",
            aliases=["acme-mug"], summary="Ceramic coffee mug by Acme", tags=["mug"],
        )
        assert (await store.memory_search(mem, WS, entity_key="acme-mug"))[0]["id"] == r["id"]
        assert any(x["id"] == r["id"] for x in await store.memory_search(mem, WS, query="coffee mug"))
    _run(go())


def test_requested_but_unmatched_arm_returns_empty_not_browse(mem):
    """A key/query that matches nothing must return [], never fall back to a
    browse of the whole store (that's the no-input case only)."""
    async def go():
        await store.memory_put(mem, WS, kind="k", record={"x": 1}, entity_key="e1")
        assert await store.memory_search(mem, WS, entity_key="nope") == []
        assert await store.memory_search(mem, WS, query="zzzznomatch") == []
        # No arm at all → browse mode returns the store.
        assert await store.memory_search(mem, WS)
    _run(go())


# ── forget / supersede ───────────────────────────────────────────────────────
def test_forget_soft_deletes_and_hides_from_search(mem):
    async def go():
        r = await store.memory_put(mem, WS, kind="k", record={"x": 1}, entity_key="e1")
        assert await store.memory_forget(mem, WS, r["id"], reason="wrong")
        assert await store.memory_search(mem, WS, entity_key="e1") == []
        # A cited id still resolves (soft-deleted rows survive for memory_get).
        assert await store.memory_get(mem, WS, r["id"]) is not None
        # Forgetting again is a no-op (no active row).
        assert await store.memory_forget(mem, WS, r["id"]) is False
    _run(go())


def test_supersede_soft_deletes_the_old_record(mem):
    async def go():
        old = await store.memory_put(mem, WS, kind="research", record={"a": 1}, entity_key="k:a")
        await store.memory_put(
            mem, WS, kind="research", record={"b": 2}, entity_key="k:b",
            supersedes=old["id"],
        )
        assert await store.memory_search(mem, WS, entity_key="k:a") == []
    _run(go())


# ── context (job-start injection) ────────────────────────────────────────────
def test_context_returns_pinned_and_entity_matches(mem):
    async def go():
        pin = await store.memory_put(
            mem, WS, kind="user_preference", record={"tone": "formal"},
            entity_key="pref:tone", mtype="procedural", scope="workspace",
            pinned=True, importance=0.9,
        )
        ent = await store.memory_put(
            mem, WS, kind="product_brief", record={"x": 1},
            entity_key="url:acme.com/mug", content_hash="fp:1",
        )
        ctx = await store.memory_context(
            mem, WS, entity_keys=["url:acme.com/mug"], content_hashes=["fp:1"],
        )
        assert any(p["id"] == pin["id"] for p in ctx["pinned"])
        assert any(e["id"] == ent["id"] for e in ctx["entity"])
    _run(go())


def test_context_workspace_isolation(mem):
    async def go():
        await store.memory_put(mem, "ws-A", kind="k", record={"x": 1}, entity_key="e", pinned=True)
        ctx = await store.memory_context(mem, "ws-B")
        assert ctx["pinned"] == [] and ctx["entity"] == []
    _run(go())


# ── tool gating: memory is the one hosted capability that stays on offline ────
def test_memory_tools_are_not_platform_only():
    # Splitting them out of PLATFORM_ONLY_TOOLS is what keeps them available on
    # a local run; media/web/drive helpers must stay platform-only.
    assert not (agent_runner.MEMORY_TOOLS & agent_runner.PLATFORM_ONLY_TOOLS)
    for name in ("generate_image", "web_search", "drive_pull"):
        assert name in agent_runner.PLATFORM_ONLY_TOOLS


def _stub_skill():
    from worker.skill_loader import LoadedSkill
    return LoadedSkill(
        name="t", root=Path("."), is_agentic=True,
        input_schema={"type": "object"}, output_schema={"type": "object"},
        system_prompt="x",
    )


def test_local_build_tools_offers_memory_but_not_media():
    # platform_enabled False (local), memory_enabled True (SQLite-backed).
    tools, _ = agent_runner._build_tools(
        _stub_skill(), platform_enabled=False, memory_enabled=True
    )
    names = {t["name"] for t in tools}
    assert {"memory_search", "memory_put", "memory_get", "memory_forget"} <= names
    assert "generate_image" not in names and "web_search" not in names


def test_memory_can_be_disabled_entirely():
    tools, _ = agent_runner._build_tools(
        _stub_skill(), platform_enabled=False, memory_enabled=False
    )
    names = {t["name"] for t in tools}
    assert not (agent_runner.MEMORY_TOOLS & names)
