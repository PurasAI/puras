"""Local web tools (no platform): web_fetch over HTTP, web_search via Anthropic.

Covers the seams the feature adds:
  - `_build_tools` keeps web_search/web_fetch offline ONLY when web is enabled;
  - `_local_web_enabled` is the local-run gate (always on offline);
  - `web_local.fetch` reduces HTML to readable text via a plain HTTP GET;
  - `web_local.search` drives Anthropic's server-side web_search tool (BYO key)
    and pulls the result list out of the response.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from worker import agent_runner, web_local
from worker.config import get_settings


@pytest.fixture(autouse=True)
def _hosted_env_placeholders(monkeypatch):
    """Fill the hosted-required env so `get_settings()` is loadable in dry tests
    (mirrors local_run._prepare_env)."""
    for name in (
        "DATABASE_URL", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
        "PURAS_SERVICE_TOKEN", "ANTHROPIC_API_KEY",
    ):
        monkeypatch.setenv(name, "x")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@dataclass
class _Skill:
    name: str = "demo"
    tools: list = field(default_factory=list)
    disable_bash: bool = False
    is_adhoc: bool = False
    output_schema: Any = None
    allowed_tools: Any = None


_WEB = {"web_search", "web_fetch"}


def _tool_names(skill, **kw):
    tools, _ = agent_runner._build_tools(skill, **kw)
    return {t["name"] for t in tools}


class _LocalCtx:
    platform_enabled = False


class _HostedCtx:
    platform_enabled = True


def test_web_tools_dropped_offline_without_web():
    names = _tool_names(_Skill(), platform_enabled=False, web_enabled=False)
    assert not (names & _WEB), names


def test_web_tools_kept_offline_with_web():
    names = _tool_names(_Skill(), platform_enabled=False, web_enabled=True)
    assert _WEB <= names, names
    # Only search + fetch are re-enabled — image_search / screenshot stay off.
    assert "image_search" not in names
    assert "web_screenshot" not in names


def test_web_tools_present_hosted():
    names = _tool_names(_Skill(), platform_enabled=True)
    assert _WEB <= names


def test_local_web_enabled_gate():
    assert agent_runner._local_web_enabled(_LocalCtx()) is True
    assert agent_runner._local_web_enabled(_HostedCtx()) is False


# ── web_fetch: plain HTTP GET + HTML→text ────────────────────────────────────
def _fake_httpx_client(monkeypatch, *, body: str, content_type: str, url: str):
    import httpx

    class _Resp:
        def __init__(self):
            self.text = body
            self.headers = {"content-type": content_type}
            self.url = url

        def raise_for_status(self):
            return None

    class _Client:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, _url):
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)


def test_fetch_extracts_text_and_strips_scripts(monkeypatch):
    _fake_httpx_client(
        monkeypatch,
        body=(
            "<html><head><title>Hello</title><style>.x{color:red}</style></head>"
            "<body><h1>Headline</h1><p>First paragraph.</p>"
            "<script>evil()</script></body></html>"
        ),
        content_type="text/html; charset=utf-8",
        url="https://example.com/page",
    )
    out = web_local.fetch("https://example.com/page", max_chars=5000)
    assert out["title"] == "Hello"
    assert "Headline" in out["content"]
    assert "First paragraph." in out["content"]
    assert "evil()" not in out["content"]  # script content stripped
    assert ".x{color:red}" not in out["content"]  # style content stripped
    assert out["rendered"] is False
    assert out["billed_micros"] == 0
    assert out["length"] == len(out["content"])


def test_fetch_truncates_to_max_chars(monkeypatch):
    _fake_httpx_client(
        monkeypatch,
        body="<html><body>" + ("x" * 5000) + "</body></html>",
        content_type="text/html",
        url="https://example.com/big",
    )
    out = web_local.fetch("https://example.com/big", max_chars=500)
    assert out["truncated"] is True
    assert out["content"].endswith("…[truncated]")


def test_fetch_rejects_non_http_url():
    with pytest.raises(web_local.LocalWebError):
        web_local.fetch("ftp://example.com/x")


# ── web_search: Anthropic server-side web_search tool ────────────────────────
def _fake_anthropic(captured: dict, blocks):
    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kw):
            captured.update(kw)
            return SimpleNamespace(content=blocks)

    class _Anthropic:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.messages = _Messages()

    mod.Anthropic = _Anthropic
    return mod


def test_search_drives_anthropic_web_search(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    result_block = SimpleNamespace(
        type="web_search_tool_result",
        content=[
            SimpleNamespace(
                type="web_search_result",
                title="Example",
                url="https://example.com/a",
                page_age="2 days",
            ),
            SimpleNamespace(
                type="web_search_result", title="Two", url="https://example.com/b",
            ),
        ],
    )
    captured: dict = {}
    monkeypatch.setitem(
        sys.modules, "anthropic", _fake_anthropic(captured, [result_block])
    )

    out = web_local.search("python testing", max_results=5)
    assert captured["tools"][0]["type"] == "web_search_20250305"
    assert captured["api_key"] == "sk-test"
    assert out["query"] == "python testing"
    assert out["results"][0] == {
        "title": "Example", "url": "https://example.com/a", "page_age": "2 days",
    }
    assert out["results"][1]["url"] == "https://example.com/b"
    get_settings.cache_clear()


def test_search_respects_max_results(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    items = [
        SimpleNamespace(type="web_search_result", title=f"r{i}", url=f"https://e/{i}")
        for i in range(10)
    ]
    block = SimpleNamespace(type="web_search_tool_result", content=items)
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic({}, [block]))
    out = web_local.search("q", max_results=3)
    assert len(out["results"]) == 3
    get_settings.cache_clear()


def test_search_surfaces_tool_error(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    err = SimpleNamespace(type="web_search_tool_result_error", error_code="max_uses_exceeded")
    block = SimpleNamespace(type="web_search_tool_result", content=err)
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic({}, [block]))
    with pytest.raises(web_local.LocalWebError):
        web_local.search("q")
    get_settings.cache_clear()


def test_search_requires_anthropic_key(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # present but empty
    with pytest.raises(web_local.LocalWebError):
        web_local.search("q")
    get_settings.cache_clear()
