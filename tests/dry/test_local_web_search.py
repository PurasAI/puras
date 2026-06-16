"""Dry tests for the LOCAL (offline) `web_search` tool — backed by Anthropic's
native server-side web_search instead of the platform `/v1/web` endpoint.

DB-free and network-free: `httpx.post` is monkeypatched with a canned Anthropic
Messages response so we only exercise the wire-format reshape in
`agent_runner._anthropic_web_search` (the part most likely to drift if the API
shape changes) and the model-pick helper.
"""

from __future__ import annotations

import httpx
import pytest

from worker import agent_runner


class _FakeResponse:
    def __init__(self, payload, *, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    @property
    def is_success(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


# A trimmed copy of the documented web_search response: a search query, a result
# block with two hits, and the model's cited summary text.
_OK_PAYLOAD = {
    "content": [
        {"type": "text", "text": "I'll search for that."},
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "claude shannon birth date"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://en.wikipedia.org/wiki/Claude_Shannon",
                    "title": "Claude Shannon - Wikipedia",
                    "encrypted_content": "Eqgf...",
                    "page_age": "April 30, 2025",
                },
                {
                    "type": "web_search_result",
                    "url": "https://example.com/shannon",
                    "title": "Shannon bio",
                    "page_age": None,
                },
            ],
        },
        {"type": "text", "text": "Claude Shannon was born April 30, 1916."},
    ],
    "stop_reason": "end_turn",
}


def test_web_search_reshapes_results(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse(_OK_PAYLOAD)

    monkeypatch.setattr(httpx, "post", fake_post)

    out = agent_runner._anthropic_web_search(
        "claude shannon birth date", model_id="claude-sonnet-4-6"
    )

    # The request used the native server-side web_search tool on the right model.
    assert captured["url"].endswith("/v1/messages")
    assert captured["json"]["model"] == "claude-sonnet-4-6"
    assert captured["json"]["tools"][0]["type"] == "web_search_20250305"

    # Results are reshaped to the hosted {title, url, page_age} contract.
    assert out["ok"] is True
    assert out["billed_micros"] == 0
    assert [r["url"] for r in out["results"]] == [
        "https://en.wikipedia.org/wiki/Claude_Shannon",
        "https://example.com/shannon",
    ]
    assert out["results"][0]["title"] == "Claude Shannon - Wikipedia"
    # The model's text blocks are joined into a summary; the search-decision and
    # final-answer text both land there.
    assert "Claude Shannon was born April 30, 1916." in out["summary"]


def test_web_search_honours_max_results(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(_OK_PAYLOAD))
    out = agent_runner._anthropic_web_search(
        "q", model_id="claude-sonnet-4-6", max_results=1
    )
    assert len(out["results"]) == 1


def test_web_search_surfaces_tool_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    payload = {
        "content": [
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_2",
                "content": {
                    "type": "web_search_tool_result_error",
                    "error_code": "max_uses_exceeded",
                },
            }
        ],
        "stop_reason": "end_turn",
    }
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(payload))
    out = agent_runner._anthropic_web_search("q", model_id="claude-sonnet-4-6")
    assert out["ok"] is False
    assert "max_uses_exceeded" in out["error"]


def test_web_search_needs_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = agent_runner._anthropic_web_search("q", model_id="claude-sonnet-4-6")
    assert out["ok"] is False
    assert "ANTHROPIC_API_KEY" in out["error"]


def test_web_search_http_error_is_soft(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakeResponse({}, status_code=403, text="nope")
    )
    out = agent_runner._anthropic_web_search("q", model_id="claude-sonnet-4-6")
    assert out["ok"] is False
    assert "403" in out["error"]


def test_search_model_prefers_anthropic_run_model():
    # An Anthropic run model is used as-is for the search.
    assert agent_runner._anthropic_search_model("claude/opus-4-8") == "claude-opus-4-8"
    # A non-Anthropic run model falls back to the default Claude (web_search
    # needs a Claude even when the agent itself runs via OpenRouter).
    from worker.llm_models import DEFAULT_MODEL_SLUG, resolve

    assert (
        agent_runner._anthropic_search_model("gpt/5")
        == resolve(DEFAULT_MODEL_SLUG).upstream_id
    )
    # Unknown slug also falls back rather than raising.
    assert agent_runner._anthropic_search_model("nope/nope")
