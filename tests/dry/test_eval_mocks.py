"""Eval-time tool mocking ("suite mode").

Three layers:
  - the mock engine (`worker.eval_mocks`): policy + result-shaping;
  - the manifest / dataset parsers (`evals.mocks`, per-case `mocks`);
  - the agent-loop chokepoint, end-to-end via `run_eval_local`: a side-effecting
    tool is short-circuited with a stub so a suite run never executes it for real.
"""

from __future__ import annotations

import json

import pytest

from worker import agent_runner as ar
from worker.eval_mocks import SIDE_EFFECTING, merge_mocks, mock_tool
from worker.eval_local import parse_cases_jsonl, run_eval_local
from worker.local_run import LocalRunError
from worker.manifest import ManifestError, _parse_eval_mocks
from worker.providers.base import NormalizedResponse, NormalizedToolUse


# ── mock engine ─────────────────────────────────────────────────────────────
def test_mock_tool_none_for_pure_and_unknown_tools():
    # Pure/local built-ins and un-mocked custom tools run for real (None).
    assert mock_tool("bash", {"command": "echo hi"}, {}) is None
    assert mock_tool("file_read", {"path": "x"}, {}) is None
    assert mock_tool("my_custom_tool", {"a": 1}, {}) is None
    assert mock_tool("my_custom_tool", {"a": 1}, None) is None


def test_mock_tool_default_stub_for_side_effecting_builtins():
    for name in SIDE_EFFECTING:
        content = mock_tool(name, {}, {})
        assert content is not None, name
        payload = json.loads(content)
        assert payload.get("mock") is True, name
    # Media stubs carry a drive_path so a downstream step has something to use.
    assert "drive_path" in json.loads(mock_tool("generate_video", {}, {}))


def test_declared_mock_overrides_default_and_covers_custom_tools():
    # A declared mock wins over the built-in default…
    out = mock_tool("generate_video", {}, {"generate_video": {"drive_path": "drive/fix/v.mp4"}})
    assert json.loads(out) == {"drive_path": "drive/fix/v.mp4"}
    # …and is the only way to mock a custom tool (otherwise None).
    out = mock_tool("publish_post", {}, {"publish_post": {"ok": True, "id": "p1"}})
    assert json.loads(out)["id"] == "p1"
    # A plain-string mock passes through verbatim (some tools return text).
    assert mock_tool("publish_post", {}, {"publish_post": "done"}) == "done"


def test_merge_mocks_case_overrides_skill():
    skill = {"generate_video": {"drive_path": "skill.mp4"}, "send_email": {"ok": True}}
    case = {"generate_video": {"drive_path": "case.mp4"}}
    merged = merge_mocks(skill, case)
    assert merged["generate_video"] == {"drive_path": "case.mp4"}  # case wins
    assert merged["send_email"] == {"ok": True}                    # skill kept
    assert merge_mocks(None, None) == {}


# ── manifest: evals.mocks ────────────────────────────────────────────────────
def test_parse_eval_mocks_ok():
    data = {"evals": {"mocks": {"generate_video": {"drive_path": "x.mp4"}}}}
    assert _parse_eval_mocks("s", data, is_agentic=True) == {
        "generate_video": {"drive_path": "x.mp4"}
    }
    # No evals / bare-list evals / no mocks key → empty.
    assert _parse_eval_mocks("s", {}, is_agentic=True) == {}
    assert _parse_eval_mocks("s", {"evals": []}, is_agentic=True) == {}
    assert _parse_eval_mocks("s", {"evals": {"graders": []}}, is_agentic=True) == {}


def test_parse_eval_mocks_rejects_bad_shapes():
    with pytest.raises(ManifestError, match="mapping of tool-name"):
        _parse_eval_mocks("s", {"evals": {"mocks": [1, 2]}}, is_agentic=True)
    with pytest.raises(ManifestError, match="non-empty tool names"):
        _parse_eval_mocks("s", {"evals": {"mocks": {"": {}}}}, is_agentic=True)
    with pytest.raises(ManifestError, match="only applies to agentic"):
        _parse_eval_mocks("s", {"evals": {"mocks": {"t": {}}}}, is_agentic=False)


# ── dataset: per-case mocks ──────────────────────────────────────────────────
def test_parse_cases_jsonl_per_case_mocks():
    cases = parse_cases_jsonl(
        '{"id": "a", "inputs": {}, "mocks": {"generate_video": {"drive_path": "v.mp4"}}}\n'
        '{"id": "b", "inputs": {}}\n'
    )
    assert cases[0]["mocks"] == {"generate_video": {"drive_path": "v.mp4"}}
    assert cases[1]["mocks"] is None


def test_parse_cases_jsonl_rejects_non_dict_mocks():
    with pytest.raises(LocalRunError, match="`mocks` must be a mapping"):
        parse_cases_jsonl('{"id": "a", "inputs": {}, "mocks": [1]}\n')


# ── end-to-end: suite mode short-circuits a side-effecting tool ──────────────
@pytest.fixture
def offline_env(monkeypatch):
    from worker.config import get_settings

    for k in (
        "DATABASE_URL", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
        "PURAS_SERVICE_TOKEN", "PURAS_LOCAL_MODE", "LOCAL_DRIVE_PATH",
        "WORKDIR_ROOT", "ANTHROPIC_API_KEY",
    ):
        monkeypatch.setenv(k, "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _VideoThenOutputProvider:
    """First turn calls `generate_video` (a real, billed side effect); second turn
    records the output. In suite mode the first call must be stubbed — never
    really dispatched — so the run completes deterministically with no backend."""

    def __init__(self):
        self.calls = 0

    def messages_create(self, system, messages, tools, max_tokens, *, cache_messages=False):
        self.calls += 1
        if self.calls == 1:
            tu = NormalizedToolUse(id="v1", name="generate_video", input={"prompt": "a cat"})
        else:
            tu = NormalizedToolUse(id="o1", name="set_output", input={"message": "done"})
        return NormalizedResponse(
            stop_reason="tool_use", tool_uses=[tu],
            input_tokens=5, output_tokens=2, upstream_cost_micros=10,
        )


def _write_video_bundle(root):
    import yaml

    sd = root / "vid"
    (sd / "evals").mkdir(parents=True)
    (sd / "skill.yaml").write_text(
        yaml.safe_dump({
            "entrypoint": "SKILL.md",
            "description": "video skill",
            "input_schema": {"type": "object", "properties": {}},
            "output_schema": {"type": "object", "properties": {"message": {"type": "string"}}},
            "evals": {
                "dataset": "evals/cases.jsonl",
                "graders": [{"name": "shape", "kind": "schema", "weight": 1.0}],
            },
        })
    )
    (sd / "SKILL.md").write_text("Render a video then finish.")
    (sd / "evals" / "cases.jsonl").write_text('{"id": "c1", "inputs": {}}\n')
    return root


def test_suite_run_mocks_side_effecting_tool_end_to_end(tmp_path, monkeypatch, offline_env):
    _write_video_bundle(tmp_path / "b")
    monkeypatch.setattr(ar, "make_provider", lambda *a, **k: _VideoThenOutputProvider())

    events: list[tuple[str, dict]] = []
    rep = run_eval_local(
        str(tmp_path / "b"), api_key="sk-test",
        on_event=lambda t, p: events.append((t, p)),
    )

    # The run completed and the schema grader passed — proving generate_video was
    # short-circuited (a real dispatch has no backend offline).
    assert rep["total"] == 1 and rep["passed"] == 1
    # The stub surfaced as a tool_result tagged mock=True for generate_video.
    mock_results = [
        p for t, p in events
        if t == "tool_result" and p.get("mock") is True
    ]
    assert mock_results, "expected a mocked tool_result event"
    assert any("video" in (p.get("preview") or "") for p in mock_results)
