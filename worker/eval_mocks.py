"""Side-effect mocking for eval/test runs ("suite mode").

Evals are to a skill what unit tests are to code — but a unit test must not fire
the real side effects of the code under test. A skill that renders 20 videos,
sends email, or writes to memory on every run would be ruinous (and
non-deterministic) to test for real. So when the agent loop runs in SUITE mode
(an offline `puras eval` suite or a hosted eval-suite job — NEVER a live run the
user submitted), its side-effecting tools return canned stubs instead of really
executing:

  - Built-in side-effecting verbs (`SIDE_EFFECTING` below) get a SAFE DEFAULT
    stub automatically, even when the skill declares no mock — so a brand-new
    skill's eval suite is protected the moment it's run.
  - A skill DECLARES explicit mock responses per tool in `evals.mocks` (and a
    dataset case can override per-case), to feed the agent a realistic value (a
    fixed drive path, a known row) so the run reaches its `set_output`
    deterministically. A declared mock overrides the built-in default AND is the
    only way to mock a CUSTOM tool — the platform can't know a custom tool's side
    effects, so the author opts it in by declaring one.

Pure / workdir-local / read-only built-ins are NEVER mocked — they're
deterministic, confined to the run's workdir, and the agent needs their real
results to make progress: `set_output`, `bash`, `file_read`/`file_write`/
`file_edit`, `todo_write`, `drive_url`, `drive_pull`, `memory_search`/
`memory_get`, `describe_subagent`, `run_subagent`. (A nested `run_subagent`
inherits suite mode through the same `ctx`, so the nested run's OWN
side-effecting tools are mocked too.)

This module is the policy + the result-shaping; `agent_runner` calls `mock_tool`
at its single dispatch chokepoint, and `eval_local` / the worker job runner flip
`ctx.suite_mode` on and hand over the merged mock table.
"""

from __future__ import annotations

import json
from typing import Any

# Built-in tools with real external / billed / persistent side effects. In suite
# mode each returns a safe default stub even when the skill declares no mock for
# it, so a test run can't render media, send email, fetch the web, or write to
# memory just by being executed. An `evals.mocks` entry overrides the default.
SIDE_EFFECTING: frozenset[str] = frozenset(
    {
        "generate_image",
        "generate_video",
        "generate_audio",
        "transcribe",
        "web_search",
        "image_search",
        "web_fetch",
        "web_screenshot",
        "download_url",
        "send_email",
        "memory_put",
        "memory_forget",
    }
)


def _default_stub(tool_name: str, inp: dict) -> dict:
    """A safe, side-effect-free canned result for a built-in side-effecting verb,
    shaped to roughly match the real tool's result so the model isn't confused.
    Every stub is tagged `"mock": true` so a grader (or a human reading the
    timeline) can tell a stubbed step from a real one."""
    if tool_name == "generate_image":
        return {"drive_path": "drive/mock/image-1.png", "kind": "image", "billed_micros": 0, "mock": True}
    if tool_name == "generate_video":
        return {"drive_path": "drive/mock/video-1.mp4", "kind": "video", "billed_micros": 0, "mock": True}
    if tool_name == "generate_audio":
        return {"drive_path": "drive/mock/audio-1.mp3", "kind": "audio", "billed_micros": 0, "mock": True}
    if tool_name == "transcribe":
        return {"text": "[mock transcript]", "billed_micros": 0, "mock": True}
    if tool_name in ("web_search", "image_search"):
        return {"results": [], "mock": True}
    if tool_name == "web_fetch":
        return {"url": inp.get("url"), "text": "[mock fetched content]", "mock": True}
    if tool_name == "web_screenshot":
        return {"drive_path": "drive/mock/screenshot-1.png", "mock": True}
    if tool_name == "download_url":
        return {"drive_path": "drive/mock/download-1", "bytes": 0, "mock": True}
    if tool_name == "send_email":
        to = inp.get("to")
        recipients = to if isinstance(to, list) else ([to] if to else [])
        return {"ok": True, "recipients": recipients, "mock": True}
    if tool_name == "memory_put":
        return {"ok": True, "id": "mock-memory-1", "mock": True}
    if tool_name == "memory_forget":
        return {"ok": True, "forgotten": 0, "mock": True}
    # Defensive: a verb added to SIDE_EFFECTING without a stub still no-ops safely.
    return {"ok": True, "mock": True}


def _as_content(value: Any) -> str:
    """Render a mock response as the tool-result CONTENT string the agent sees —
    a plain string passes through; anything else is compact JSON (the shape the
    side-effecting verbs return)."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def mock_tool(tool_name: str, tool_input: Any, mocks: dict[str, Any] | None) -> str | None:
    """Decide whether to short-circuit a tool call in suite mode and, if so,
    return the result CONTENT string the agent should see (the same shape a real
    tool result carries). Returns None to let the tool run for real.

    Policy, in order:
      1. A declared mock (`mocks[tool_name]`) always wins — for built-ins AND
         custom tools. The declared value IS the result.
      2. Otherwise a built-in `SIDE_EFFECTING` verb falls back to a safe default
         stub (safe-by-default protection).
      3. Otherwise None — the tool runs for real (pure/local built-ins, and any
         custom tool the author didn't opt into mocking).

    The caller (`agent_runner`) is responsible for only invoking this in suite
    mode and for never passing `set_output` (run infrastructure, never mocked).
    """
    mocks = mocks or {}
    inp = tool_input if isinstance(tool_input, dict) else {}
    if tool_name in mocks:
        return _as_content(mocks[tool_name])
    if tool_name in SIDE_EFFECTING:
        return _as_content(_default_stub(tool_name, inp))
    return None


def merge_mocks(
    skill_mocks: dict[str, Any] | None, case_mocks: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge skill-level `evals.mocks` with a dataset case's per-case `mocks` for
    one suite run — the case wins on a key collision, so a case can override (or
    add) a tool's mock without touching the skill default."""
    out = dict(skill_mocks or {})
    if isinstance(case_mocks, dict):
        out.update(case_mocks)
    return out
