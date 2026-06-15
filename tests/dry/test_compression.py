"""Dry tests for content-aware context compression (token economy).

Two layers:

  * `worker.compression.compress_text` — the pure, deterministic content router
    and its JSON / code / text compressors. Must never grow, never raise, never
    lose silently, and flag lossy passes so the caller can keep a reversible
    backup.

  * `worker.agent_runner._shrink_tool_result` — the loop seam that compresses
    then offloads a result before it enters history: a lossless pass is adopted
    inline with NO drive write; a lossy pass persists the exact original to the
    drive and leaves a `file_read` pointer; and when compression is off / not
    worthwhile it delegates to the legacy `_offload_tool_result` unchanged.

No network: the drive root is a tmp dir and the bucket push is stubbed.
"""

from __future__ import annotations

import json
import types

import pytest

from worker import agent_runner as ar
from worker import drive as drive_mod
from worker import compression as c

WS = "ws1"
JOB = "job1"


# ---- compress_text: the pure compressors ----------------------------------


def test_json_minify_is_lossless_and_equivalent():
    pretty = json.dumps({"a": 1, "b": [1, 2, 3], "c": {"d": "e"}}, indent=4)
    r = c.compress_text(pretty)
    assert r.kind == "json" and r.applied and r.lossless
    assert json.loads(r.text) == {"a": 1, "b": [1, 2, 3], "c": {"d": "e"}}
    assert r.compressed_chars < r.original_chars


def test_json_collapses_long_object_array_lossy_with_marker():
    arr = json.dumps([{"id": i, "name": "x" * 20, "v": i * 2} for i in range(40)])
    r = c.compress_text(arr)
    assert r.applied and not r.lossless  # structural collapse → lossy
    assert "__compressed__" in r.text
    out = json.loads(r.text)
    # leading sample kept verbatim + one summary object with the elision count
    assert out[0] == {"id": 0, "name": "x" * 20, "v": 0}
    assert out[-1]["item_keys"] == ["id", "name", "v"]
    assert r.ratio > 0.5


def test_json_scalar_array_is_kept_intact():
    # cheap scalar arrays aren't collapsed — only verbose object-arrays are
    arr = json.dumps({"ids": list(range(50))})
    r = c.compress_text(arr)
    # minify may still apply, but the scalar list survives in full
    assert json.loads(r.text) == {"ids": list(range(50))}


def test_code_strips_comments_but_preserves_hash_in_strings():
    code = (
        "import os\n"
        "from sys import path\n"
        "\n"
        "def foo(x):\n"
        "    # a real comment to drop\n"
        '    y = "not # a comment"   # trailing comment\n'
        "    return y\n"
        "class Bar:\n"
        "    pass\n"
    )
    r = c.compress_text(code)
    assert r.kind == "code" and r.applied and not r.lossless
    assert "a real comment to drop" not in r.text
    assert "trailing comment" not in r.text
    assert 'not # a comment' in r.text  # `#` inside a string literal is untouched


def test_text_folds_identical_line_runs_lossy():
    log = "start\n" + "WARN retrying\n" * 12 + "done\n"
    r = c.compress_text(log)
    assert r.kind == "text" and r.applied and not r.lossless
    assert "×12 identical lines" in r.text
    assert "start" in r.text and "done" in r.text


def test_text_whitespace_and_ansi_cleanup_is_lossless():
    noisy = ("line one   \n\x1b[31mred\x1b[0m   \n\n\n\nlast\n") * 30
    r = c.compress_text(noisy)
    assert r.kind == "text" and r.applied and r.lossless
    assert "\x1b[" not in r.text  # ANSI stripped
    assert "red" in r.text


def test_compress_is_deterministic():
    arr = json.dumps([{"id": i, "k": "v" * 30} for i in range(30)])
    assert c.compress_text(arr).text == c.compress_text(arr).text


def test_compress_never_grows_and_handles_non_string():
    # high-entropy short text can't be compressed → returned unchanged
    r = c.compress_text("a9f3kz-Q!7")
    assert not r.applied and r.text == "a9f3kz-Q!7"
    for junk in (None, 123, b"bytes", ["x"]):
        out = c.compress_text(junk)
        assert out.applied is False


def test_malformed_json_falls_through_to_text():
    # starts like JSON but isn't — must not raise, just route to the text pass
    broken = '{"a": 1, "b": ' + "oops\n" * 20
    r = c.compress_text(broken)
    assert r.kind in ("text", "none")  # never "json", never an exception


# ---- _shrink_tool_result: the loop seam -----------------------------------


@pytest.fixture
def drive(tmp_path, monkeypatch):
    uploads: list[str] = []
    monkeypatch.setattr(drive_mod, "_drive_root", tmp_path)
    monkeypatch.setattr(ar, "upload_drive_file", lambda ws, rel: uploads.append(rel) or True)
    return types.SimpleNamespace(root=tmp_path, uploads=uploads)


def _settings(monkeypatch, **over):
    base = {
        "tool_result_offload_chars": 100000,  # high, so offload doesn't fire by default
        "tool_result_offload_head_chars": 200,
        "tool_result_compress_enabled": True,
        "tool_result_compress_min_chars": 200,
        "tool_result_compress_min_ratio": 0.2,
    }
    base.update(over)
    monkeypatch.setattr(ar, "get_settings", lambda: types.SimpleNamespace(**base))


def _toolout(drive, tid):
    return drive.root / WS / "_jobs" / JOB / "_toolout" / f"{tid}.txt"


def test_shrink_lossless_compress_adopts_inline_no_drive_write(drive, monkeypatch):
    _settings(monkeypatch)
    pretty = json.dumps([{"id": i} for i in range(3)] + [{"big": "x" * 400}], indent=6)
    out = ar._shrink_tool_result("bash", "t1", pretty, JOB, WS)
    # adopted the minified JSON inline, smaller, still valid — and NO backup file
    assert isinstance(out, str) and len(out) < len(pretty)
    assert json.loads(out) == json.loads(pretty)
    assert not _toolout(drive, "t1").exists()


def test_shrink_lossy_compress_persists_original_and_points(drive, monkeypatch):
    _settings(monkeypatch)
    arr = json.dumps([{"id": i, "name": "n" * 30} for i in range(60)])
    out = ar._shrink_tool_result("web_fetch", "t2", arr, JOB, WS)
    assert "__compressed__" in out and "file_read" in out
    assert "_jobs/job1/_toolout/t2.txt" in out
    # the byte-exact original is recoverable from the drive
    assert _toolout(drive, "t2").read_text() == arr
    assert len(out) < len(arr)


def test_shrink_delegates_to_offload_when_disabled(drive, monkeypatch):
    # compression off → behave exactly like the legacy offload
    _settings(monkeypatch, tool_result_compress_enabled=False, tool_result_offload_chars=100)
    big = json.dumps([{"id": i, "name": "n" * 30} for i in range(60)])
    out = ar._shrink_tool_result("web_fetch", "t3", big, JOB, WS)
    assert "__compressed__" not in out  # not compressed
    assert "file_read" in out and "_toolout/t3.txt" in out
    assert _toolout(drive, "t3").read_text() == big


def test_shrink_delegates_when_compression_not_worthwhile(drive, monkeypatch):
    # a result that barely shrinks (< min_ratio) must not get a compression marker
    _settings(monkeypatch, tool_result_compress_min_ratio=0.99, tool_result_offload_chars=100000)
    text = "unique line number %d here\n" % 0 + "".join(
        f"distinct row {i} value {i*7}\n" for i in range(60)
    )
    out = ar._shrink_tool_result("bash", "t4", text, JOB, WS)
    # below offload limit and compression rejected → passes through untouched
    assert out == text
    assert not _toolout(drive, "t4").exists()


def test_shrink_compress_then_offload_for_huge_result(drive, monkeypatch):
    # lossy-compressed but STILL over the offload cap → head + pointer to original
    _settings(monkeypatch, tool_result_offload_chars=300, tool_result_offload_head_chars=120)
    arr = json.dumps([{"id": i, "name": "n" * 40} for i in range(400)])
    out = ar._shrink_tool_result("web_fetch", "t5", arr, JOB, WS)
    assert len(out) <= 300 + 400  # bounded by head + the pointer footer
    assert "file_read" in out and "_toolout/t5.txt" in out
    # full ORIGINAL (not the compressed form) is what's retrievable
    assert _toolout(drive, "t5").read_text() == arr


def test_shrink_passes_through_non_string(drive, monkeypatch):
    _settings(monkeypatch)
    blocks = [{"type": "text", "text": "x" * 500}, {"type": "image", "source": {}}]
    assert ar._shrink_tool_result("file_read", "t6", blocks, JOB, WS) is blocks


def test_shrink_write_failure_keeps_data_intact(drive, monkeypatch):
    # a lossy compression whose backup write fails must keep the FULL original
    # inline rather than serve an unrecoverable compressed stub
    _settings(monkeypatch)
    monkeypatch.setattr(ar, "_run_file_write", lambda *a, **k: {"ok": False, "error": "boom"})
    arr = json.dumps([{"id": i, "name": "n" * 30} for i in range(60)])
    out = ar._shrink_tool_result("web_fetch", "t7", arr, JOB, WS)
    assert out == arr  # never lose data we can't restore
