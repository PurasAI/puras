"""Dry tests for Hindsight's pure detector core (worker/worker/hindsight_core.py).

Network- and DB-free: we build small windows of synthetic runs and assert each
detector finds (and only finds) the recurring pattern, plus the normalization
that makes "the same script with different literals" group together. These are
the regression contract for the deterministic detection layer; the cloud
LLM-prose/draft step is tested separately.
"""

from __future__ import annotations

from worker import hindsight_core as hc


def _call(job_id, name, ok=True, error="", **inp):
    return hc.ToolCall(job_id=job_id, name=name, input=inp, ok=ok, error_preview=error)


def _run(job_id, calls, *, injected=None, missed=False, miss_keys=None):
    return hc.RunTrace(
        job_id=job_id,
        tool_calls=calls,
        memory=hc.MemorySignal(
            injected_ids=list(injected or []),
            injected_count=len(injected or []),
            missed=missed,
            miss_keys=list(miss_keys or []),
        ),
    )


# ── normalization ───────────────────────────────────────────────────────────
def test_normalize_command_collapses_literals_urls_paths_numbers():
    a = hc.normalize_command('curl https://a.com/x -o /tmp/a.json --retry 3')
    b = hc.normalize_command('curl https://b.org/y -o /var/b.json --retry 9')
    assert a == b


def test_signature_stable_across_incidental_variation():
    assert hc._signature("python f.py 1") == hc._signature("python g.py 2")
    assert hc._signature("python f.py") != hc._signature("node f.js")


def test_normalize_error_groups_same_class():
    assert hc.normalize_error("file not found: /a/1.txt") == hc.normalize_error(
        "file not found: /b/2.txt"
    )


# ── tool: adhoc_code ──────────────────────────────────────────────────────────
def test_adhoc_code_fires_when_same_script_recurs_across_runs():
    runs = [
        _run("j1", [_call("j1", "bash", command="python conv.py in1.csv out1.json")]),
        _run("j2", [_call("j2", "bash", command="python conv.py in2.csv out2.json")]),
        _run("j3", [_call("j3", "bash", command="python conv.py in3.csv out3.json")]),
    ]
    findings = hc.detect_adhoc_code(hc.Window("s", runs))
    assert len(findings) == 1
    f = findings[0]
    assert f.family == "tool" and f.kind == "adhoc_code"
    assert f.evidence["distinct_runs"] == 3
    assert f.severity == "high"  # 3/3 of the window


def test_adhoc_code_ignores_one_off_script():
    runs = [
        _run("j1", [_call("j1", "bash", command="python once.py")]),
        _run("j2", [_call("j2", "bash", command="ls -la")]),
    ]
    assert hc.detect_adhoc_code(hc.Window("s", runs)) == []


def test_adhoc_code_only_counts_bash_not_other_tools():
    runs = [
        _run("j1", [_call("j1", "web_fetch", url="https://a")]),
        _run("j2", [_call("j2", "web_fetch", url="https://a")]),
    ]
    assert hc.detect_adhoc_code(hc.Window("s", runs)) == []


# ── error: repeated_error ─────────────────────────────────────────────────────
def test_repeated_error_groups_same_tool_and_error_class():
    runs = [
        _run("j1", [_call("j1", "bash", ok=False, error="file not found: /a/1")]),
        _run("j2", [_call("j2", "bash", ok=False, error="file not found: /b/2")]),
    ]
    findings = hc.detect_repeated_errors(hc.Window("s", runs))
    assert len(findings) == 1
    assert findings[0].evidence["tool"] == "bash"
    assert findings[0].evidence["total_failures"] == 2


def test_repeated_error_ignores_successes():
    runs = [
        _run("j1", [_call("j1", "bash", ok=True)]),
        _run("j2", [_call("j2", "bash", ok=True)]),
    ]
    assert hc.detect_repeated_errors(hc.Window("s", runs)) == []


# ── redundancy: redundant_call ────────────────────────────────────────────────
def test_redundant_call_detects_repeated_identical_call_in_one_run():
    runs = [
        _run(
            "j1",
            [
                _call("j1", "web_fetch", url="https://x.com"),
                _call("j1", "web_fetch", url="https://x.com"),
                _call("j1", "web_fetch", url="https://x.com"),
            ],
        )
    ]
    findings = hc.detect_redundant_calls(hc.Window("s", runs))
    assert len(findings) == 1
    assert findings[0].evidence["tool"] == "web_fetch"
    assert findings[0].evidence["wasted_calls"] == 2  # 3 calls, 1 legitimate


def test_redundant_call_ignores_distinct_args():
    runs = [
        _run(
            "j1",
            [
                _call("j1", "web_fetch", url="https://a.com"),
                _call("j1", "web_fetch", url="https://b.com"),
            ],
        )
    ]
    assert hc.detect_redundant_calls(hc.Window("s", runs)) == []


# ── memory ────────────────────────────────────────────────────────────────────
def test_memory_miss_fires_on_repeated_cold_lookups():
    runs = [
        _run("j1", [], missed=True, miss_keys=["url:a"]),
        _run("j2", [], missed=True, miss_keys=["url:a"]),
    ]
    findings = hc.detect_memory_misses(hc.Window("s", runs))
    assert len(findings) == 1
    assert findings[0].kind == "memory_miss"
    assert "url:a" in findings[0].evidence["top_miss_keys"]


def test_duplicate_writes_fires_on_same_subject_across_runs():
    runs = [
        _run("j1", [_call("j1", "memory_put", entity_key="acme-mug", title="Mug")]),
        _run("j2", [_call("j2", "memory_put", entity_key="acme-mug", title="Mug")]),
    ]
    findings = hc.detect_duplicate_writes(hc.Window("s", runs))
    assert len(findings) == 1
    assert findings[0].evidence["total_writes"] == 2


def test_unused_injected_fires_when_injected_but_never_read():
    runs = [
        _run("j1", [_call("j1", "bash", command="echo hi")], injected=["mem-1"]),
        _run("j2", [_call("j2", "bash", command="echo hi")], injected=["mem-1"]),
    ]
    findings = hc.detect_unused_injected(hc.Window("s", runs))
    assert len(findings) == 1
    assert findings[0].evidence["memory_id"] == "mem-1"
    assert findings[0].evidence["unused_runs"] == 2


def test_unused_injected_silent_when_row_is_read():
    runs = [
        _run("j1", [_call("j1", "memory_get", id="mem-1")], injected=["mem-1"]),
        _run("j2", [_call("j2", "memory_get", id="mem-1")], injected=["mem-1"]),
    ]
    assert hc.detect_unused_injected(hc.Window("s", runs)) == []


def test_refetch_candidate_fires_when_fetched_every_run():
    runs = [_run(f"j{i}", [_call(f"j{i}", "memory_get", id="mem-9")]) for i in range(4)]
    findings = hc.detect_refetch_candidates(hc.Window("s", runs))
    assert len(findings) == 1
    assert findings[0].evidence["memory_id"] == "mem-9"
    assert findings[0].evidence["fetched_runs"] == 4


# ── aggregate ─────────────────────────────────────────────────────────────────
def test_analyze_runs_all_detectors_and_sorts_by_severity():
    runs = [
        _run(
            "j1",
            [
                _call("j1", "bash", command="python conv.py a.csv"),
                _call("j1", "web_fetch", url="https://x"),
                _call("j1", "web_fetch", url="https://x"),
            ],
        ),
        _run("j2", [_call("j2", "bash", command="python conv.py b.csv")]),
    ]
    findings = hc.analyze(hc.Window("s", runs))
    kinds = {f.kind for f in findings}
    assert "adhoc_code" in kinds
    assert "redundant_call" in kinds
    sev_rank = ["high", "medium", "low"]
    ranks = [sev_rank.index(f.severity) for f in findings]
    assert ranks == sorted(ranks)  # high-severity first


def test_analyze_empty_window_is_empty():
    assert hc.analyze(hc.Window("s", [])) == []
