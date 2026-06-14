"""`puras serve` — the local API server that mirrors the hosted job contract.

These exercise the HTTP surface end to end (submit → run → poll → events) with
the actual offline runner STUBBED, so the test needs no LLM key and no network:
it proves the server's routing, job lifecycle, auth, and CORS — the contract the
SDKs depend on — not the agent loop (covered elsewhere). Stdlib only, so it ships
fine in the dependency-light open-source runner.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from worker import local_run, local_server


def _req(method, url, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None), dict(r.headers)
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, (json.loads(raw) if raw else None), dict(e.headers)


def _fake_run_local(bundle_dir, inputs, *, skill=None, model=None, api_key=None, on_event=None):
    if on_event:
        on_event("model_response", {"step": 1, "input_tokens": 10, "output_tokens": 5})
        on_event("tool_use", {"name": "bash", "label": "echo hi"})
    return {
        "output": {"echo": inputs.get("name", "?"), "skill": skill},
        "steps": 1,
        "usage": {"input_tokens": 10, "output_tokens": 5, "cost_micros": 0},
        "spans": [
            {"span_id": "a", "parent_span_id": None, "kind": "run",
             "name": "run", "duration_ms": 12, "attributes": {}}
        ],
    }


def _start(app):
    httpd = app.make_server("127.0.0.1", 0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{port}"


@pytest.fixture()
def base(tmp_path, monkeypatch):
    monkeypatch.setattr(local_run, "run_local", _fake_run_local)
    app = local_server.LocalServer(str(tmp_path))
    httpd, url = _start(app)
    try:
        yield url
    finally:
        httpd.shutdown()
        httpd.server_close()


def _poll_terminal(base, job_id, tries=100):
    for _ in range(tries):
        st, job, _h = _req("GET", f"{base}/v1/jobs/{job_id}")
        assert st == 200
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        import time

        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached a terminal status")


def test_submit_wait_returns_result(base):
    st, job, _h = _req(
        "POST", f"{base}/v1/jobs?wait=true&timeout=10",
        {"skill": "greeter", "inputs": {"name": "Ada"}},
    )
    assert st == 201
    assert job["status"] == "succeeded"
    assert job["result"] == {"echo": "Ada", "skill": "greeter"}
    assert job["skill_name"] == "greeter"
    assert job["workspace_id"] == local_server._LOCAL_WORKSPACE_ID


def test_qualified_skill_path_resolves_to_last_segment(base):
    st, job, _h = _req(
        "POST", f"{base}/v1/jobs?wait=true&timeout=10",
        {"skill": "acme/demo/greeter", "inputs": {}},
    )
    assert st == 201 and job["status"] == "succeeded"
    assert job["result"]["skill"] == "greeter"


def test_async_submit_then_poll_and_events(base):
    st, job, _h = _req("POST", f"{base}/v1/jobs", {"skill": "greeter", "inputs": {"name": "Bo"}})
    assert st == 201
    job = _poll_terminal(base, job["id"])
    assert job["status"] == "succeeded"

    st, events, _h = _req("GET", f"{base}/v1/jobs/{job['id']}/events")
    assert st == 200
    types = [e["type"] for e in events]
    assert "model_response" in types and "tool_use" in types
    # ?after returns only the tail.
    first_id = events[0]["id"]
    st, tail, _h = _req("GET", f"{base}/v1/jobs/{job['id']}/events?after={first_id}")
    assert all(e["id"] > first_id for e in tail)

    st, spans, _h = _req("GET", f"{base}/v1/jobs/{job['id']}/spans")
    assert st == 200 and spans and spans[0]["kind"] == "run"


def test_run_failure_is_reported_not_crashed(base, monkeypatch):
    def _boom(*a, **k):
        raise local_run.LocalRunError("no LLM key")

    monkeypatch.setattr(local_run, "run_local", _boom)
    st, job, _h = _req("POST", f"{base}/v1/jobs?wait=true&timeout=10", {"skill": "greeter", "inputs": {}})
    assert st == 201
    assert job["status"] == "failed"
    assert "no LLM key" in job["error"]


def test_unknown_job_is_404(base):
    st, body, _h = _req("GET", f"{base}/v1/jobs/does-not-exist")
    assert st == 404 and "not found" in body["detail"]


def test_cors_preflight(base):
    st, _body, headers = _req("OPTIONS", f"{base}/v1/jobs")
    assert st == 204
    assert headers.get("Access-Control-Allow-Origin") == "*"


def test_bad_json_body_is_400(base):
    req = urllib.request.Request(
        f"{base}/v1/jobs", data=b"{not json", method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code
    assert status == 400


def test_require_key_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(local_run, "run_local", _fake_run_local)
    app = local_server.LocalServer(str(tmp_path), require_key="sekret")
    httpd, base = _start(app)
    try:
        # /health is open even with auth on.
        st, _b, _h = _req("GET", f"{base}/health")
        assert st == 200
        # /v1/* without (or with a wrong) token is rejected.
        st, _b, _h = _req("GET", f"{base}/v1/jobs")
        assert st == 401
        st, _b, _h = _req("GET", f"{base}/v1/jobs", headers={"Authorization": "Bearer nope"})
        assert st == 401
        # …and accepted with the right one.
        st, _b, _h = _req("GET", f"{base}/v1/jobs", headers={"Authorization": "Bearer sekret"})
        assert st == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
