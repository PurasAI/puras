"""Smoke test: the standalone runner imports and the CLI is wired.

Dependency-light by design — no DB, no network, no LLM call. It proves that a
plain `pip install -e . && pip install -e worker/sdk` gives you a working
offline entrypoint and a `puras` CLI with the local subcommands. The heavier
"the offline import path pulls no hosted deps" guarantee is enforced
separately by `tests/dry/test_local_import_isolation.py`.
"""

from __future__ import annotations


def test_offline_entrypoint_imports():
    from worker.local_run import run_local

    assert callable(run_local)


def test_cli_has_local_subcommands():
    # The published `puras` CLI must expose `run --local` and `eval --local`.
    from puras.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["run", "--local", "--dir", ".", "-i", "name=Ada"])
    assert args.cmd == "run" and args.local is True and args.dir == "."

    ev = parser.parse_args(["eval", "--local", "--dir", ".", "--threshold", "80"])
    assert ev.cmd == "eval" and ev.local is True and ev.threshold == 80


def test_cli_feedback_parses():
    from puras.cli import build_parser

    parser = build_parser()
    a = parser.parse_args(["feedback", "job-123", "--up", "-c", "great"])
    assert a.cmd == "feedback" and a.job_id == "job-123"
    assert a.rating == "up" and a.comment == "great"

    b = parser.parse_args(["feedback", "job-9", "--down", "--end-user", "u1"])
    assert b.rating == "down" and b.end_user == "u1"

    # no flag → no thumb (comment-only / explicit none)
    c = parser.parse_args(["feedback", "job-9"])
    assert c.rating == "none"


def test_client_feedback_posts_to_endpoint():
    import httpx

    import puras

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        import json as _json

        seen["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"id": "f1", "job_id": "j9", "rating": 1})

    transport = httpx.MockTransport(handler)
    client = puras.Client(api_key="k", api_base="https://api.test")
    # Route the client's httpx through the mock transport.
    orig = httpx.Client

    def _patched(*a, **kw):
        kw.pop("follow_redirects", None)
        return orig(transport=transport, **kw)

    httpx.Client = _patched  # type: ignore[assignment]
    try:
        out = client.feedback("j9", rating=1, comment="nice", end_user_id="u42")
    finally:
        httpx.Client = orig  # type: ignore[assignment]

    assert out["rating"] == 1
    assert seen["method"] == "POST"
    assert seen["url"].endswith("/v1/jobs/j9/feedback")
    assert seen["body"] == {"rating": 1, "comment": "nice", "end_user_id": "u42"}


def test_client_feedback_requires_signal():
    import puras

    client = puras.Client(api_key="k", api_base="https://api.test")
    try:
        client.feedback("j9")  # no thumb, no comment
    except ValueError as e:
        assert "rating" in str(e) or "comment" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for empty feedback")
