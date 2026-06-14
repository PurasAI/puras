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
