"""Faz 1 guard for the Puras contestant's tool, without spending an LLM call.

We can't drive the real agent loop offline (that needs an API key), but we can
prove the make_guess bridge is correct and stateful: replaying guesses through
the tool must reproduce exactly the observations a directly-built game gives,
and must never leak the secret. We also confirm the skillpack manifest parses.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.experiment import build_game  # noqa: E402

_TOOL = ROOT / "players/puras_player/codebreaker/tools/make_guess.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("make_guess_under_test", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tool_matches_engine_and_keeps_state():
    tool = _load_tool()
    seed, config = 1, 0
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as sess:
        # The runner writes inputs to _inputs.json in the job workdir (cwd).
        (Path(work) / "_inputs.json").write_text(json.dumps(
            {"seed": seed, "config": config, "max_guesses": 6, "session_dir": sess}))
        cwd0 = os.getcwd()
        os.chdir(work)
        try:
            guesses = ["slate", "crane"]
            # Ground truth from a directly-built game.
            truth = build_game(seed, config, max_guesses=6)
            for g in guesses:
                want = truth.guess(g)
                got = tool.run(g)  # tool rebuilds + replays each call
                assert got["feedback"] == want.feedback
                assert got["status"] == want.status.value
                # Never leaks secret / true marks.
                assert "secret" not in got and "true_marks" not in got
        finally:
            os.chdir(cwd0)
        # Authoritative state persisted outside the (ephemeral) workdir.
        state = json.loads((Path(sess) / "state.json").read_text())
        assert state["guesses"] == guesses
        assert "secret" not in state


def test_skillpack_manifest_parses():
    # Needs the worker package importable; that's the repo we live in.
    sys.path.insert(0, str(ROOT.parents[1]))  # .../puras (repo root)
    os.environ.setdefault("ANTHROPIC_API_KEY", "x")
    try:
        from worker.manifest import parse_bundle_dir
    except Exception as e:  # pragma: no cover - environment guard
        print(f"skip manifest parse (worker import failed: {e})")
        return
    manifest = parse_bundle_dir(ROOT / "players/puras_player")
    names = {s.name for s in manifest.skills}
    assert "codebreaker" in names
    cb = next(s for s in manifest.skills if s.name == "codebreaker")
    assert cb.disable_bash is True


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
