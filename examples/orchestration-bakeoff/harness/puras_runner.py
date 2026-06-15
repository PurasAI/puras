"""Run the Puras contestant for one (seed, config) cell via the local runner.

The agent plays the whole game inside one ``run_local`` call (calling make_guess
each turn); we then read the *engine-truth* state the tool persisted — we never
trust the agent's own claim of whether it won. Cost comes from the run's usage
tally.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../orchestration-bakeoff
REPO = ROOT.parents[1]                               # .../puras (worker lives here)
SKILLPACK = ROOT / "players/puras_player"

from engine.scoring import GameResult  # noqa: E402


def _ensure_paths() -> None:
    # engine importable inside the tool subprocess (function_runner appends the
    # parent's PYTHONPATH); worker + engine importable here in the parent.
    extra = f"{ROOT}:{REPO}"
    cur = os.environ.get("PYTHONPATH", "")
    if extra not in cur:
        os.environ["PYTHONPATH"] = f"{extra}:{cur}" if cur else extra
    for p in (str(REPO), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def run_puras_game(
    seed: int,
    config_id: int,
    *,
    max_guesses: int = 6,
    model: str | None = None,
    rate: float = 0.0,
    quiet: bool = True,
) -> GameResult:
    _ensure_paths()
    from worker.local_run import run_local  # imported after paths/env are ready

    sess = Path(tempfile.mkdtemp(prefix=f"bakeoff-{seed}-{config_id}-"))
    on_event = (lambda *_a, **_k: None) if quiet else None
    t0 = time.perf_counter()
    error = None
    cost_micros = 0
    try:
        res = run_local(
            str(SKILLPACK),
            {"seed": seed, "config": config_id, "session_dir": str(sess),
             "max_guesses": max_guesses, "word_length": 5},
            skill="codebreaker",
            model=model,
            on_event=on_event,
        )
        cost_micros = (res.get("usage") or {}).get("cost_micros", 0)
    except Exception as exc:  # noqa: BLE001 — a runner failure is a (recorded) loss
        error = f"{type(exc).__name__}: {exc}"

    state_file = sess / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    shutil.rmtree(sess, ignore_errors=True)

    return GameResult(
        player="puras-agent",
        secret="",  # not needed for aggregate; engine-truth lives in `state`
        seed=seed,
        rate=rate,
        won=bool(state.get("won")),
        guesses_used=int(state.get("guesses_used", 0)),
        attempts=int(state.get("attempts", 0)),
        perturbed=config_id != 0,
        elapsed_s=time.perf_counter() - t0,
        cost_micros=int(cost_micros or 0),
        error=error or (None if state else "no state written"),
        history=[],
    )
