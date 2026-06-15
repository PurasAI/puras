"""Side-by-side replay of ONE game — the demo's money shot.

Plays a single (seed, config) game with both contestants and prints it turn by
turn, side by side: left = LangGraph (deterministic), right = Puras (agent). When
the mischievous host fires a perturbation, the turn is flagged with ⚡ and the
rule, so a viewer can watch the deterministic side stall on the exact turn the
rules bend while the agent reads the reply and keeps going.

    python -m harness.replay --seed 1000 --config 2

The deterministic side replays deterministically. The agent side is played once
through the runner (spends a little of your key), then its recorded guesses are
replayed through a fresh game to recover the per-turn feedback + perturbations.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.experiment import CONFIGS, build_game  # noqa: E402
from engine.protocol import Status  # noqa: E402
from engine.scoring import play  # noqa: E402
from players.langgraph_player import LangGraphSolver  # noqa: E402


def _replay_history(seed, config_id, guesses, max_guesses):
    """Re-run a recorded guess list through a fresh game to recover each turn's
    feedback and which perturbations fired (the host is deterministic per seed)."""
    game = build_game(seed, config_id, max_guesses=max_guesses)
    rows = []
    for g in guesses:
        obs = game.guess(g)
        rows.append((g, obs.feedback, obs.accepted, list(game.history[-1].perturbations_fired)))
        if game.status is not Status.ONGOING:
            break
    return rows, game.status


def _det_rows(seed, config_id, max_guesses):
    game = build_game(seed, config_id, max_guesses=max_guesses)
    try:
        play(game, LangGraphSolver(), seed=seed)
    except Exception:
        pass
    rows = [(t.guess, t.observation.feedback, t.observation.accepted,
             list(t.perturbations_fired)) for t in game.history]
    return rows, game.status


def _puras_rows(seed, config_id, max_guesses, model):
    os.environ["PYTHONPATH"] = f"{ROOT}:{ROOT.parents[1]}:" + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, str(ROOT.parents[1]))
    from worker.local_run import run_local

    sess = Path(tempfile.mkdtemp(prefix="replay-"))
    try:
        run_local(str(ROOT / "players/puras_player"),
                  {"seed": seed, "config": config_id, "session_dir": str(sess),
                   "max_guesses": max_guesses, "word_length": 5},
                  skill="codebreaker", model=model, on_event=lambda *a, **k: None)
        guesses = json.loads((sess / "state.json").read_text()).get("guesses", [])
    finally:
        shutil.rmtree(sess, ignore_errors=True)
    return _replay_history(seed, config_id, guesses, max_guesses)


def _fmt(rows, i):
    if i >= len(rows):
        return " " * 26
    g, fb, acc, perts = rows[i]
    mark = "" if acc else " ✗"
    cell = f"{g:<7} {fb[:12]:<12}{mark}"
    return f"{cell:<26}"


def _pert_label(det_rows, pur_rows, i):
    perts = []
    for rows in (det_rows, pur_rows):
        if i < len(rows):
            perts += rows[i][3]
    if not perts:
        return ""
    names = sorted({p.split(":")[0] for p in perts})
    return "  ⚡ " + ", ".join(names)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Side-by-side single-game replay")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--config", type=int, default=2)
    ap.add_argument("--max-guesses", type=int, default=6)
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY") and Path("/tmp/.akey").exists():
        os.environ["ANTHROPIC_API_KEY"] = Path("/tmp/.akey").read_text().strip()

    cfg = CONFIGS[args.config]
    secret = build_game(args.seed, args.config).secret
    det_rows, det_status = _det_rows(args.seed, args.config, args.max_guesses)
    pur_rows, pur_status = _puras_rows(args.seed, args.config, args.max_guesses, args.model)

    print(f"\nWordle  seed={args.seed}  config={cfg['label']} (rate {cfg['rate']:.2f})  "
          f"secret={secret.upper()}")
    print("=" * 70)
    print(f"{'turn':<5}{'LangGraph (deterministic)':<26}{'Puras (agent)':<26}")
    print("-" * 70)
    n = max(len(det_rows), len(pur_rows))
    for i in range(n):
        print(f"{i+1:<5}{_fmt(det_rows, i)}{_fmt(pur_rows, i)}{_pert_label(det_rows, pur_rows, i)}")
    print("-" * 70)
    print(f"{'':5}{'→ '+det_status.value.upper():<26}{'→ '+pur_status.value.upper():<26}")


if __name__ == "__main__":
    main()
