"""`make_guess` — the agent's only window into the game.

It is a thin, honest bridge to the shared engine: it submits the guess and
returns the host's reply (which may be perturbed) — and *nothing else*. It never
leaks the secret, the true marks, or whether a perturbation fired; the agent has
to read the reply and figure things out, exactly like the deterministic side.

State across a run: each call rebuilds the deterministic game from (seed, config)
and replays the guesses recorded so far in ``<session_dir>/state.json``, then
applies this guess and appends it. ``state.json`` lives outside the per-job
workdir (which is cleaned up) so the harness can read the authoritative,
engine-truth result after the agent finishes — it never trusts the agent's own
report of whether it won.

The engine is importable because the harness puts the bake-off root on
PYTHONPATH before invoking the runner.
"""

from __future__ import annotations

import json
from pathlib import Path

from engine.experiment import build_game


def _load_spec() -> dict:
    # The runner writes the skill inputs to _inputs.json in the job workdir (cwd).
    return json.loads(Path("_inputs.json").read_text())


def run(guess: str) -> dict:
    spec = _load_spec()
    seed = int(spec["seed"])
    config = int(spec["config"])
    max_guesses = int(spec.get("max_guesses", 6))
    session_dir = Path(spec["session_dir"])
    state_file = session_dir / "state.json"

    prior = []
    if state_file.exists():
        prior = json.loads(state_file.read_text()).get("guesses", [])

    game = build_game(seed, config, max_guesses=max_guesses)
    for g in prior:
        game.guess(g)
    obs = game.guess(guess or "")
    prior.append(guess or "")

    session_dir.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({
        "guesses": prior,
        "status": game.status.value,
        "won": game.status.value == "won",
        "guesses_used": game.guesses_made,
        "attempts": game.attempts,
    }))

    # Player-visible fields only.
    return {
        "accepted": obs.accepted,
        "feedback": obs.feedback,
        "status": obs.status.value,
        "guesses_made": obs.guesses_made,
        "guesses_left": obs.guesses_left,
    }
