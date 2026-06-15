"""Run exactly one Puras game and print its result as a JSON line.

Isolating each agent game in its own process sidesteps the global state
``run_local`` mutates (env vars, settings cache), so the harness can run many
games concurrently with a process pool. Not meant to be called by hand:

    python -m harness.one_game <seed> <config_id> [max_guesses] [model]
"""

from __future__ import annotations

import json
import sys

from harness.puras_runner import run_puras_game
from engine.experiment import CONFIGS


def main(argv=None):
    argv = argv or sys.argv[1:]
    seed = int(argv[0])
    config_id = int(argv[1])
    max_guesses = int(argv[2]) if len(argv) > 2 else 6
    model = argv[3] if len(argv) > 3 and argv[3] else None
    gr = run_puras_game(seed, config_id, max_guesses=max_guesses, model=model,
                        rate=CONFIGS[config_id]["rate"])
    print(json.dumps({
        "player": gr.player, "seed": gr.seed, "config": config_id, "rate": gr.rate,
        "won": gr.won, "guesses_used": gr.guesses_used, "attempts": gr.attempts,
        "cost_micros": gr.cost_micros, "elapsed_s": round(gr.elapsed_s, 1),
        "error": gr.error,
    }))


if __name__ == "__main__":
    main()
