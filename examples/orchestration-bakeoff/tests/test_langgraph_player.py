"""Faz 1 guard for the deterministic contestant: it must (a) solve clean games
near-optimally and (b) break *structurally* under perturbation — that asymmetry
is the whole demo. Runs under pytest or standalone."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import Status, new_game  # noqa: E402
from engine.game import WordleGame  # noqa: E402
from engine.perturbations import Injector  # noqa: E402
from engine.scoring import play  # noqa: E402
from players.langgraph_player import LangGraphSolver  # noqa: E402


def test_solves_clean_games():
    # On unperturbed games the steelman should win nearly always, in few guesses.
    wins = 0
    total = 30
    for seed in range(total):
        g = new_game(seed=seed)
        r = play(g, LangGraphSolver(), seed=seed, rate=0.0)
        wins += r.won
    assert wins >= 28, f"clean win-rate too low: {wins}/{total}"


def test_format_shift_breaks_the_parser():
    # A latched format shift firing every turn must derail the fixed parser.
    inj = Injector(["format_shift"], rate=1.0, seed=5, turns=6)
    g = WordleGame(new_game(seed=5).secret, max_guesses=6, injector=inj)
    r = play(g, LangGraphSolver(), seed=5, rate=1.0)
    assert not r.won
    assert r.error is not None  # parser raised — informative, not a hidden crash


def test_silent_rule_change_misleads_the_filter():
    inj = Injector(["silent_rule_change"], rate=1.0, seed=9, turns=6)
    g = WordleGame(new_game(seed=9).secret, max_guesses=6, injector=inj)
    r = play(g, LangGraphSolver(), seed=9, rate=1.0)
    assert not r.won  # positional logic on presence-only feedback can't win


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
