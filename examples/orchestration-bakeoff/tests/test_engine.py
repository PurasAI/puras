"""Faz 0 guards: scoring truth, the fairness invariants, and that win is always
judged honestly even while the host misbehaves. Runs under pytest or standalone
(`python tests/test_engine.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import Mark, Status, new_game, words  # noqa: E402
from engine.game import WordleGame  # noqa: E402
from engine.perturbations import Injector  # noqa: E402
from engine.protocol import GameView  # noqa: E402


def test_score_basic_and_duplicates():
    H, P, M = Mark.HIT, Mark.PRESENT, Mark.MISS
    assert words.score("crane", "crane") == (H, H, H, H, H)
    assert words.score("slate", "crane") == (M, M, H, M, H)
    # Duplicate-letter rule: only as many yellows as remain after greens.
    # secret has one 'l'; guess has two — the already-placed 'l' takes it,
    # so the other 'l' is a miss, not a yellow.
    assert words.score("lolly", "ulcer") == (P, M, M, M, M)


def test_word_lists_load():
    assert len(words.answers()) == 2315
    assert "crane" in words.allowed_guesses()
    assert words.answers()[0].islower()


def test_clean_game_win_and_default_format():
    g = new_game(seed=1, secret="crane")
    assert isinstance(g.view(), GameView)
    obs = g.guess("slate")
    assert obs.accepted and obs.feedback == "XXGXG"  # default G/Y/X legend
    assert obs.status is Status.ONGOING
    win = g.guess("crane")
    assert win.status is Status.WON and g.guesses_made == 2


def test_builtin_rejections_do_not_consume_a_guess():
    g = new_game(seed=1, secret="crane")
    before = g.guesses_made
    r = g.guess("toolong")
    assert not r.accepted and "letters" in r.feedback
    assert g.guesses_made == before  # rejection costs an attempt, not a guess
    assert g.attempts == 1


def test_fairness_schedule_is_deterministic_and_play_independent():
    # Same args -> identical per-turn activation, regardless of who plays.
    a = Injector(["format_shift", "noise", "new_constraint"], rate=0.5, seed=42, turns=6)
    b = Injector(["format_shift", "noise", "new_constraint"], rate=0.5, seed=42, turns=6)
    for t in range(6):
        fa = [(p.name, params) for p, params in a.active(t)]
        fb = [(p.name, params) for p, params in b.active(t)]
        assert fa == fb
    # A different seed should (almost surely) differ somewhere.
    c = Injector(["format_shift", "noise", "new_constraint"], rate=0.5, seed=7, turns=6)
    diff = any([(p.name, params) for p, params in a.active(t)]
               != [(p.name, params) for p, params in c.active(t)] for t in range(6))
    assert diff


def test_win_is_truthful_under_lies():
    # Even with a lie + silent-rule-change firing every turn, guessing the
    # secret wins. The host can corrupt feedback, never the verdict.
    inj = Injector(["temporary_lie", "silent_rule_change"], rate=1.0, seed=3, turns=6)
    g = WordleGame("crane", max_guesses=6, injector=inj)
    g.guess("slate")  # eat a turn under perturbation
    win = g.guess("crane")
    assert win.status is Status.WON


def test_new_constraint_rejects_violation():
    inj = Injector(["new_constraint"], rate=1.0, seed=0, turns=6)
    g = WordleGame("crane", max_guesses=6, injector=inj)
    rule = inj.active(0)[0][1]["rule"]
    bad = {"no_repeats": "lolly", "must_contain_e": "warts", "no_vowel_start": "audio"}[rule]
    r = g.guess(bad)
    assert not r.accepted and "rejected" in r.feedback


def test_stuck_player_loses_via_attempt_cap():
    # A player that keeps proposing the same rejected guess must eventually lose.
    inj = Injector(["new_constraint"], rate=1.0, seed=0, turns=6)
    g = WordleGame("crane", max_guesses=6, injector=inj, max_attempts=5)
    rule = inj.active(0)[0][1]["rule"]
    bad = {"no_repeats": "lolly", "must_contain_e": "warts", "no_vowel_start": "audio"}[rule]
    for _ in range(5):
        g.guess(bad)
    assert g.status is Status.LOST


def test_format_shift_changes_rendering():
    inj = Injector(["format_shift"], rate=1.0, seed=11, turns=6)
    g = WordleGame("crane", max_guesses=6, injector=inj)
    obs = g.guess("slate")
    # Whatever format it picked, it is NOT the default G/Y/X string.
    assert obs.feedback != "XXGXG"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
