"""Word lists + the canonical Wordle scoring rule.

Data is the official Wordle pair (2315 answers, 10657 allowed guesses), bundled
under ``data/`` so the engine stays offline and dependency-light — the same
spirit as the Puras local runner.
"""

from __future__ import annotations

import functools
from pathlib import Path

from .protocol import Mark

_DATA = Path(__file__).parent / "data"


@functools.lru_cache(maxsize=None)
def answers() -> tuple[str, ...]:
    return tuple((_DATA / "answers.txt").read_text().split())


@functools.lru_cache(maxsize=None)
def allowed_guesses() -> frozenset[str]:
    """Everything a player may legally guess: the allowed list plus the answers."""
    extra = (_DATA / "allowed_guesses.txt").read_text().split()
    return frozenset(answers()) | frozenset(extra)


def score(guess: str, secret: str) -> tuple[Mark, ...]:
    """Canonical Wordle marks, handling duplicate letters the real way.

    Two passes: greens first (so a duplicate letter already placed isn't also
    counted as yellow), then yellows against the remaining unmatched letters.
    """
    guess, secret = guess.lower(), secret.lower()
    marks: list[Mark] = [Mark.MISS] * len(guess)
    # Count letters still available for a "present" match after greens are taken.
    leftover: dict[str, int] = {}
    for g, s in zip(guess, secret):
        if g == s:
            continue
        leftover[s] = leftover.get(s, 0) + 1
    for i, (g, s) in enumerate(zip(guess, secret)):
        if g == s:
            marks[i] = Mark.HIT
    for i, g in enumerate(guess):
        if marks[i] is Mark.HIT:
            continue
        if leftover.get(g, 0) > 0:
            marks[i] = Mark.PRESENT
            leftover[g] -= 1
    return tuple(marks)
