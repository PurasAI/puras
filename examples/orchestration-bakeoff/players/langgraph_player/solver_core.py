"""The deterministic Wordle solver's *logic* — pure, no graph, no LLM.

This is the steelman (DESIGN.md §3.2): on an unperturbed game it plays
near-optimally — a fixed strong opener, then information-maximising guesses
restricted to the remaining candidate set. There is deliberately **no LLM here**;
the honest, strongest form of deterministic orchestration for Wordle is an
algorithm. The Puras side pays for an LLM agent; that cost gap is real and we
report it rather than hide it.

The brittleness that the bake-off exposes is *structural*, not a strawman:

  * ``parse_feedback`` understands exactly one format — the default G/Y/X legend.
    Anything else raises. (P1 format shift)
  * ``filter_candidates`` treats every clue as gospel positional truth, so noisy,
    lying, or semantically-shifted feedback drives the candidate set to empty —
    a contradiction the deductive model cannot represent. (P3/P5/P7)
  * It has no concept of a guess being *rejected*, so a new constraint makes it
    re-propose the same illegal word until the attempt cap loses the game. (P4)

Each of those is the natural consequence of fixing the pipeline up front — the
exact thing the agent side doesn't do.
"""

from __future__ import annotations

import math

from engine.protocol import Mark
from engine.words import answers, score

_PARSE = {"G": Mark.HIT, "Y": Mark.PRESENT, "X": Mark.MISS}
STRONG_OPENER = "slate"  # a conventional high-coverage Wordle opener


class ContradictionError(RuntimeError):
    """Every candidate has been eliminated — the clues are mutually impossible
    under positional Wordle logic. A deductive solver cannot proceed; an
    adaptive player would instead question whether a clue was a lie/shift."""


def parse_feedback(feedback: str, word_length: int) -> tuple[Mark, ...]:
    """Parse the host's reply in the ONE format this solver knows. Raises on
    anything else — which is precisely what a format shift triggers."""
    s = feedback.strip()
    if len(s) != word_length or any(ch not in _PARSE for ch in s):
        raise ValueError(f"unrecognized feedback format: {feedback!r}")
    return tuple(_PARSE[ch] for ch in s)


def filter_candidates(candidates: list[str], guess: str, marks: tuple[Mark, ...]) -> list[str]:
    """Keep only words that would have produced exactly these marks. Uses the
    canonical scorer, so all Wordle constraints fall out for free."""
    return [c for c in candidates if score(guess, c) == marks]


def _entropy_pick(candidates: list[str]) -> str:
    """Pick the candidate whose feedback distribution splits the rest most evenly
    (max Shannon entropy). Restricted to candidates ('hard mode') — fast and
    still strong. For large sets, fall back to a letter-frequency heuristic so
    a batch of hundreds of games stays cheap."""
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 300:
        freq: dict[str, int] = {}
        for w in candidates:
            for ch in set(w):
                freq[ch] = freq.get(ch, 0) + 1
        return max(candidates, key=lambda w: sum(freq.get(ch, 0) for ch in set(w)))

    best, best_h = candidates[0], -1.0
    n = len(candidates)
    for g in candidates:
        buckets: dict[tuple, int] = {}
        for c in candidates:
            k = score(g, c)
            buckets[k] = buckets.get(k, 0) + 1
        h = -sum((b / n) * math.log2(b / n) for b in buckets.values())
        if h > best_h:
            best, best_h = g, h
    return best


def pick_guess(candidates: list[str], turn_index: int) -> str:
    """The graph's terminal node: fixed opener on turn 0, else max-info guess."""
    if turn_index == 0:
        return STRONG_OPENER
    if not candidates:
        raise ContradictionError("no candidate words remain consistent with the clues")
    return _entropy_pick(candidates)


def initial_candidates() -> list[str]:
    return list(answers())
