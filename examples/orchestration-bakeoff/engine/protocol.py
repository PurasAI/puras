"""The contract every player sees — and the only thing they see.

The whole bake-off hinges on one rule (DESIGN.md §3): the game engine and the
perturbation injector are independent of the players. A player interacts with
the game *only* through these types. Crucially, an ``Observation`` carries the
host's rendered feedback string but **never a flag saying "a rule changed this
turn."** If the host shifts the feedback format, lies, or adds a constraint, the
player has to notice it from the feedback alone — exactly the situation a fixed
graph can't model and an agent can.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class Mark(enum.Enum):
    """The canonical, ground-truth score for one letter of a guess.

    This is the engine's internal truth. What the *player* receives is a
    rendered string (see ``Observation.feedback``) that may be in a shifted
    format, noisy, or an outright lie — the player never sees ``Mark`` directly.
    """

    HIT = "hit"          # right letter, right position  (Wordle green)
    PRESENT = "present"  # right letter, wrong position  (Wordle yellow)
    MISS = "miss"        # letter not in the word         (Wordle grey)


class Status(enum.Enum):
    ONGOING = "ongoing"
    WON = "won"
    LOST = "lost"


@dataclass(frozen=True)
class GameView:
    """What the host tells a player at the start — the rules as initially stated.

    Deliberately minimal and *honest at t=0*: perturbations are never announced
    here. ``feedback_legend`` documents the DEFAULT feedback format; if the host
    later shifts it, this legend no longer applies and the player must adapt.
    """

    word_length: int
    max_guesses: int
    alphabet: str
    feedback_legend: str
    note: str = ""


@dataclass(frozen=True)
class Observation:
    """The host's reply to one guess. The player's entire window into the game."""

    accepted: bool          # was the guess admitted? (a constraint may reject it)
    feedback: str           # the host's rendered message — feedback OR rejection reason
    status: Status
    guesses_made: int
    guesses_left: int


@dataclass
class Turn:
    """One row of history the harness records (and replays to players)."""

    guess: str
    observation: Observation
    # Ground truth, for scoring/diagnostics only — NEVER handed to a player.
    true_marks: tuple[Mark, ...] | None = None
    perturbations_fired: list[str] = field(default_factory=list)


@runtime_checkable
class Player(Protocol):
    """Both contestants implement this. Same surface, no privileged access."""

    name: str

    def reset(self, view: GameView) -> None:
        """Start a fresh game with the initially-stated rules."""
        ...

    def next_guess(self, history: list[Turn]) -> str:
        """Given everything observed so far, return the next guess."""
