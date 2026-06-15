"""The "mischievous host" — the edge-case injector (DESIGN.md §5).

Fairness rule that everything here exists to protect: the perturbation schedule
is **pre-computed from the seed before any play**, independent of either
player's behaviour. Both contestants replaying the same seed therefore meet the
same surprises at the same turn index. A rejected guess doesn't advance the turn
index, so a player simply faces the same turn's constraint again — consistent
for both. Nothing here ever tells a player "a rule changed"; they infer it from
the feedback or they don't.

Each perturbation precomputes, for every turn, either ``None`` (dormant) or a
small ``params`` dict fixing exactly how it fires that turn. Then the engine
calls three optional hooks on the active ones:

  validate(guess, params, ctx) -> (accepted, reason)   # P4 new constraint
  mutate(marks, guess, params, ctx) -> marks           # P3 noise, P5 lie, P7 rule swap
  render(marks, guess, params, ctx) -> str | None      # P1 format shift
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from .protocol import Mark
from .words import score as true_score


@dataclass
class Ctx:
    """Read-only-ish context handed to perturbation hooks."""

    secret: str
    word_length: int
    turn_index: int


class Perturbation:
    """Base class. Default schedule is *per-turn*: each turn fires independently
    with probability ``rate``. Hooks are no-ops; subclasses override what they need.
    """

    name = "base"

    def schedule(self, rng: random.Random, turns: int, rate: float) -> list[dict | None]:
        return [self.params(rng) if rng.random() < rate else None for _ in range(turns)]

    def params(self, rng: random.Random) -> dict:
        return {}

    # --- hooks (override the relevant one) ---------------------------------
    def validate(self, guess: str, params: dict, ctx: Ctx) -> tuple[bool, str | None]:
        return True, None

    def mutate(self, marks: tuple[Mark, ...], guess: str, params: dict, ctx: Ctx) -> tuple[Mark, ...]:
        return marks

    def render(self, marks: tuple[Mark, ...], guess: str, params: dict, ctx: Ctx) -> str | None:
        return None


# --- scope helpers ---------------------------------------------------------

class _Latching(Perturbation):
    """Fires once (prob ``rate`` each turn until it triggers), then stays on for
    the rest of the game with the params it rolled at trigger time. Models a
    rule that changes mid-game and *stays* changed."""

    def schedule(self, rng, turns, rate):
        out: list[dict | None] = []
        live: dict | None = None
        for _ in range(turns):
            if live is None and rng.random() < rate:
                live = self.params(rng)
            out.append(live)
        return out


# ---------------------------------------------------------------------------
# P1 — Format shift. The feedback's *representation* changes; its meaning
# doesn't. A parser hard-wired to the default G/Y/X string breaks; a reader
# that understands what the host is saying adapts.
# ---------------------------------------------------------------------------

_SYM = {Mark.HIT: "G", Mark.PRESENT: "Y", Mark.MISS: "X"}
_EMOJI = {Mark.HIT: "🟩", Mark.PRESENT: "🟨", Mark.MISS: "⬛"}
_WORDS = {Mark.HIT: "hit", Mark.PRESENT: "present", Mark.MISS: "miss"}


def render_default(marks: tuple[Mark, ...]) -> str:
    """The format announced in the GameView legend: one char per position."""
    return "".join(_SYM[m] for m in marks)


class FormatShift(_Latching):
    name = "format_shift"
    _FORMATS = ("emoji", "words", "json", "reversed")

    def params(self, rng):
        return {"format": rng.choice(self._FORMATS)}

    def render(self, marks, guess, params, ctx):
        fmt = params["format"]
        if fmt == "emoji":
            return "".join(_EMOJI[m] for m in marks)
        if fmt == "words":
            return ",".join(_WORDS[m] for m in marks)
        if fmt == "json":
            return json.dumps([{"letter": g, "mark": _WORDS[m]} for g, m in zip(guess, marks)])
        if fmt == "reversed":  # same default symbols, position order flipped
            return render_default(tuple(reversed(marks)))
        return None


# ---------------------------------------------------------------------------
# P3 — Noisy hint. With some chance a single position is reported wrong. A
# solver that treats every clue as gospel eliminates candidates it shouldn't and
# can paint itself into an empty set; a player that holds clues loosely survives.
# ---------------------------------------------------------------------------

class Noise(Perturbation):
    name = "noise"

    def params(self, rng):
        return {"pos": rng.random(), "to": rng.random()}

    def mutate(self, marks, guess, params, ctx):
        if not marks:
            return marks
        i = int(params["pos"] * len(marks))
        cur = marks[i]
        others = [m for m in (Mark.HIT, Mark.PRESENT, Mark.MISS) if m is not cur]
        flipped = others[int(params["to"] * len(others))]
        out = list(marks)
        out[i] = flipped
        return tuple(out)


# ---------------------------------------------------------------------------
# P4 — New constraint, announced only by rejecting guesses that violate it. The
# rejection message is in the *current* format. A fixed graph keeps re-proposing
# the same now-illegal guess and stalls; a reader updates its candidate filter.
# ---------------------------------------------------------------------------

class NewConstraint(_Latching):
    name = "new_constraint"
    _RULES = ("no_repeats", "must_contain_e", "no_vowel_start")

    def params(self, rng):
        return {"rule": rng.choice(self._RULES)}

    def validate(self, guess, params, ctx):
        rule = params["rule"]
        if rule == "no_repeats" and len(set(guess)) != len(guess):
            return False, "rejected: repeated letters are no longer allowed"
        if rule == "must_contain_e" and "e" not in guess:
            return False, "rejected: guesses must contain the letter E"
        if rule == "no_vowel_start" and guess[:1] in "aeiou":
            return False, "rejected: guesses may not start with a vowel"
        return True, None


# ---------------------------------------------------------------------------
# P5 — Temporary lie. For a window of turns the host systematically corrupts
# feedback (swap hit<->miss), then silently reverts. Inconsistent-looking clues
# lock up a deductive solver; an adaptive player learns to distrust the streak.
# Win is still judged truthfully (guess == secret), so a lie can't deny a win.
# ---------------------------------------------------------------------------

class TemporaryLie(Perturbation):
    name = "temporary_lie"

    def schedule(self, rng, turns, rate):
        out: list[dict | None] = [None] * turns
        if turns and rng.random() < rate:
            start = rng.randrange(turns)
            length = rng.randint(1, max(1, turns // 3))
            for t in range(start, min(turns, start + length)):
                out[t] = {"mode": "swap_hit_miss"}
        return out

    def mutate(self, marks, guess, params, ctx):
        swap = {Mark.HIT: Mark.MISS, Mark.MISS: Mark.HIT, Mark.PRESENT: Mark.PRESENT}
        return tuple(swap[m] for m in marks)


# ---------------------------------------------------------------------------
# P7 — Silent rule change. Scoring semantics shift from positional to
# presence-only: every letter that occurs anywhere in the secret reads HIT, the
# rest MISS — no positional information at all. A position-based filter quietly
# builds a wrong model of the world; a player that re-reads the evidence notices
# greens stopped meaning "right spot".
# ---------------------------------------------------------------------------

class SilentRuleChange(_Latching):
    name = "silent_rule_change"

    def mutate(self, marks, guess, params, ctx):
        secret = set(ctx.secret)
        return tuple(Mark.HIT if g in secret else Mark.MISS for g in guess)


# ---------------------------------------------------------------------------
# The injector: composes enabled perturbations into one pre-computed schedule.
# ---------------------------------------------------------------------------

REGISTRY: dict[str, type[Perturbation]] = {
    cls.name: cls
    for cls in (FormatShift, Noise, NewConstraint, TemporaryLie, SilentRuleChange)
}

# P2 (length change) and P6 (alphabet expansion) from the taxonomy are not yet
# wired here — see DESIGN.md §5 / §11. They need variable-length scoring and a
# wider symbol set; tracked as follow-ups so the first bake-off can ship.


@dataclass
class FiredPerturbation:
    name: str
    params: dict


class Injector:
    """Holds the per-turn plan for one game. Built once from a seed; pure after."""

    def __init__(self, names: list[str], rate: float, seed: int, turns: int):
        self.rate = rate
        self.turns = turns
        rng = random.Random(seed)
        # One schedule list per perturbation, length == turns.
        self._plans: list[tuple[Perturbation, list[dict | None]]] = []
        for nm in names:
            pert = REGISTRY[nm]()
            self._plans.append((pert, pert.schedule(rng, turns, rate)))

    def active(self, turn_index: int) -> list[tuple[Perturbation, dict]]:
        out = []
        for pert, plan in self._plans:
            if turn_index < len(plan) and plan[turn_index] is not None:
                out.append((pert, plan[turn_index]))
        return out
