"""The Wordle game engine — player-agnostic, perturbation-aware.

One game = one secret + a pre-computed perturbation schedule. The engine is the
single shared environment both contestants talk to through the protocol; it has
no idea who is playing. Win is decided by truthful equality (``guess == secret``)
regardless of any lie or rule shift, so the score stays objective even when the
host is misbehaving.
"""

from __future__ import annotations

import random

from . import words
from .perturbations import Ctx, Injector, render_default
from .protocol import GameView, Observation, Status, Turn

_LEGEND = (
    "Feedback is one symbol per letter, left to right: "
    "G = right letter, right spot; Y = right letter, wrong spot; X = not in the word."
)


class WordleGame:
    def __init__(
        self,
        secret: str,
        *,
        max_guesses: int = 6,
        injector: Injector | None = None,
        max_attempts: int | None = None,
    ):
        self.secret = secret.lower()
        self.word_length = len(self.secret)
        self.max_guesses = max_guesses
        self.injector = injector
        # Rejected guesses don't consume a guess, so a stuck player could loop
        # forever. Cap total attempts (accepted + rejected) to force a loss.
        self.max_attempts = max_attempts if max_attempts is not None else max_guesses * 4

        self.guesses_made = 0
        self.attempts = 0
        self.status = Status.ONGOING
        self.history: list[Turn] = []

    def view(self) -> GameView:
        """The honest, perturbation-free rules a player is told at the start."""
        return GameView(
            word_length=self.word_length,
            max_guesses=self.max_guesses,
            alphabet="abcdefghijklmnopqrstuvwxyz",
            feedback_legend=_LEGEND,
            note="A standard game of Wordle. Read each reply carefully.",
        )

    def guess(self, raw: str) -> Observation:
        if self.status is not Status.ONGOING:
            return self._obs(True, "game over", )

        guess = (raw or "").strip().lower()
        self.attempts += 1
        ctx = Ctx(secret=self.secret, word_length=self.word_length, turn_index=self.guesses_made)
        active = self.injector.active(self.guesses_made) if self.injector else []
        fired = [f"{p.name}:{params}" for p, params in active]

        # --- legality: built-in rules, then any active constraint -----------
        reason = self._builtin_reject(guess)
        if reason is None:
            for pert, params in active:
                ok, why = pert.validate(guess, params, ctx)
                if not ok:
                    reason = why
                    break
        if reason is not None:
            obs = self._obs(False, self._render(reason, active, None, guess, ctx))
            self._record(guess, obs, None, fired)
            if self.attempts >= self.max_attempts:
                self.status = Status.LOST
                obs = self._obs(False, obs.feedback)  # refresh status field
            return obs

        # --- accepted: truth first (win is never perturbed) -----------------
        true_marks = words.score(guess, self.secret)
        self.guesses_made += 1
        if guess == self.secret:
            self.status = Status.WON
            obs = self._obs(True, self._render(None, active, true_marks, guess, ctx))
            self._record(guess, obs, true_marks, fired)
            return obs

        # --- reported marks: apply content perturbations --------------------
        reported = true_marks
        for pert, params in active:
            reported = pert.mutate(reported, guess, params, ctx)

        if self.guesses_made >= self.max_guesses or self.attempts >= self.max_attempts:
            self.status = Status.LOST
        obs = self._obs(True, self._render(None, active, reported, guess, ctx))
        self._record(guess, obs, true_marks, fired)
        return obs

    # --- helpers -----------------------------------------------------------

    def _builtin_reject(self, guess: str) -> str | None:
        if len(guess) != self.word_length:
            return f"rejected: a guess must be exactly {self.word_length} letters"
        if not guess.isalpha():
            return "rejected: letters only"
        if guess not in words.allowed_guesses():
            return "rejected: not in the word list"
        return None

    def _render(self, reason, active, marks, guess, ctx) -> str:
        if reason is not None:
            # Rejection text: shift its format too, if a format-shift is active,
            # by appending the host's current style cue is overkill — keep plain.
            return reason
        # Let an active format-shift perturbation own the rendering.
        for pert, params in active:
            r = pert.render(marks, guess, params, ctx)
            if r is not None:
                return r
        return render_default(marks)

    def _obs(self, accepted: bool, feedback: str) -> Observation:
        return Observation(
            accepted=accepted,
            feedback=feedback,
            status=self.status,
            guesses_made=self.guesses_made,
            guesses_left=self.max_guesses - self.guesses_made,
        )

    def _record(self, guess, obs, true_marks, fired):
        self.history.append(Turn(guess=guess, observation=obs, true_marks=true_marks,
                                 perturbations_fired=fired))


def new_game(
    *,
    seed: int,
    perturbations: list[str] | None = None,
    rate: float = 0.0,
    max_guesses: int = 6,
    secret: str | None = None,
) -> WordleGame:
    """Build a reproducible game from a seed: picks the secret and the schedule."""
    rng = random.Random(seed)
    secret = secret or rng.choice(words.answers())
    injector = None
    if perturbations and rate > 0:
        injector = Injector(perturbations, rate, seed, max_guesses)
    return WordleGame(secret, max_guesses=max_guesses, injector=injector)
