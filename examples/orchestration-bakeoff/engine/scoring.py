"""Run a player through a game and aggregate results into the bake-off metrics.

The story metric (DESIGN.md §6) is the robustness curve: win-rate as a function
of perturbation rate. We also track recovery-rate — the share of post-shock
turns where a player that was knocked off course gets back on it — because that
is the cleanest measure of "solved an unforeseen problem".
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from .game import WordleGame
from .protocol import Player, Status


@dataclass
class GameResult:
    player: str
    secret: str
    seed: int
    rate: float
    won: bool
    guesses_used: int          # accepted guesses
    attempts: int              # accepted + rejected
    perturbed: bool            # did any perturbation fire this game?
    elapsed_s: float = 0.0
    error: str | None = None   # player blew up (e.g. a parser exception)
    history: list = field(default_factory=list)


def play(game: WordleGame, player: Player, *, seed: int = 0, rate: float = 0.0) -> GameResult:
    """Drive one game to its end. A player that raises is recorded as a loss with
    the error captured — a deterministic parser dying on a shifted format is a
    legitimate, informative outcome, not a crash we hide."""
    t0 = time.perf_counter()
    error = None
    player.reset(game.view())
    try:
        while game.status is Status.ONGOING:
            guess = player.next_guess(list(game.history))
            game.guess(guess)
    except Exception as exc:  # noqa: BLE001 — the loss reason IS the data
        error = f"{type(exc).__name__}: {exc}"
        if game.status is Status.ONGOING:
            game.status = Status.LOST

    return GameResult(
        player=getattr(player, "name", player.__class__.__name__),
        secret=game.secret,
        seed=seed,
        rate=rate,
        won=game.status is Status.WON,
        guesses_used=game.guesses_made,
        attempts=game.attempts,
        perturbed=any(t.perturbations_fired for t in game.history),
        elapsed_s=time.perf_counter() - t0,
        error=error,
        history=game.history,
    )


def summarize(results: list[GameResult]) -> dict:
    """Aggregate a batch (typically one player at one perturbation rate)."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    wins = [r for r in results if r.won]
    win_guesses = [r.guesses_used for r in wins]
    return {
        "n": n,
        "win_rate": len(wins) / n,
        "errors": sum(1 for r in results if r.error),
        "avg_guesses_on_win": round(statistics.mean(win_guesses), 2) if win_guesses else None,
        "avg_attempts": round(statistics.mean(r.attempts for r in results), 2),
        "avg_elapsed_s": round(statistics.mean(r.elapsed_s for r in results), 4),
    }
