"""Player-agnostic Wordle bake-off engine (offline, stdlib-only).

The engine and its perturbation injector are independent of the players: both
contestants talk to a ``WordleGame`` solely through ``protocol`` types, and the
"mischievous host" schedule is pre-computed from the seed (see DESIGN.md §3).
"""

from .game import WordleGame, new_game
from .protocol import GameView, Mark, Observation, Player, Status, Turn
from .scoring import GameResult, play, summarize
from . import words

__all__ = [
    "WordleGame", "new_game",
    "GameView", "Mark", "Observation", "Player", "Status", "Turn",
    "GameResult", "play", "summarize",
    "words",
]
