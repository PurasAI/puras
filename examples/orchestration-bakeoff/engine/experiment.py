"""The experiment grid — the one table the harness and the Puras tool share.

A fairness subtlety (DESIGN.md §3.7): no contestant may be told that
perturbations are coming. The naive LangGraph solver has zero perturbation
awareness, so the agent must not either. But the ``make_guess`` tool *does* need
the perturbation spec to drive the host. We square that by handing the agent
only two opaque integers — ``seed`` and ``config`` — and keeping the mapping
from ``config`` to (rate, perturbations) here, in code the tool imports but the
agent never sees. The agent learns nothing about the host's mischief from its
inputs; it has to notice it in the replies, exactly like the deterministic side.

Both contestants build their game from the *same* (seed, config), so they face
the identical secret and the identical pre-computed perturbation schedule.
"""

from __future__ import annotations

from .game import WordleGame, new_game

# Canonical order — must match on both sides so the seeded schedule is identical.
ALL_PERTURBATIONS = [
    "format_shift", "noise", "new_constraint", "temporary_lie", "silent_rule_change",
]

CONFIGS: list[dict] = [
    {"id": 0, "label": "clean",    "rate": 0.0,  "perturbations": []},
    {"id": 1, "label": "mild",     "rate": 0.10, "perturbations": ALL_PERTURBATIONS},
    {"id": 2, "label": "moderate", "rate": 0.25, "perturbations": ALL_PERTURBATIONS},
    {"id": 3, "label": "harsh",    "rate": 0.50, "perturbations": ALL_PERTURBATIONS},
]


def build_game(seed: int, config_id: int, *, max_guesses: int = 6) -> WordleGame:
    """Deterministically build the game for a (seed, config) cell. Called by both
    the in-process LangGraph harness and the Puras tool subprocess — same args in,
    byte-identical game out."""
    cfg = CONFIGS[config_id]
    return new_game(
        seed=seed,
        perturbations=cfg["perturbations"],
        rate=cfg["rate"],
        max_guesses=max_guesses,
    )
