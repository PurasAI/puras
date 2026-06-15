"""Naive deterministic player — a real LangGraph ``StateGraph``.

This is the "deterministic orchestration" contestant. The solving strategy is
expressed as a fixed graph: ``parse_filter → (pick | contradiction)``. The flow
is drawn up front; the model is never asked to decide what to do when something
unexpected happens — there simply is no node for it. That is the whole thesis,
made concrete: on a clean game this graph is excellent; the moment the host does
something the graph's author didn't anticipate, it parses-errors, hits an empty
candidate set, or re-proposes a rejected word forever.

The Puras side, by contrast, hands the same situation to an agent loop that can
re-read the evidence and change course (see ../puras_player).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

# Let this file import the bundled engine whether run from the harness or alone.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, START, StateGraph  # noqa: E402

from engine.protocol import GameView, Status, Turn  # noqa: E402
from . import solver_core as sc  # noqa: E402


class _State(TypedDict):
    history: list
    word_length: int
    turn_index: int
    candidates: list
    guess: str


def _parse_filter(state: _State) -> dict:
    """Rebuild the candidate set from the full accepted history. A parser that
    only knows the default format raises here on a format shift; a poisoned clue
    (noise/lie/rule-shift) silently filters the set toward empty."""
    candidates = sc.initial_candidates()
    accepted = 0
    for turn in state["history"]:
        obs = turn.observation
        if not obs.accepted:
            continue  # the graph has no model for a rejection; it just ignores it
        marks = sc.parse_feedback(obs.feedback, state["word_length"])
        candidates = sc.filter_candidates(candidates, turn.guess, marks)
        accepted += 1
    return {"candidates": candidates, "turn_index": accepted}


def _route(state: _State) -> str:
    if state["turn_index"] > 0 and not state["candidates"]:
        return "contradiction"
    return "pick"


def _pick(state: _State) -> dict:
    return {"guess": sc.pick_guess(state["candidates"], state["turn_index"])}


def _contradiction(state: _State) -> dict:
    raise sc.ContradictionError("no candidate words remain consistent with the clues")


def _build_graph():
    g = StateGraph(_State)
    g.add_node("parse_filter", _parse_filter)
    g.add_node("pick", _pick)
    g.add_node("contradiction", _contradiction)
    g.add_edge(START, "parse_filter")
    g.add_conditional_edges("parse_filter", _route,
                            {"pick": "pick", "contradiction": "contradiction"})
    g.add_edge("pick", END)
    g.add_edge("contradiction", END)
    return g.compile()


class LangGraphSolver:
    """Adapts the compiled StateGraph to the bake-off ``Player`` protocol."""

    def __init__(self):
        self.name = "langgraph-naive"
        self._app = _build_graph()
        self._word_length = 5

    def reset(self, view: GameView) -> None:
        self._word_length = view.word_length

    def next_guess(self, history: list[Turn]) -> str:
        out = self._app.invoke({
            "history": history,
            "word_length": self._word_length,
            "turn_index": 0,
            "candidates": [],
            "guess": "",
        })
        return out["guess"]
