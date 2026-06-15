"""The *agentic* LangGraph contestant — a real ReAct agent (LLM + tools).

This exists to answer the honest question the deterministic StateGraph can't:
LangGraph isn't only the hand-drawn-graph extreme — its prebuilt
``create_react_agent`` gives you exactly the same kind of free-form, tool-calling
agent loop Puras runs. So "agentic vs deterministic" is a spectrum *within*
LangGraph, not Puras-vs-LangGraph.

To keep the match fair, this agent gets the SAME model (Haiku), the SAME single
``make_guess`` tool semantics, and the SAME instructions as the Puras skill
(loaded from its SKILL.md). The only thing that differs from the Puras side is
the runtime/packaging — which is the real point of comparison. If this agent and
the Puras skill perform alike, the lesson is "agentic beats deterministic," and
Puras's value is making the agentic path a packaged, eval-gated, prod-parity
default rather than something you wire and operate yourself.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from engine.experiment import build_game
from engine.protocol import Status
from engine.scoring import GameResult

_SKILL_MD = Path(__file__).resolve().parents[1] / "puras_player/codebreaker/SKILL.md"

# Approximate published Claude Haiku 4.5 rate (USD per 1M tokens), for a rough
# cost column only — win-rate is the metric that matters.
_PRICE_IN, _PRICE_OUT = 1.00, 5.00


def _system_prompt() -> str:
    """Reuse the Puras skill's instructions verbatim (minus the frontmatter and
    the set_output-specific 'Finishing' section, which is Puras-runner-only)."""
    text = _SKILL_MD.read_text()
    text = re.sub(r"^---.*?---\s*", "", text, count=1, flags=re.DOTALL)  # drop frontmatter
    text = re.split(r"\n##\s+Finishing", text)[0].strip()               # drop set_output bit
    return text + (
        "\n\n## Finishing\n\nWhen the reply's `status` is `won` or `lost`, stop — "
        "do not call make_guess again; reply with the final word."
    )


def run_react_game(
    seed: int,
    config_id: int,
    *,
    max_guesses: int = 6,
    model: str = "claude-haiku-4-5",
    rate: float = 0.0,
) -> GameResult:
    # Imported lazily so the rest of the bake-off doesn't require these packages.
    from langchain_anthropic import ChatAnthropic
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    game = build_game(seed, config_id, max_guesses=max_guesses)

    @tool
    def make_guess(guess: str) -> str:
        """Submit one guess to the host and get its reply (accepted, feedback,
        status, guesses_made, guesses_left)."""
        obs = game.guess(guess or "")
        return json.dumps({
            "accepted": obs.accepted, "feedback": obs.feedback,
            "status": obs.status.value, "guesses_made": obs.guesses_made,
            "guesses_left": obs.guesses_left,
        })

    llm = ChatAnthropic(model=model, temperature=0, max_tokens=1024)
    agent = create_react_agent(llm, [make_guess], prompt=_system_prompt())

    t0 = time.perf_counter()
    error = None
    tok_in = tok_out = 0
    try:
        result = agent.invoke(
            {"messages": [("user", "Let's play. Make your first guess.")]},
            config={"recursion_limit": 120},
        )
        for m in result.get("messages", []):
            um = getattr(m, "usage_metadata", None) or {}
            tok_in += um.get("input_tokens", 0)
            tok_out += um.get("output_tokens", 0)
    except Exception as exc:  # noqa: BLE001 — recursion cap / API error = recorded loss
        error = f"{type(exc).__name__}: {exc}"

    cost_micros = int((tok_in * _PRICE_IN + tok_out * _PRICE_OUT))  # per-Mtok → micros
    return GameResult(
        player="langgraph-react", secret=game.secret, seed=seed, rate=rate,
        won=game.status is Status.WON, guesses_used=game.guesses_made,
        attempts=game.attempts, perturbed=config_id != 0,
        elapsed_s=time.perf_counter() - t0, cost_micros=cost_micros, error=error,
    )
