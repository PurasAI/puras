"""The steelman LangGraph contestant: create_react_agent + a real long-term
memory store (LangGraph's BaseStore via InMemoryStore), persisting lessons across
rounds. This is the honest, built-to-win opponent — it has memory, it learns the
client's rulebook from eval feedback, exactly like the Puras side.

One InMemoryStore is created per *session* and shared across rounds, so what the
agent remembers in round 1 is available in round 5 — cross-round learning, which
is what the experiment measures.
"""

from __future__ import annotations

import time
import uuid

from task import Brief, grade
from contestants.shared import BASE_INSTRUCTIONS, brief_message, feedback_payload


def run_langgraph_session(
    briefs: list[Brief],
    *,
    model: str = "claude-haiku-4-5",
    instructions: str = BASE_INSTRUCTIONS,
):
    """Run one contestant session over a list of briefs. Returns
    (scores_per_round, total_cost_micros, error_or_None)."""
    import json as _json

    from langchain_anthropic import ChatAnthropic
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()           # long-term memory, shared across rounds
    NS = ("client", "lessons")
    llm = ChatAnthropic(model=model, temperature=0, max_tokens=1024)

    scores: list[float] = []
    cost_micros = 0
    error = None

    for i, brief in enumerate(briefs):
        round_state: dict = {"scored": False, "score": 0.0}

        @tool
        def recall(query: str = "") -> str:
            """Retrieve lessons learned about this client's style guide."""
            items = store.search(NS)
            return _json.dumps([it.value.get("lesson", "") for it in items]) or "[]"

        @tool
        def remember(lesson: str) -> str:
            """Save one durable lesson about the client's style for future briefs."""
            store.put(NS, str(uuid.uuid4()), {"lesson": lesson})
            return "saved"

        @tool
        def submit_copy(text: str) -> str:
            """Submit the finished copy. Returns a score in [0,1] and which of the
            client's rules it broke. Only the first submission per brief is scored."""
            score, failed = grade(text, brief)
            if not round_state["scored"]:
                round_state["scored"] = True
                round_state["score"] = score
            return _json.dumps(feedback_payload(score, failed))

        agent = create_react_agent(llm, [recall, remember, submit_copy], prompt=instructions)
        try:
            result = agent.invoke(
                {"messages": [("user", brief_message(brief, i))]},
                config={"recursion_limit": 60},
            )
            for m in result.get("messages", []):
                um = getattr(m, "usage_metadata", None) or {}
                cost_micros += um.get("input_tokens", 0) * 1 + um.get("output_tokens", 0) * 5
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

        scores.append(round_state["score"])

    return scores, cost_micros, error
