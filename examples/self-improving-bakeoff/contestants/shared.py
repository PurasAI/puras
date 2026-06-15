"""The shared round protocol + base instructions both contestants use.

Fairness: both get the SAME model, the SAME three tools (recall / remember /
submit_copy), the SAME base instructions, and the SAME one-scored-submission rule.
The only thing the optimizer (P2) changes is the Puras skill's prompt — and only
the Puras side — which is the whole point of the experiment.

The learning is *cross-round*: a contestant submits once per brief (so the score
reflects what it knew before submitting), then sees the feedback and may store
lessons. Memory carries those lessons to later briefs, so a contestant that learns
the hidden rulebook climbs over the rounds.
"""

from __future__ import annotations

from task import Brief

# Shared pricing so both contestants' cost is computed identically (apples-to-apples).
# Claude Haiku 4.5 published rate: $1 / Mtok input, $5 / Mtok output. Both sides run
# Haiku, so the cost RATIO is exact regardless of the absolute rate.
HAIKU_USD_PER_MTOK_IN = 1.0
HAIKU_USD_PER_MTOK_OUT = 5.0


def usd(tok_in: int, tok_out: int) -> float:
    return (tok_in * HAIKU_USD_PER_MTOK_IN + tok_out * HAIKU_USD_PER_MTOK_OUT) / 1_000_000

# Base instructions — deliberately do NOT reveal the rulebook. This is the prompt
# the optimizer will iterate (for the Puras side only). Kept identical across
# contestants at the start so the curves are comparable.
BASE_INSTRUCTIONS = """\
You are a copywriter for a client who enforces a specific, consistent style guide \
that you must figure out over time. You are NOT told the rules up front.

For each brief, do exactly this:
1. Call recall to retrieve any lessons you've already learned about this client.
2. Write one marketing copy for the product, applying everything you recalled.
3. Call submit_copy(copy) EXACTLY ONCE. You get a score in [0,1] and feedback \
listing any of the client's rules your copy broke.
4. For each broken rule in the feedback, call remember with a short, durable \
lesson so future briefs score higher. Then stop.

You only get one scored submission per brief, so rely on what you remember."""


def brief_message(brief: Brief, round_index: int) -> str:
    return (
        f"Brief #{round_index + 1}. Write marketing copy for {brief.product}. "
        f"Feature to highlight: {brief.fact}. A relevant figure: {brief.number}."
    )


def feedback_payload(score: float, failed: list[str]) -> dict:
    return {"score": round(score, 3),
            "broken_rules": failed,
            "perfect": not failed}
