"""LLM pricing tables (internal). Money in MICROS (1 USD = 1,000,000 micros).

All upstream rates below are pre-margin. The runner records both upstream
and billed (with_margin) micros per call; the dashboard shows billed only.

Pricing strategy by family:
  - Claude family → priced from the table below, indexed by upstream id.
  - GPT / Gemini families → priced from upstream-reported `usage.cost` when
    available; otherwise priced from the OPENROUTER_FALLBACK table.

Margin: flat % markup. `with_margin(upstream) = upstream * (1 + margin/100)`.
Margin is configured via env; it is NEVER surfaced in user-facing docs/UI.
"""

from __future__ import annotations

import os

MICROS_PER_DOLLAR = 1_000_000
# LLM tokens carry their own margin, separate from media/web (`PURAS_MARGIN_PCT`).
# Default 0 — text is billed at upstream token cost.
MARGIN_PCT = float(os.environ.get("PURAS_TEXT_MARGIN_PCT", "0"))

# (input_per_million_micros, output_per_million_micros)
# $5/MTok input  → 5 * 1_000_000 = 5_000_000 micros per 1M tokens
ANTHROPIC = {
    # Opus 4.8 — $5 / $25
    "claude-opus-4-8":             ( 5_000_000, 25_000_000),
    # Sonnet 4.x — $3 / $15
    "claude-sonnet-4-6":           ( 3_000_000, 15_000_000),
    "claude-sonnet-4-5":           ( 3_000_000, 15_000_000),
    "claude-sonnet-4-20250514":    ( 3_000_000, 15_000_000),
    # Haiku 4.5 — $0.25 / $1.25
    "claude-haiku-4-5":            (   250_000,  1_250_000),
    "claude-haiku-4-5-20251001":   (   250_000,  1_250_000),
}

# Fallback when model is unknown — use Sonnet pricing (most common, mid-tier)
_UNKNOWN_FALLBACK = (3_000_000, 15_000_000)


def anthropic_cost_micros(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> int:
    """Anthropic upstream cost in micros, accounting for prompt-cache tokens.

    Anthropic's reported `input_tokens` excludes both cache_creation and
    cache_read counts — we bill each at its own multiplier of the model's
    base input rate:
      - uncached input    → 1.0× input_rate
      - cache write (1h)  → 2.0× input_rate
      - cache read (hit)  → 0.1× input_rate
      - output            → output_rate (unchanged by caching)

    The 2.0× write multiplier matches the 1-HOUR cache TTL set on all
    cache_control breakpoints (see providers.anthropic_provider._CACHE_CONTROL).
    If that TTL ever reverts to the 5-minute default, drop this back to 1.25×.
    """
    in_rate, out_rate = ANTHROPIC.get(model, _UNKNOWN_FALLBACK)
    cost = (
        input_tokens * in_rate
        + cache_creation_input_tokens * in_rate * 2.0
        + cache_read_input_tokens * in_rate * 0.1
        + output_tokens * out_rate
    ) / 1_000_000
    return int(round(cost))


def with_margin(upstream_micros: int) -> int:
    return int(round(upstream_micros * (1.0 + MARGIN_PCT / 100.0)))


def usd_from_micros(micros: int) -> float:
    return micros / MICROS_PER_DOLLAR


def micros_from_usd(usd: float) -> int:
    return int(round(usd * MICROS_PER_DOLLAR))
