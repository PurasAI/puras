"""Public model registry.

`MODELS` is the single source of truth for which agentic models users may put
in `skill.yaml`'s `model:` field. The public slug (`family/variant`, e.g.
`claude/opus-4-8`, `gpt/5`, `gemini/2.5-pro`) is what users write; the
upstream provider and id are an internal routing concern and MUST NOT leak
into docs, API responses, or error messages.

Upstream pricing is in `pricing.py`; vision/PDF capability is below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PUBLIC_SLUG_RE = re.compile(r"^[a-z][a-z0-9]*/[a-z0-9][a-z0-9._-]*$")

# Default fallback when a skill omits `model:`. Keep this a recognized slug.
DEFAULT_MODEL_SLUG = "claude/sonnet-4-6"


@dataclass(frozen=True)
class ModelInfo:
    slug: str                 # public: what users write
    family: str               # short human name for billing UI / errors
    upstream_provider: str    # internal: "anthropic" | "openrouter"
    upstream_id: str          # internal: what we pass to the provider client
    supports_vision: bool
    supports_pdf: bool


# Public slugs MUST stay stable — they live in customers' skill.yaml.
MODELS: dict[str, ModelInfo] = {
    # ---- Claude ----------------------------------------------------------
    "claude/opus-4-8": ModelInfo(
        slug="claude/opus-4-8", family="Opus 4.8",
        upstream_provider="anthropic", upstream_id="claude-opus-4-8",
        supports_vision=True, supports_pdf=True,
    ),
    "claude/sonnet-4-6": ModelInfo(
        slug="claude/sonnet-4-6", family="Sonnet 4.6",
        upstream_provider="anthropic", upstream_id="claude-sonnet-4-6",
        supports_vision=True, supports_pdf=True,
    ),
    "claude/sonnet-4-5": ModelInfo(
        slug="claude/sonnet-4-5", family="Sonnet 4.5",
        upstream_provider="anthropic", upstream_id="claude-sonnet-4-5",
        supports_vision=True, supports_pdf=True,
    ),
    "claude/haiku-4-5": ModelInfo(
        slug="claude/haiku-4-5", family="Haiku 4.5",
        upstream_provider="anthropic", upstream_id="claude-haiku-4-5",
        supports_vision=True, supports_pdf=True,
    ),

    # ---- GPT (routed through OpenRouter) ---------------------------------
    "gpt/5": ModelInfo(
        slug="gpt/5", family="GPT-5",
        upstream_provider="openrouter", upstream_id="openai/gpt-5",
        supports_vision=True, supports_pdf=False,
    ),
    "gpt/5-mini": ModelInfo(
        slug="gpt/5-mini", family="GPT-5 mini",
        upstream_provider="openrouter", upstream_id="openai/gpt-5-mini",
        supports_vision=True, supports_pdf=False,
    ),
    "gpt/4o": ModelInfo(
        slug="gpt/4o", family="GPT-4o",
        upstream_provider="openrouter", upstream_id="openai/gpt-4o",
        supports_vision=True, supports_pdf=False,
    ),
    "gpt/4o-mini": ModelInfo(
        slug="gpt/4o-mini", family="GPT-4o mini",
        upstream_provider="openrouter", upstream_id="openai/gpt-4o-mini",
        supports_vision=True, supports_pdf=False,
    ),

    # ---- Gemini (routed through OpenRouter) ------------------------------
    "gemini/2.5-pro": ModelInfo(
        slug="gemini/2.5-pro", family="Gemini 2.5 Pro",
        upstream_provider="openrouter", upstream_id="google/gemini-2.5-pro",
        supports_vision=True, supports_pdf=False,
    ),
    "gemini/2.5-flash": ModelInfo(
        slug="gemini/2.5-flash", family="Gemini 2.5 Flash",
        upstream_provider="openrouter", upstream_id="google/gemini-2.5-flash",
        supports_vision=True, supports_pdf=False,
    ),
    "gemini/2.0-flash": ModelInfo(
        slug="gemini/2.0-flash", family="Gemini 2.0 Flash",
        upstream_provider="openrouter", upstream_id="google/gemini-2.0-flash-001",
        supports_vision=True, supports_pdf=False,
    ),
}


def resolve(slug: str) -> ModelInfo:
    """Return ModelInfo for a public slug, raising ValueError if unknown.

    The error message lists known slugs so callers can surface a clean
    `unknown model` to the user without leaking upstream identifiers.
    """
    info = MODELS.get(slug)
    if info is None:
        known = ", ".join(sorted(MODELS))
        raise ValueError(f"unknown model `{slug}` (known: {known})")
    return info


def is_known_slug(slug: str) -> bool:
    return slug in MODELS
