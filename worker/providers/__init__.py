"""LLM provider abstraction.

The agent runner builds messages and tools in **Anthropic format** (the most
expressive of the bunch — tool_use blocks and tool_result blocks live inside
content lists). A Provider takes that input and returns a NormalizedResponse.
Provider implementations translate to/from their native wire format internally.

Provider key resolution priority:
1. Skillpack secret named `<PROVIDER>_API_KEY` (e.g. ANTHROPIC_API_KEY, OPENROUTER_API_KEY)
2. Worker process env var with the same name (dev convenience)

Add a provider by:
- creating worker/worker/providers/<name>_provider.py with a Provider subclass
- adding it to PROVIDERS in this file + SUPPORTED_PROVIDERS in manifest.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .anthropic_provider import AnthropicProvider
from .base import NormalizedResponse, NormalizedToolUse, Provider, ProviderError
from .openrouter_provider import OpenRouterProvider

PROVIDERS = {
    "anthropic": AnthropicProvider,
    "openrouter": OpenRouterProvider,
}


def make_provider(provider_name: str, model_id: str) -> Provider:
    """LLM provider keys are PLATFORM-owned (worker env). Users never bring
    their own LLM keys — they pay us via credit balance, we pay upstream.
    Skillpack secrets are still used for the skillpack's OWN third-party
    services (their app DB, FAL_KEY if they're DIY-ing media, etc.)."""
    cls = PROVIDERS.get(provider_name)
    if cls is None:
        raise ProviderError(
            f"unknown provider `{provider_name}` (supported: {sorted(PROVIDERS)})"
        )
    key_name = cls.api_key_secret_name()
    api_key = os.environ.get(key_name)
    if not api_key:
        raise ProviderError(
            f"platform misconfigured: missing worker env var {key_name}"
        )
    return cls(model_id=model_id, api_key=api_key)


__all__ = [
    "NormalizedResponse",
    "NormalizedToolUse",
    "Provider",
    "ProviderError",
    "make_provider",
]
