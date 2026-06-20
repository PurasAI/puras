"""Provider interface + normalized response shape."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ProviderError(RuntimeError):
    pass


@dataclass
class NormalizedToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class NormalizedResponse:
    # Subset of Anthropic stop_reasons; OpenAI finish_reasons get mapped.
    # "end_turn" | "tool_use" | "max_tokens" | "stop_sequence"
    stop_reason: str
    text_blocks: list[str] = field(default_factory=list)
    tool_uses: list[NormalizedToolUse] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    # Anthropic prompt caching counters. `input_tokens` from the API already
    # excludes both — these report what was newly written to / read from the
    # prompt cache on this call. Zero for providers without prompt caching.
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    # Upstream cost the provider says (or we compute) — in MICROS (1 USD = 1e6 micros).
    # The platform decrements the project balance by (upstream + margin).
    upstream_cost_micros: int = 0
    # Server-side context-editing stats for this call (Anthropic), best-effort:
    # {"applied_edits": [...], "cleared_input_tokens": N, "cleared_tool_uses": N}.
    # None when context editing didn't run or the provider doesn't support it.
    context_management_applied: dict[str, Any] | None = None


class Provider(ABC):
    """Abstract LLM provider.

    The agent runner passes messages in **Anthropic format** (assistant content
    is a list of {type: text|tool_use} blocks; user content can be a string or
    a list of {type: text|tool_result} blocks). Implementations translate
    internally and normalize the response.
    """

    def __init__(self, model_id: str, api_key: str):
        self.model_id = model_id
        self.api_key = api_key

    @classmethod
    @abstractmethod
    def api_key_secret_name(cls) -> str:
        """Conventional env-var / project-secret name for this provider's key."""

    @abstractmethod
    def messages_create(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        *,
        cache_messages: bool = False,
        cache_ttl: str = "5m",
    ) -> NormalizedResponse:
        """Run one inference. Tools are in Anthropic schema; provider translates.

        `cache_messages`: when True, request a conversation-history cache
        breakpoint on the last message. Providers without prompt caching
        (OpenRouter today) ignore this flag.

        `cache_ttl`: Anthropic prompt-cache TTL for every cache_control
        breakpoint on this call — "5m" (default) or "1h". Ignored by providers
        without Anthropic-style prompt caching.
        """
