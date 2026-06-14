"""Anthropic native provider — input is already Anthropic-shaped, near-identity."""

from __future__ import annotations

from anthropic import Anthropic

from ..config import get_settings
from ..pricing import anthropic_cost_micros
from .base import NormalizedResponse, NormalizedToolUse, Provider

# Prompt-cache breakpoints use a 1-HOUR TTL (vs Anthropic's 5-minute default) so
# a long tool gap — e.g. a video/image generation that runs minutes between LLM
# turns — doesn't expire the cached prefix and force a full (and pricey)
# re-write. Trade-off: a 1h cache WRITE costs 2× the input rate (5m = 1.25×);
# that multiplier lives in pricing.anthropic_cost_micros and must stay in sync.
_CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}

# Server-side context editing (beta). Clears old tool results once the prompt
# grows past a threshold, keeping recent ones, so a long multi-step run doesn't
# carry (and re-read) every stale tool result forever. See config.py for the
# tuning rationale and the prompt-cache trade-off.
_CONTEXT_MGMT_BETA = "context-management-2025-06-27"


def _context_management_body() -> dict | None:
    """Build the `context_management` request body from settings, or None when
    disabled. `clear_at_least` is sized so a clear only fires when it removes
    enough tokens to be worth invalidating the cached prefix once."""
    s = get_settings()
    if not s.context_editing_enabled or s.context_editing_trigger_tokens <= 0:
        return None
    edit: dict = {
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "input_tokens", "value": s.context_editing_trigger_tokens},
        "keep": {"type": "tool_uses", "value": s.context_editing_keep_tool_uses},
        "clear_tool_inputs": bool(s.context_editing_clear_tool_inputs),
    }
    if s.context_editing_clear_at_least_tokens > 0:
        edit["clear_at_least"] = {
            "type": "input_tokens",
            "value": s.context_editing_clear_at_least_tokens,
        }
    return {"edits": [edit]}


def _extract_context_management(resp) -> dict | None:
    """Best-effort read of the context-editing stats the API echoes back. The
    field is beta and not in the typed `Message` model, so fall back to the
    pydantic extras. Never raises — this is for telemetry only."""
    try:
        cm = getattr(resp, "context_management", None)
        if cm is None:
            extra = getattr(resp, "model_extra", None) or {}
            cm = extra.get("context_management")
        if cm is None:
            return None
        return cm if isinstance(cm, dict) else cm.model_dump()  # type: ignore[union-attr]
    except Exception:
        return None


class AnthropicProvider(Provider):
    @classmethod
    def api_key_secret_name(cls) -> str:
        return "ANTHROPIC_API_KEY"

    def __init__(self, model_id: str, api_key: str):
        super().__init__(model_id, api_key)
        self._client = Anthropic(api_key=api_key)

    def messages_create(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        *,
        cache_messages: bool = False,
    ) -> NormalizedResponse:
        sys_blocks = _system_with_cache(system)
        cached_tools = _tools_with_cache(tools)
        msgs = _messages_with_cache_breakpoint(messages) if cache_messages else messages
        # Context editing rides on extra_headers/extra_body so it works across
        # the whole anthropic>=0.42 pin without depending on typed beta params.
        ctx_mgmt = _context_management_body()
        extra_headers = {"anthropic-beta": _CONTEXT_MGMT_BETA} if ctx_mgmt else None
        extra_body = {"context_management": ctx_mgmt} if ctx_mgmt else None
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=sys_blocks,
            tools=cached_tools or None,
            messages=msgs,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
        text_blocks: list[str] = []
        tool_uses: list[NormalizedToolUse] = []
        for block in resp.content:
            if block.type == "text":
                text_blocks.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(
                    NormalizedToolUse(id=block.id, name=block.name, input=dict(block.input))
                )
        cache_write = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
        return NormalizedResponse(
            stop_reason=resp.stop_reason or "end_turn",
            text_blocks=text_blocks,
            tool_uses=tool_uses,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
            upstream_cost_micros=anthropic_cost_micros(
                self.model_id,
                resp.usage.input_tokens,
                resp.usage.output_tokens,
                cache_creation_input_tokens=cache_write,
                cache_read_input_tokens=cache_read,
            ),
            context_management_applied=_extract_context_management(resp),
        )


def _system_with_cache(system: str) -> list[dict] | str:
    """Wrap the system prompt in a single text block with an ephemeral
    cache_control breakpoint, so the system prefix is reused across calls.

    Returns the plain string when `system` is empty — Anthropic rejects
    empty text blocks, and there's nothing worth caching anyway.
    """
    if not system:
        return system
    return [{"type": "text", "text": system, "cache_control": {**_CACHE_CONTROL}}]


def _tools_with_cache(tools: list[dict] | None) -> list[dict] | None:
    """Add a cache_control breakpoint to the last tool definition.

    Anthropic caches the entire `tools` array up to and including the tool
    that carries `cache_control`, so placing it on the last entry caches
    every tool in one breakpoint. Safe to leave the originals untouched —
    we return a shallow-copied list.
    """
    if not tools:
        return tools
    return [*tools[:-1], {**tools[-1], "cache_control": {**_CACHE_CONTROL}}]


def _messages_with_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """Return a copy of `messages` with an ephemeral cache_control marker on
    the last content block of the last message.

    Used in agent loops: the previous turn's breakpoint becomes the
    longest-match prefix for the next call, so each iteration only pays
    write cost on its own delta. Callers' lists are never mutated.
    """
    if not messages:
        return messages
    out = list(messages)
    last = dict(out[-1])
    content = last.get("content")
    if isinstance(content, str):
        if not content:
            return out
        last["content"] = [
            {"type": "text", "text": content, "cache_control": {**_CACHE_CONTROL}}
        ]
    elif isinstance(content, list) and content:
        new_blocks = [dict(b) for b in content]
        new_blocks[-1] = {**new_blocks[-1], "cache_control": {**_CACHE_CONTROL}}
        last["content"] = new_blocks
    else:
        return out
    out[-1] = last
    return out
