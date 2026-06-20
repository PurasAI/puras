"""OpenRouter provider — OpenAI SDK with OpenRouter base URL.

OpenRouter exposes nearly every popular model (Claude, GPT, Gemini, Llama, ...)
through an OpenAI-compatible API. We translate to/from Anthropic-shaped tool_use
blocks here so the rest of the agent_runner stays provider-agnostic.

Model id format passed through as-is to OpenRouter, e.g.
    openrouter:anthropic/claude-3.5-sonnet
    openrouter:openai/gpt-4o
    openrouter:google/gemini-2.0-flash-001
"""

from __future__ import annotations

import json
from typing import Any

from ..pricing import MICROS_PER_DOLLAR
from .base import NormalizedResponse, NormalizedToolUse, Provider, ProviderError

OPENAI_FINISH_TO_ANTHROPIC = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(Provider):
    @classmethod
    def api_key_secret_name(cls) -> str:
        return "OPENROUTER_API_KEY"

    def __init__(self, model_id: str, api_key: str):
        super().__init__(model_id, api_key)
        # Lazy so importing this module (e.g. via the provider registry on the
        # offline runner, which uses Anthropic) doesn't require the openai SDK.
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    # ----------------------------------------------------------------- translate
    @staticmethod
    def _anth_to_openai_tools(tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    @staticmethod
    def _anth_to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            role = m["role"]
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            # content is a list of typed blocks
            if role == "user":
                main_blocks: list[dict] = []
                tool_msgs: list[dict] = []
                for b in content:
                    bt = b.get("type")
                    if bt == "text":
                        main_blocks.append({"type": "text", "text": b["text"]})
                    elif bt == "image":
                        main_blocks.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": _anth_image_to_data_url(b["source"])},
                            }
                        )
                    elif bt == "document":
                        raise ProviderError(
                            "document/PDF attachments are not supported by OpenRouter/OpenAI "
                            "models — switch to an Anthropic model or convert the PDF to images"
                        )
                    elif bt == "tool_result":
                        tool_msgs.append(
                            {
                                "role": "tool",
                                "tool_call_id": b["tool_use_id"],
                                "content": _tool_result_to_str(b.get("content", "")),
                            }
                        )
                if main_blocks:
                    if all(blk["type"] == "text" for blk in main_blocks):
                        out.append(
                            {
                                "role": "user",
                                "content": "\n".join(blk["text"] for blk in main_blocks),
                            }
                        )
                    else:
                        out.append({"role": "user", "content": main_blocks})
                out.extend(tool_msgs)
            elif role == "assistant":
                texts = [b["text"] for b in content if b.get("type") == "text"]
                tool_uses = [b for b in content if b.get("type") == "tool_use"]
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(texts) if texts else None,
                }
                if tool_uses:
                    msg["tool_calls"] = [
                        {
                            "id": tu["id"],
                            "type": "function",
                            "function": {
                                "name": tu["name"],
                                "arguments": json.dumps(tu.get("input", {})),
                            },
                        }
                        for tu in tool_uses
                    ]
                out.append(msg)
            else:
                out.append({"role": role, "content": _as_str(content)})
        return out

    # -------------------------------------------------------------------- call
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
        # OpenRouter doesn't expose Anthropic-style prompt caching uniformly
        # across upstreams, so we accept the flag + TTL and no-op.
        del cache_messages, cache_ttl
        try:
            # `usage={"include": True}` makes OpenRouter return upstream cost so we
            # don't have to maintain a per-model pricing table for them.
            resp = self._client.chat.completions.create(
                model=self.model_id,
                max_tokens=max_tokens,
                messages=self._anth_to_openai_messages(system, messages),
                tools=self._anth_to_openai_tools(tools),
                extra_body={"usage": {"include": True}},
            )
        except Exception as e:
            raise ProviderError(f"openrouter call failed: {e}") from e

        choice = resp.choices[0]
        msg = choice.message
        text_blocks = [msg.content] if msg.content else []
        tool_uses: list[NormalizedToolUse] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_uses.append(NormalizedToolUse(id=tc.id, name=tc.function.name, input=args))

        usage = resp.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        # OpenRouter returns `usage.cost` (USD float) when requested. Fall back
        # to 0 if missing — we'd rather under-charge than crash.
        cost_usd = getattr(usage, "cost", None) or 0.0
        try:
            cost_usd = float(cost_usd)
        except (TypeError, ValueError):
            cost_usd = 0.0
        upstream_micros = int(round(cost_usd * MICROS_PER_DOLLAR))

        return NormalizedResponse(
            stop_reason=OPENAI_FINISH_TO_ANTHROPIC.get(
                choice.finish_reason or "stop", choice.finish_reason or "end_turn"
            ),
            text_blocks=text_blocks,
            tool_uses=tool_uses,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            upstream_cost_micros=upstream_micros,
        )


def _as_str(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, default=str)


def _anth_image_to_data_url(source: dict) -> str:
    """Convert an Anthropic image source to an OpenAI image_url value.

    base64 → 'data:<media_type>;base64,<data>'
    url    → pass through
    """
    st = source.get("type")
    if st == "url":
        return source["url"]
    if st == "base64":
        return f"data:{source['media_type']};base64,{source['data']}"
    raise ProviderError(f"unsupported image source type: {st}")


def _tool_result_to_str(content: Any) -> str:
    """OpenAI tool messages only accept a string content.

    For block lists (e.g. file_read returning image blocks), we flatten:
    text blocks pass through, image/document become placeholders. This is
    lossy — non-vision models can't 'see' images surfaced via tool_result.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _as_str(content)
    parts: list[str] = []
    for b in content:
        bt = b.get("type")
        if bt == "text":
            parts.append(b.get("text", ""))
        elif bt == "image":
            mt = (b.get("source") or {}).get("media_type", "image")
            parts.append(f"[image attached ({mt}) — not visible to this model]")
        elif bt == "document":
            mt = (b.get("source") or {}).get("media_type", "document")
            parts.append(f"[document attached ({mt}) — not visible to this model]")
        else:
            parts.append(_as_str(b))
    return "\n".join(parts)
