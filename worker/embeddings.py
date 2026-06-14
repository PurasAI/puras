"""Optional embedding client for memory v2's semantic-retrieval branch.

Wraps any OpenAI-compatible `/v1/embeddings` endpoint (configured via
PURAS_EMBEDDINGS_API_KEY / _BASE_URL / _MODEL, falling back to OPENAI_API_KEY).
Everything here is BEST-EFFORT: when no key is configured — or a call fails —
`embed_text` returns None and the caller degrades to exact+FTS retrieval.
Memory must never break a job, so this module never raises.
"""

from __future__ import annotations

import asyncio
import os

import structlog

from .config import get_settings

log = structlog.get_logger()

# Embedding inputs are short (a memory summary or a search query); this cap
# just bounds a pathological caller. ~8k chars ≈ 2k tokens.
_TEXT_CAP = 8_000
_TIMEOUT_S = 10.0

_client = None
_client_init_failed = False


def _api_key() -> str | None:
    s = get_settings()
    return s.embeddings_api_key or os.environ.get("OPENAI_API_KEY") or None


def embeddings_enabled() -> bool:
    """True when an embedding endpoint is configured (key present)."""
    return _api_key() is not None and not _client_init_failed


def _get_client():
    global _client, _client_init_failed
    if _client is not None or _client_init_failed:
        return _client
    key = _api_key()
    if not key:
        return None
    try:
        from openai import AsyncOpenAI

        s = get_settings()
        _client = AsyncOpenAI(
            api_key=key,
            base_url=s.embeddings_base_url or None,
            timeout=_TIMEOUT_S,
            max_retries=1,
        )
    except Exception as e:  # SDK import/config problems must not break jobs
        _client_init_failed = True
        log.warning("embeddings_client_init_failed", error=str(e))
        return None
    return _client


def to_pgvector_literal(vec: list[float]) -> str:
    """Render a vector as pgvector's text input form: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.7g}" for v in vec) + "]"


async def embed_text(text: str) -> list[float] | None:
    """Embed one text. Returns None when unconfigured, on any error, or on a
    dimension mismatch with the DB column — callers treat None as 'no semantic
    branch this time' and move on."""
    cleaned = (text or "").strip()[:_TEXT_CAP]
    if not cleaned:
        return None
    client = _get_client()
    if client is None:
        return None
    s = get_settings()
    try:
        # `dimensions` makes MRL-trained models (text-embedding-3-*,
        # gemini-embedding-001) return exactly the DB column's width instead
        # of their native default (e.g. Gemini's 3072 vs our vector(1536)).
        resp = await asyncio.wait_for(
            client.embeddings.create(
                model=s.embeddings_model,
                input=cleaned,
                dimensions=s.embeddings_dims,
            ),
            timeout=_TIMEOUT_S + 2,
        )
        vec = list(resp.data[0].embedding)
    except Exception as e:
        # repr, not str: asyncio.TimeoutError stringifies to "" and an empty
        # `error=` log line is undebuggable.
        log.warning("embed_text_failed", error=repr(e))
        return None
    if len(vec) != s.embeddings_dims:
        log.warning(
            "embed_text_dim_mismatch", got=len(vec), expected=s.embeddings_dims
        )
        return None
    return vec
