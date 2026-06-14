"""PostHog product analytics (worker, server-side).

Mirror of the API's analytics module (see api/app/analytics.py). No-op unless
POSTHOG_API_KEY (the phc_ project token) is set; posthog is imported lazily so
dev/tests never depend on it. `distinct_id` is always the workspace_id, so
worker job events unify with the API + frontend events under one PostHog
person. Capture is fire-and-forget and never raises.
"""

from __future__ import annotations

from typing import Any

import structlog

from .config import get_settings

log = structlog.get_logger()

_LIB = "purasbackend-worker"

_client: Any = None
_init_done = False


def _client_or_none() -> Any:
    global _client, _init_done
    if _init_done:
        return _client
    _init_done = True
    s = get_settings()
    if not s.posthog_api_key:
        return None
    try:
        from posthog import Posthog

        _client = Posthog(
            project_api_key=s.posthog_api_key,
            host=s.posthog_host,
            flush_interval=5.0,
            max_retries=3,
        )
    except Exception:
        log.warning("posthog_init_failed", exc_info=True)
        _client = None
    return _client


def capture(
    distinct_id: str | None,
    event: str,
    properties: dict | None = None,
) -> None:
    """Emit a product event keyed to a workspace. Silently no-ops when PostHog
    is disabled or distinct_id is missing."""
    client = _client_or_none()
    if client is None or not distinct_id:
        return
    s = get_settings()
    props: dict = {
        "environment": s.environment,
        "$lib": _LIB,
        **(properties or {}),
    }
    try:
        client.capture(distinct_id=str(distinct_id), event=event, properties=props)
    except Exception:
        log.warning("posthog_capture_failed", event=event, exc_info=True)


def shutdown() -> None:
    """Flush queued events on process shutdown."""
    if _client is not None:
        try:
            _client.shutdown()
        except Exception:
            pass
