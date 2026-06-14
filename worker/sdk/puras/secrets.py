"""puras.secrets — read skillpack secrets that the worker injected as env vars."""

from __future__ import annotations

import os


class SecretError(RuntimeError):
    pass


def secret(name: str, default: str | None = None) -> str:
    """Read a skillpack secret by NAME. Skillpack secrets are injected as env
    vars into your function subprocess at run time (see Secrets tab in the
    dashboard). They travel with the skillpack code — when another workspace
    runs your public skillpack, your secrets are still what the skill sees.

    Raises SecretError if missing and no default is provided.
    """
    v = os.environ.get(name, default)
    if v is None:
        raise SecretError(
            f"missing secret `{name}` — set it in the skillpack's Secrets tab "
            f"or pass a default to puras.secret(name, default=...)"
        )
    return v
