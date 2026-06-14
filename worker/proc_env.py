"""Allowlisted environment for skill subprocesses (P1-5 isolation).

Skill code — the `bash` tool and deterministic Python functions — used to inherit
the worker's ENTIRE `os.environ`, which holds the PLATFORM's own secrets:
DATABASE_URL, the Supabase service-role key, ANTHROPIC_API_KEY,
PURAS_SERVICE_TOKEN, Sentry/PostHog tokens, S3 keys, the JWT secret. A malicious
or merely buggy skill could `echo $DATABASE_URL` (the report's over-scoped-
credential incident, in miniature) and exfiltrate them across the tenant
boundary. Full per-session microVM isolation is a deploy-architecture choice;
this closes the most acute leak at the code level: build the subprocess env from
an ALLOWLIST of safe system/runtime vars, so platform secrets never cross into
untrusted skill code.

The skillpack's OWN secrets and the few `PURAS_*` the bundled SDK needs are
layered ON TOP by the callers (function_runner / agent_runner) — this is only the
base. `extra_allow` (SKILL_ENV_PASSTHROUGH) is a migration escape hatch for a
deployment that legitimately relied on some inherited var.
"""

from __future__ import annotations

import os

# Safe, non-secret system/runtime vars a subprocess (and its venv, ffmpeg,
# Pillow, httpx/TLS) needs. EVERYTHING NOT LISTED — every platform secret — is
# dropped. Keep this an allowlist (a denylist silently leaks the next secret
# someone adds to the worker env).
_ALLOW: frozenset[str] = frozenset(
    {
        # Shell / process basics
        "PATH", "HOME", "USER", "LOGNAME", "SHELL", "PWD", "TERM", "HOSTNAME",
        # Locale / time
        "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "LC_NUMERIC", "TZ",
        # Temp dirs
        "TMPDIR", "TEMP", "TMP",
        # TLS + proxy so outbound HTTPS from skill code still works
        "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "no_proxy",
        # Fonts for media/image tooling (Pillow/ffmpeg drawtext)
        "FONTCONFIG_PATH", "FONTCONFIG_FILE", "XDG_CACHE_HOME", "XDG_DATA_HOME",
    }
)


def safe_base_env(extra_allow: list[str] | None = None) -> dict[str, str]:
    """The worker `os.environ` filtered to non-secret system/runtime vars. Callers
    layer the skillpack's own secrets + the needed PURAS_* on top of this."""
    allow = set(_ALLOW)
    if extra_allow:
        allow |= {n.strip() for n in extra_allow if n and n.strip()}
    return {k: v for k, v in os.environ.items() if k in allow}
