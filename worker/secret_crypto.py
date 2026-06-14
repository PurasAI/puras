"""At-rest encryption for skillpack secrets (P1-5).

Secret VALUES were stored plaintext (RLS-protected, worker reads via service
role). This encrypts them at rest when `SECRETS_ENC_KEY` is set: the API encrypts
on write, the worker decrypts on read. The SAME helper lives in the worker
(api/app/secret_crypto.py) so both sides agree on the format.

Backward compatible by design — a stored value without the `enc:v1:` prefix is
legacy plaintext and returned as-is, so existing rows keep working and the key
can be rolled out with NO migration / backfill (new writes encrypt; old reads
pass through). Key rotation: `SECRETS_ENC_KEY` may be a comma-separated list —
the first key encrypts, every key is tried for decryption (MultiFernet).

Generate a key:  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import os
from functools import lru_cache

_PREFIX = "enc:v1:"


@lru_cache(maxsize=1)
def _cipher():
    raw = os.environ.get("SECRETS_ENC_KEY", "").strip()
    if not raw:
        return None
    from cryptography.fernet import Fernet, MultiFernet

    keys = [Fernet(k.strip().encode()) for k in raw.split(",") if k.strip()]
    return MultiFernet(keys) if keys else None


def encryption_enabled() -> bool:
    return _cipher() is not None


def encrypt(value: str) -> str:
    """Ciphertext (prefixed) when a key is configured; plaintext otherwise so dev
    / local runs without a key keep working."""
    c = _cipher()
    if c is None:
        return value
    return _PREFIX + c.encrypt(value.encode()).decode()


def decrypt(stored: str) -> str:
    """Inverse of `encrypt`. A non-prefixed value is legacy plaintext, returned
    as-is. A prefixed value with no key configured is a misconfiguration."""
    if not isinstance(stored, str) or not stored.startswith(_PREFIX):
        return stored
    c = _cipher()
    if c is None:
        raise RuntimeError(
            "SECRETS_ENC_KEY is not set but a secret is encrypted at rest"
        )
    return c.decrypt(stored[len(_PREFIX):].encode()).decode()


def is_encrypted(stored: str) -> bool:
    return isinstance(stored, str) and stored.startswith(_PREFIX)
