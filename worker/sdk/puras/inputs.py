"""puras.inputs — normalize agent/function inputs into bytes or local paths.

Skills don't have to care how the caller passed the file. The same `image`
input may arrive as:

    {"drive_path": "uploads/abc.jpg"}      # uploaded via /v1/drive/upload
    {"url": "https://example.com/x.jpg"}   # public URL
    {"data": "data:image/jpeg;base64,..."} # inline dataURL
    {"data": "iVBORw0KGgo..."}              # raw base64 string
    "https://..."                            # bare URL
    "data:image/png;base64,..."              # bare dataURL
    "uploads/abc.jpg"                        # bare drive path

`load_bytes(value)` returns the raw bytes regardless of which shape was used.
`load_path(value)` returns a local filesystem path — either the existing drive
symlink (for `drive_path` inputs) or a freshly downloaded temp file. Useful
when you want to hand the file straight to a library that wants a path.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import httpx

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_DATAURL_RE = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.DOTALL)


class InputError(ValueError):
    pass


def _drive_root() -> Path:
    """Per-job cwd has a `drive/` symlink to the workspace's persistent area."""
    return Path("drive")


def _resolve_drive_path(rel: str) -> Path:
    rel = rel.lstrip("/")
    if ".." in rel.split("/"):
        raise InputError("'..' segments not allowed in drive paths")
    return _drive_root() / rel


def _decode_dataurl_or_b64(s: str) -> bytes:
    m = _DATAURL_RE.match(s.strip())
    if m:
        body = m.group(3)
        if m.group(2):  # ;base64
            return base64.b64decode(body)
        # urlencoded text dataURL — rare for images, but handle it
        import urllib.parse
        return urllib.parse.unquote_to_bytes(body)
    # Bare base64 string (no data: prefix). Be lenient with whitespace.
    cleaned = "".join(s.split())
    if not _BASE64_RE.match(cleaned):
        raise InputError("string is not a valid dataURL or base64 payload")
    try:
        return base64.b64decode(cleaned, validate=True)
    except binascii.Error as e:
        raise InputError(f"base64 decode failed: {e}") from e


def _fetch_url(url: str, timeout_s: float = 60.0, max_bytes: int = 50 * 1024 * 1024) -> bytes:
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as c:
        r = c.get(url)
        r.raise_for_status()
        blob = r.content
    if len(blob) > max_bytes:
        raise InputError(f"download exceeds {max_bytes} bytes (got {len(blob)})")
    return blob


def load_bytes(value: Any) -> bytes:
    """Resolve an input value to raw bytes.

    Accepts dicts (`{drive_path|url|data: ...}`) or bare strings (URL,
    dataURL, base64, or drive path).
    """
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict):
        if "bytes" in value and isinstance(value["bytes"], (bytes, bytearray)):
            return bytes(value["bytes"])
        if "drive_path" in value:
            return _resolve_drive_path(str(value["drive_path"])).read_bytes()
        if "path" in value:  # alias
            return _resolve_drive_path(str(value["path"])).read_bytes()
        if "url" in value:
            return _fetch_url(str(value["url"]))
        if "data" in value:
            return _decode_dataurl_or_b64(str(value["data"]))
        raise InputError(
            "dict input must have one of: drive_path, url, data, bytes"
        )
    if isinstance(value, str):
        s = value.strip()
        if s.startswith(("http://", "https://")):
            return _fetch_url(s)
        if s.startswith("data:"):
            return _decode_dataurl_or_b64(s)
        # Heuristic: a drive path if it looks like a relative file path that
        # exists under drive/; otherwise treat as base64.
        candidate = _drive_root() / s.lstrip("/")
        if candidate.exists():
            return candidate.read_bytes()
        return _decode_dataurl_or_b64(s)
    raise InputError(f"unsupported input type: {type(value).__name__}")


def load_path(value: Any, *, suffix: str | None = None) -> Path:
    """Resolve an input value to a local filesystem path.

    For `drive_path` inputs we return the live symlink path so the caller can
    read it lazily. For URL / base64 / dataURL inputs we download/decode to a
    temp file and return that. The temp file is left on disk for the duration
    of the job — the workdir is cleaned up on job teardown.
    """
    if isinstance(value, dict) and "drive_path" in value:
        return _resolve_drive_path(str(value["drive_path"]))
    if isinstance(value, dict) and "path" in value and "url" not in value and "data" not in value:
        return _resolve_drive_path(str(value["path"]))
    if isinstance(value, str):
        s = value.strip()
        if not (s.startswith(("http://", "https://")) or s.startswith("data:")):
            candidate = _drive_root() / s.lstrip("/")
            if candidate.exists():
                return candidate

    blob = load_bytes(value)
    fd, tmp = tempfile.mkstemp(prefix="puras_input_", suffix=suffix or "")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return Path(tmp)


__all__ = ["load_bytes", "load_path", "InputError"]
