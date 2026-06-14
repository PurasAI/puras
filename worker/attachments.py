"""Convert drive files / URLs / base64 blobs into Anthropic content blocks.

Used by:
- `inputs.attachments` at job start — embedded in the first user message
- the `file_read` agent tool — embedded in a tool_result

Output blocks are Anthropic-shaped (`image` / `document` / `text`). The OpenRouter
provider re-translates them to OpenAI format.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from .drive import workspace_drive

IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
DOCUMENT_MIMES = {"application/pdf"}
TEXT_MIMES_EXACT = {
    "application/json",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "application/javascript",
    "application/typescript",
}

MAX_FILE_BYTES = 5 * 1024 * 1024  # Anthropic per-image hard byte limit

# Anthropic also rejects any image whose width OR height exceeds this many pixels
# (a 28000px-tall full-page screenshot is a real case) with a 400 that would
# otherwise fail the whole job. We read the dimensions cheaply from the file
# header and return a clean error instead of sending it. The producing tool still
# keeps the full-resolution file on the drive.
MAX_MODEL_IMAGE_EDGE = 8000


def _image_pixel_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) for PNG / GIF / JPEG / WEBP read straight from the header,
    or None if the format/dimensions can't be determined. Pure stdlib — no Pillow."""
    import struct

    if len(data) < 26:
        return None
    # PNG: 8-byte sig, then IHDR with width/height as big-endian uint32 at offset 16
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    # GIF: logical screen descriptor at offset 6, little-endian uint16
    if data[:6] in (b"GIF87a", b"GIF89a"):
        w, h = struct.unpack("<HH", data[6:10])
        return int(w), int(h)
    # JPEG: scan segment markers for a Start-Of-Frame (SOFn) which carries h, w
    if data[:2] == b"\xff\xd8":
        i, n = 2, len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                return int(w), int(h)
            if marker in (0x01, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg = struct.unpack(">H", data[i + 2 : i + 4])[0]
            i += 2 + seg
        return None
    # WEBP: RIFF container, then a VP8 / VP8L / VP8X chunk
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        fmt = data[12:16]
        if fmt == b"VP8X" and len(data) >= 30:
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
            return w, h
        if fmt == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
            b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
            w = 1 + (((b1 & 0x3F) << 8) | b0)
            h = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
            return w, h
        if fmt == b"VP8 " and len(data) >= 30:
            w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
            h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
            return w, h
        return None
    return None


def oversize_image_reason(data: bytes, mime: str) -> str | None:
    """A human-readable reason if this image exceeds the model's per-side pixel
    limit, else None. Non-raster (e.g. SVG) and undetermined sizes pass through."""
    if mime not in IMAGE_MIMES:
        return None
    size = _image_pixel_size(data)
    if not size:
        return None
    w, h = size
    if max(w, h) > MAX_MODEL_IMAGE_EDGE:
        return (
            f"image is {w}x{h}px but the model accepts at most "
            f"{MAX_MODEL_IMAGE_EDGE}px per side"
        )
    return None
MAX_TEXT_INLINE_CHARS = 100_000


def _resolve_drive_path(workspace_id: str, drive_path: str) -> Path:
    root = workspace_drive(workspace_id).resolve()
    p = drive_path.strip().lstrip("/")
    if p.startswith("drive/"):
        p = p[len("drive/") :]
    full = (root / p).resolve()
    # Ensure resolved path stays under the workspace's drive root.
    try:
        full.relative_to(root)
    except ValueError as e:
        raise ValueError(f"drive path escapes workspace root: {drive_path}") from e
    if not full.exists():
        # Local drive is a cache over the bucket — pull the object on a miss
        # (a file an earlier job produced, or one not declared as an input)
        # before declaring it gone.
        from .storage import ensure_local_drive_file

        if not ensure_local_drive_file(workspace_id, p):
            raise FileNotFoundError(f"drive file not found: {drive_path}")
    if not full.is_file():
        raise ValueError(f"drive path is not a file: {drive_path}")
    return full


def _guess_mime(path: str | Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


def _sniff_image_mime(data: bytes) -> str | None:
    """Detect an image's real MIME from its magic bytes, independent of the
    filename. Returns None if the bytes aren't a recognized image.

    Extensions lie — a `.webp` that actually holds PNG bytes makes Anthropic
    (and upstream image models) reject the request with a media-type mismatch.
    Trusting the bytes over the extension is the robust fix.
    """
    if len(data) < 12:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def sanitize_url(url: Any) -> Any:
    """Strip whitespace and remove embedded tab/CR/LF from a URL string.

    LLM-emitted URLs routinely arrive soft-wrapped (a `\\n` mid-URL) — httpx
    refuses those with `InvalidURL` (NOT an `httpx.HTTPError` subclass, so it
    escapes the usual transport-error handling). Browsers strip tab/newline
    anywhere in a URL per the WHATWG URL spec; do the same. Non-strings pass
    through untouched so call sites can sanitize before their own type checks.
    """
    if not isinstance(url, str):
        return url
    return url.strip().replace("\t", "").replace("\r", "").replace("\n", "")


def _is_text_mime(mime: str) -> bool:
    return mime.startswith("text/") or mime in TEXT_MIMES_EXACT


def _label(hint: str, mime: str, size_bytes: int) -> str:
    return f"{hint} ({mime}, {round(size_bytes / 1024, 1)}KB)"


def load_attachment(att: dict[str, Any], workspace_id: str) -> dict[str, Any]:
    """Resolve one attachment spec to a labeled content block.

    Returns a dict with:
        label: human-readable header (path/url + mime + size)
        block: Anthropic content block (image|document) — None for text files
        text:  decoded text content — None for binary files
        mime:  resolved MIME type
    """
    hint = att.get("drive_path") or att.get("url") or "<base64>"
    explicit_mime: str | None = att.get("media_type")

    if "drive_path" in att:
        path = _resolve_drive_path(workspace_id, att["drive_path"])
        data = path.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(
                f"attachment {att['drive_path']} exceeds {MAX_FILE_BYTES}B "
                f"limit ({len(data)}B)"
            )
        mime = explicit_mime or _guess_mime(path)
    elif "url" in att:
        url = sanitize_url(att["url"])
        url_mime = explicit_mime or _guess_mime(url)
        # Pass image/document URLs straight to Anthropic — no need to download.
        if url_mime in IMAGE_MIMES:
            return {
                "label": f"{url} ({url_mime})",
                "block": {"type": "image", "source": {"type": "url", "url": url}},
                "text": None,
                "mime": url_mime,
            }
        if url_mime in DOCUMENT_MIMES:
            return {
                "label": f"{url} ({url_mime})",
                "block": {"type": "document", "source": {"type": "url", "url": url}},
                "text": None,
                "mime": url_mime,
            }
        with httpx.Client(timeout=30, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
            data = r.content
            if len(data) > MAX_FILE_BYTES:
                raise ValueError(
                    f"attachment {url} exceeds {MAX_FILE_BYTES}B limit ({len(data)}B)"
                )
            if explicit_mime:
                mime = explicit_mime
            else:
                ct = r.headers.get("content-type", "").split(";")[0].strip()
                mime = ct or url_mime or "application/octet-stream"
    elif "base64" in att:
        try:
            data = base64.b64decode(att["base64"], validate=True)
        except Exception as e:
            raise ValueError(f"invalid base64: {e}") from e
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(
                f"attachment exceeds {MAX_FILE_BYTES}B limit ({len(data)}B)"
            )
        mime = explicit_mime or "application/octet-stream"
    else:
        raise ValueError("attachment must have drive_path, url, or base64")

    # Trust the actual bytes over a (possibly wrong) extension/Content-Type so a
    # mislabeled image — e.g. a `.webp` that's really PNG — isn't rejected by
    # Anthropic or the upstream model with a media-type mismatch.
    sniffed = _sniff_image_mime(data)
    if sniffed and sniffed != mime:
        mime = sniffed

    label = _label(hint, mime, len(data))

    # A 0-byte image/document still clears the size/oversize checks above, but
    # Anthropic rejects a block whose base64 is "" with
    # `image.source.base64: image cannot be empty` (or the document equivalent),
    # which 400s and fails the whole job. The usual cause is a generation,
    # screenshot, or download that wrote a truncated 0-byte file. Surface it as a
    # clean tool error the agent can react to (re-run the step) instead.
    if not data and (mime in IMAGE_MIMES or mime in DOCUMENT_MIMES):
        raise ValueError(
            f"attachment {hint}: file is empty (0 bytes) — nothing to attach. "
            f"Re-generate or re-fetch it, then read it again."
        )

    # Guard the model's per-side pixel limit so an oversized image returns a clean
    # error here instead of a 400 that fails the whole job downstream.
    reason = oversize_image_reason(data, mime)
    if reason:
        raise ValueError(
            f"attachment {hint}: {reason} — resize it or capture a smaller region "
            f"(for a tall page, screenshot a single viewport and pass scroll_y)"
        )

    if mime in IMAGE_MIMES:
        return {
            "label": label,
            "block": {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": base64.b64encode(data).decode("ascii"),
                },
            },
            "text": None,
            "mime": mime,
        }
    if mime in DOCUMENT_MIMES:
        return {
            "label": label,
            "block": {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": base64.b64encode(data).decode("ascii"),
                },
            },
            "text": None,
            "mime": mime,
        }
    if _is_text_mime(mime):
        text = data.decode("utf-8", errors="replace")
        if len(text) > MAX_TEXT_INLINE_CHARS:
            text = text[:MAX_TEXT_INLINE_CHARS] + "\n…[truncated]"
        return {"label": label, "block": None, "text": text, "mime": mime}

    raise ValueError(f"unsupported attachment mime type: {mime} ({hint})")


def model_supports_vision(model_slug: str) -> bool:
    """Return True if the model can receive image/document blocks.

    Reads capability from the public model registry. Unknown slugs are
    treated as non-vision (the runner already rejects unknown slugs
    upstream, so this is just a safety net).
    """
    from .llm_models import MODELS
    info = MODELS.get(model_slug)
    return bool(info and info.supports_vision)


def model_supports_pdf(model_slug: str) -> bool:
    from .llm_models import MODELS
    info = MODELS.get(model_slug)
    return bool(info and info.supports_pdf)
