"""puras.drive — mint URLs for files in the workspace drive.

Deterministic skills write their outputs into the drive (e.g. via
`puras.media.generate_image`, which returns a `drive_path`). When a skill needs
a real URL — to feed a file into a URL-consuming model input, or to expose it as
the skill's result — call `puras.drive.url(drive_path)`:

    from puras import drive, media

    def run(prompt: str) -> dict:
        img = media.generate_image(prompt)
        return {"image_url": drive.url(img["drive_path"], ttl=86_400)}

The drive_path is the canonical, durable pointer; signed URLs are short-lived
and minted on demand, so nothing goes stale in a persisted result.
"""

from __future__ import annotations

from ._client import drive_sign


def sign(path: str, *, ttl: int = 3600) -> dict:
    """Sign a drive file and return the full `{path, signed_url, expires_in}`.

    `path` is workspace-relative (e.g. 'media/abc.png'); a leading
    '<workspace_id>/' is accepted and kept as-is. `ttl` is seconds until the
    URL expires (30 .. 30 days, default 1h).
    """
    return drive_sign(path, ttl=ttl)


def url(path: str, *, ttl: int = 3600) -> str:
    """Convenience wrapper around `sign` that returns just the signed URL."""
    return sign(path, ttl=ttl)["signed_url"]


__all__ = ["sign", "url"]
