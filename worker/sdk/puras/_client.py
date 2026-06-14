"""Tiny HTTP client used by puras.media to call our /v1/media/generate endpoint.

User code shouldn't import this directly — use `puras.media.generate_*`.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class PurasClientError(RuntimeError):
    pass


def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise PurasClientError(
            f"missing {name} env var — this function should be invoked through "
            f"the purasbackend worker; the worker injects it. (Running outside "
            f"the worker? You need to set it manually.)"
        )
    return v


def media_generate(
    model: str,
    inputs: dict[str, Any] | None = None,
    *,
    verb: str | None = None,
    output_path: str | None = None,
    output_url_path: str | None = None,
    kind: str = "auto",
    timeout_s: int = 600,
) -> dict[str, Any]:
    api_base = _env("PURAS_API_BASE").rstrip("/")
    token = _env("PURAS_SERVICE_TOKEN")
    workspace_id = _env("PURAS_WORKSPACE_ID")
    job_id = os.environ.get("PURAS_JOB_ID")
    body = {
        "workspace_id": workspace_id,
        "job_id": job_id,
        "verb": verb,
        "model": model,
        "inputs": inputs or {},
        "output_path": output_path,
        "output_url_path": output_url_path,
        "kind": kind,
        # We persist the output ourselves, through the worker's own drive mount,
        # so a follow-up file read is cache-consistent (an API→S3 write isn't
        # visible to the worker's s3fs mount until its cache refreshes).
        "persist": False,
    }
    r = httpx.post(
        f"{api_base}/v1/media/generate",
        headers={"X-Puras-Service-Token": token, "Content-Type": "application/json"},
        json=body,
        timeout=timeout_s,
    )
    if not r.is_success:
        raise PurasClientError(f"media/generate failed ({r.status_code}): {r.text}")
    data = r.json()
    out_url = data.pop("output_url", None)
    drive_path = data.get("drive_path")
    if out_url and drive_path:
        _persist_to_drive(out_url, drive_path)
    return data


def _persist_to_drive(url: str, drive_path: str) -> None:
    """Stream a (temporary) upstream media URL into the workspace drive.

    Skill code runs with cwd = the job workdir, where `drive/` is a symlink to
    the workspace drive — so writing under `drive/<path>` lands the bytes on the
    worker's own mount (cache-consistent reads). Streams in chunks so large
    video outputs don't have to be buffered whole."""
    rel = drive_path.lstrip("/")
    if rel.startswith("drive/"):
        rel = rel[len("drive/"):]
    dest = os.path.join("drive", *rel.split("/"))
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)


def drive_sign(path: str, *, ttl: int = 3600, timeout_s: int = 30) -> dict[str, Any]:
    """Mint a signed URL for a file already in the workspace drive.

    Backs `puras.drive.url`. Posts to the internal /v1/drive/sign endpoint with
    the worker-injected service token + workspace.
    """
    api_base = _env("PURAS_API_BASE").rstrip("/")
    token = _env("PURAS_SERVICE_TOKEN")
    workspace_id = _env("PURAS_WORKSPACE_ID")
    r = httpx.post(
        f"{api_base}/v1/drive/sign",
        headers={"X-Puras-Service-Token": token, "Content-Type": "application/json"},
        json={"workspace_id": workspace_id, "path": path, "ttl": ttl},
        timeout=timeout_s,
    )
    if not r.is_success:
        raise PurasClientError(f"drive/sign failed ({r.status_code}): {r.text}")
    return r.json()
