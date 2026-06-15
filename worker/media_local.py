"""Local media generation — drive the media generate_* verbs straight against
Fal on the user's own key, with NO platform (no API, no billing, no bucket).

This is the open-core counterpart to the hosted `api/app/routers/media.py`. A
local run (`puras run --local` / `puras serve`) has no platform API to call, so
when `FAL_KEY` is set the worker resolves the verb itself (`media_verbs`), maps
the slug to a Fal endpoint (`media_registry`), prepares input media, and calls
`fal_client.subscribe` directly. It returns the SAME shape the API endpoint does
(`model / kind / drive_path / request_id / meta / output_url`, plus the inline
transcript for `transcribe`) so `agent_runner._call_media_local` can persist and
surface the result exactly like the hosted `_call_media` path.

What's intentionally dropped vs. hosted: billing/usage, idempotency replay, the
bucket round-trip, and Kling input upscaling (a local-dev convenience — pass an
already-≥300px reference if a Kling render rejects a tiny one). Input references
that are local drive files are inlined as `data:` URIs (Fal fetches those
directly), so refs work with no bucket to sign against; http(s)/data URLs pass
through untouched.
"""

from __future__ import annotations

import base64
import copy
import mimetypes
import os
import re
import uuid
from typing import Any
from urllib.parse import urlparse

from .config import get_settings
from .drive import workspace_drive
from .media_registry import get as get_model
from .media_verbs import VerbError, resolve_and_adapt


class LocalMediaError(RuntimeError):
    """A local media generation could not be set up or completed."""


# ---------------------------------------------------------------------------
# Output discovery — mirrors api/app/routers/media.py (auto-detect the primary
# media URL in Fal's response, or follow an explicit output_url_path).
# ---------------------------------------------------------------------------
_KIND_EXT = {"image": "png", "video": "mp4", "audio": "mp3"}
_EXT_TO_KIND = {
    "png": "image", "jpg": "image", "jpeg": "image", "webp": "image", "gif": "image",
    "mp4": "video", "webm": "video", "mov": "video",
    "mp3": "audio", "wav": "audio", "ogg": "audio", "m4a": "audio",
}


def _ext_for_url(url: str) -> str | None:
    path = urlparse(url).path
    m = re.search(r"\.([a-zA-Z0-9]{2,5})(?:$|\?)", path)
    return m.group(1).lower() if m else None


def _walk_json_path(obj: Any, path: str) -> Any:
    """Tiny jq-style path resolver: "video.url", "images[0].url"."""
    tokens = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    cur = obj
    for tok in tokens:
        if tok.startswith("[") and tok.endswith("]"):
            idx = int(tok[1:-1])
            if not isinstance(cur, list) or idx >= len(cur):
                raise KeyError(f"path '{path}' miss at '{tok}'")
            cur = cur[idx]
        else:
            if not isinstance(cur, dict) or tok not in cur:
                raise KeyError(f"path '{path}' miss at '{tok}'")
            cur = cur[tok]
    return cur


def _from_image_slot(result: dict) -> tuple[str | None, dict[str, Any]]:
    imgs = result.get("images")
    if isinstance(imgs, list) and imgs:
        first = imgs[0]
        if isinstance(first, dict) and isinstance(first.get("url"), str):
            return first["url"], {k: v for k, v in first.items() if k != "url"}
        if isinstance(first, str):
            return first, {}
    img = result.get("image")
    if isinstance(img, dict) and isinstance(img.get("url"), str):
        return img["url"], {k: v for k, v in img.items() if k != "url"}
    return None, {}


def _from_video_slot(result: dict) -> tuple[str | None, dict[str, Any]]:
    v = result.get("video")
    if isinstance(v, dict) and isinstance(v.get("url"), str):
        meta = {k: v[k] for k in ("duration", "size", "content_type") if k in v}
        return v["url"], meta
    if isinstance(result.get("video_url"), str):
        return result["video_url"], {}
    return None, {}


def _from_audio_slot(result: dict) -> tuple[str | None, dict[str, Any]]:
    a = result.get("audio")
    if isinstance(a, dict) and isinstance(a.get("url"), str):
        return a["url"], {k: a[k] for k in ("duration",) if k in a}
    if isinstance(result.get("audio_url"), str):
        return result["audio_url"], {}
    return None, {}


def _deep_find_url(obj: Any, depth: int = 0) -> str | None:
    if depth > 4:
        return None
    if isinstance(obj, dict):
        for k in ("url", "file_url", "output_url"):
            v = obj.get(k)
            if isinstance(v, str) and v.startswith(("http://", "https://")):
                return v
        for v in obj.values():
            r = _deep_find_url(v, depth + 1)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _deep_find_url(v, depth + 1)
            if r:
                return r
    return None


def _find_output(
    result: dict, kind: str, override_path: str | None
) -> tuple[str, str, dict[str, Any]]:
    """Returns (url, resolved_kind, meta). See the hosted router for the order."""
    if override_path:
        url = _walk_json_path(result, override_path)
        if not isinstance(url, str):
            raise LocalMediaError(
                f"output_url_path '{override_path}' did not resolve to a string"
            )
        resolved = kind if kind != "auto" else _EXT_TO_KIND.get(_ext_for_url(url) or "", "image")
        return url, resolved, {}

    if kind == "image":
        url, meta = _from_image_slot(result)
        if url:
            return url, "image", meta
    if kind == "video":
        url, meta = _from_video_slot(result)
        if url:
            return url, "video", meta
    if kind == "audio":
        url, meta = _from_audio_slot(result)
        if url:
            return url, "audio", meta

    for slot_fn, k in (
        (_from_image_slot, "image"),
        (_from_video_slot, "video"),
        (_from_audio_slot, "audio"),
    ):
        url, meta = slot_fn(result)
        if url:
            return url, k, meta
    if isinstance(result.get("url"), str):
        url = result["url"]
        return url, _EXT_TO_KIND.get(_ext_for_url(url) or "", "image"), {}
    found = _deep_find_url(result)
    if found:
        return found, _EXT_TO_KIND.get(_ext_for_url(found) or "", "image"), {}
    raise LocalMediaError(
        f"could not find output URL in Fal response: keys={list(result.keys())}. "
        f"Pass `output_url_path` to point at the URL field manually."
    )


# ---------------------------------------------------------------------------
# Input media — turn every native image/audio reference into something Fal can
# fetch. http(s)/data URLs pass through; a local drive path is inlined as a
# data: URI (no bucket to sign against offline).
# ---------------------------------------------------------------------------
_INPUT_IMG_KEYS_SINGLE = ("image_url", "tail_image_url", "end_image_url")
_INPUT_IMG_KEYS_LIST = ("image_urls",)
_INPUT_AUDIO_KEYS_SINGLE = ("audio_url",)


def _iter_media_slots(inputs: dict) -> list[tuple[str, Any]]:
    """Every input media reference in the native bag → (value, set_fn). Mirrors
    the hosted `_iter_media_slots` (single keys, `image_urls`, Kling's nested
    `elements[].reference_image_urls`)."""
    slots: list[tuple[str, Any]] = []

    def _add_single(k: str) -> None:
        v = inputs.get(k)
        if isinstance(v, str) and v.strip():
            slots.append((v, lambda nu, k=k: inputs.__setitem__(k, nu)))

    for k in (*_INPUT_IMG_KEYS_SINGLE, *_INPUT_AUDIO_KEYS_SINGLE):
        _add_single(k)
    for k in _INPUT_IMG_KEYS_LIST:
        lst = inputs.get(k)
        if isinstance(lst, list):
            for i, u in enumerate(lst):
                if isinstance(u, str) and u.strip():
                    slots.append((u, lambda nu, lst=lst, i=i: lst.__setitem__(i, nu)))
    els = inputs.get("elements")
    if isinstance(els, list):
        for el in els:
            if isinstance(el, dict) and isinstance(el.get("reference_image_urls"), list):
                lst = el["reference_image_urls"]
                for i, u in enumerate(lst):
                    if isinstance(u, str) and u.strip():
                        slots.append((u, lambda nu, lst=lst, i=i: lst.__setitem__(i, nu)))
    return slots


def _drive_file_to_data_uri(value: str, workspace_id: str) -> str:
    """Read a local drive file and return a `data:` URI Fal can fetch. Accepts
    the path however the agent saw it in its workdir ("drive/brand/x.png" or
    "brand/x.png"), matching how the worker's file tools resolve drive paths."""
    rel = value.strip().lstrip("/")
    if rel.startswith("drive/"):
        rel = rel[len("drive/"):]
    path = workspace_drive(workspace_id) / rel
    if not path.is_file():
        raise LocalMediaError(
            f"input media '{value}' was not found in the local drive ({path})."
        )
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _prepare_input_media(inputs: dict, workspace_id: str) -> dict:
    """Resolve every native input media reference to a Fal-fetchable value.

    Pass through http(s)/data URLs; inline a local drive path as a data: URI.
    Returns a copy with the references rewritten (inputs left untouched)."""
    work = copy.deepcopy(inputs)
    for value, set_url in _iter_media_slots(work):
        v = value.strip().replace("\t", "").replace("\r", "").replace("\n", "")
        if v.startswith(("http://", "https://", "data:")):
            if v != value:
                set_url(v)
            continue
        set_url(_drive_file_to_data_uri(v, workspace_id))
    return work


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def generate(
    model: str,
    inputs: dict,
    workspace_id: str,
    job_id: str | None,
    *,
    verb: str | None = None,
    output_path: str | None = None,
    output_dir: str | None = None,
    output_url_path: str | None = None,
    kind: str = "auto",
) -> dict:
    """Run one media generation against Fal directly (local, BYO key). Returns
    the same shape the hosted `/v1/media/generate` does — including the
    (temporary) Fal `output_url` for the caller to stream into the local drive,
    or, for a transcript model, the inline `text` / `words` / `language_code`.

    Raises `LocalMediaError` / `VerbError` on bad inputs; the caller maps those
    to a soft tool error the agent can react to."""
    try:
        import fal_client
    except ImportError as e:
        raise LocalMediaError(
            "media generation needs the `fal-client` package — install it "
            "(`pip install fal-client`) to use generate_* in a local run."
        ) from e

    s = get_settings()
    if not s.fal_key:
        raise LocalMediaError(
            "no FAL_KEY — set it to call Fal directly in a local run "
            "(media is BYO key offline, like your LLM key)."
        )

    # Verb mode: resolve the high-level verb + canonical inputs into a concrete
    # slug + native inputs. Raw mode (transcribe) passes the slug straight through.
    effective_kind = kind
    verb_warnings: list[str] = []
    if verb:
        resolved = resolve_and_adapt(verb, model, inputs)
        effective_slug = resolved.slug
        native_inputs = resolved.inputs
        effective_kind = resolved.kind
        verb_warnings = list(resolved.warnings)
    else:
        effective_slug = model
        native_inputs = dict(inputs or {})

    entry = get_model(effective_slug)
    if entry is None:
        raise LocalMediaError(
            f"unknown media model '{effective_slug}'. Use a family token, a "
            f"registered slug, or 'auto'."
        )

    native_inputs = _prepare_input_media(native_inputs, workspace_id)

    # fal_client reads FAL_KEY from the env; set it per-call so it isn't left in
    # the process env for skill code to read afterward.
    prev_env = os.environ.get("FAL_KEY")
    os.environ["FAL_KEY"] = s.fal_key
    try:
        completed = fal_client.subscribe(
            entry.upstream, arguments=native_inputs, with_logs=False
        )
    except Exception as e:
        raise LocalMediaError(f"media generation failed upstream: {e}") from e
    finally:
        if prev_env is None:
            os.environ.pop("FAL_KEY", None)
        else:
            os.environ["FAL_KEY"] = prev_env

    if hasattr(completed, "data"):
        result = completed.data
        upstream_metrics = getattr(completed, "metrics", {}) or {}
    elif isinstance(completed, dict):
        result = completed
        upstream_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    else:
        raise LocalMediaError(f"unexpected Fal response shape: {type(completed).__name__}")

    request_id = uuid.uuid4().hex

    # Transcript model (speech-to-text): no file — hand the text straight back.
    if entry.returns_text:
        words = result.get("words") if isinstance(result.get("words"), list) else []
        text_out = result.get("text") if isinstance(result.get("text"), str) else ""
        language_code = (
            result.get("language_code")
            if isinstance(result.get("language_code"), str)
            else None
        )
        meta: dict[str, Any] = {
            "kind": "text",
            "request_id": request_id,
            "metrics": upstream_metrics,
            "char_count": len(text_out),
        }
        if verb_warnings:
            meta["warnings"] = verb_warnings
        if language_code:
            meta["language_code"] = language_code
        return {
            "model": entry.slug,
            "kind": "text",
            "drive_path": "",
            "request_id": request_id,
            "billed_micros": 0,
            "billed_usd": 0.0,
            "meta": meta,
            "text": text_out,
            "words": words,
            "language_code": language_code,
            "output_url": None,
        }

    output_url, resolved_kind, extra_meta = _find_output(
        result, effective_kind, output_url_path
    )

    if output_path:
        clean = output_path.lstrip("/")
        if ".." in clean.split("/"):
            raise LocalMediaError("'..' not allowed in output_path")
        drive_relative = clean
    else:
        ext = _ext_for_url(output_url) or _KIND_EXT.get(resolved_kind, "bin")
        base = (output_dir or "media").strip().strip("/")
        if not base or ".." in base.split("/"):
            base = "media"
        drive_relative = f"{base}/{request_id}.{ext}"

    meta = {
        "kind": resolved_kind,
        "request_id": request_id,
        "metrics": upstream_metrics,
        **extra_meta,
    }
    if verb_warnings:
        meta["warnings"] = verb_warnings

    return {
        "model": entry.slug,
        "kind": resolved_kind,
        "drive_path": drive_relative,
        "request_id": request_id,
        "billed_micros": 0,
        "billed_usd": 0.0,
        "meta": meta,
        # The caller streams this (temporary) Fal URL into the local drive.
        "output_url": output_url,
    }
