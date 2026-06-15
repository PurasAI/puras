"""Capability-typed media verbs → concrete model + native inputs.

NOTE: this is a vendored copy of `api/app/media_verbs.py`, repointed at the
worker's compact `media_registry`. The hosted API resolves verbs there; the
local runner (`media_local`) needs the same resolution offline (no API to call),
so the adapter lives here too. Keep the two copies in sync.

The user-facing media surface is the verbs `generate_image` / `generate_video` /
`generate_audio` (plus `transcribe` for speech-to-text, which goes through the
endpoint's raw path with a fixed model). The `media` endpoint still has an
internal raw mode (no `verb`) — used by `transcribe` and as the target the verb
adapter writes into — but there is no generic raw `media.run` user surface.

  - VERBS: `generate_image` / `generate_video` / `generate_audio` — a high-level,
    model-portable surface. The caller passes a CANONICAL input bag plus a
    `model` (a family token, a concrete slug, or "auto"); this module:

      1. infers the TASK MODE from which canonical inputs are present
         (e.g. a video call with `lipsync_audio` → lip-sync; with `refs` → r2v;
          with `image` → i2v; otherwise → t2v),
      2. resolves (verb, family, mode) → a concrete registered slug
         (a family that can't do the inferred mode is a clean 400 — NOT a
          silent wrong-model call),
      3. ADAPTS the canonical bag onto that model's idiosyncratic native input
         shape (field renames, aspect-ratio→size, duration snapped to the
         model's allowed set with a warning),

    then hands (slug, native_inputs, kind) back to the endpoint, which runs the
    existing fal call + registry billing unchanged (the adapter emits exactly
    the native keys the per-model `cost_fn`s read, so pricing stays correct).

Why verbs instead of one global `generate_video(model=...)`: media models are
NOT interchangeable the way LLMs are — each has its own input shape, task
topology, capability set, and duration rules. The verb name fixes the kind; the
present inputs fix the mode; the family map says which concrete model serves
that (family, mode). Swap = change `model`; if the new family can't do the
inferred mode you get a clear error, never a silently-wrong paid render.

`model` accepts a family token (`bytedance/seedance`, `kuaishou/kling`,
`google/veo`, bare aliases like `seedance`/`kling`/`veo` also work), a concrete
registered slug, or `"auto"` (the verb's default family for that mode).

This module owns the family→slug maps and the per-model adapters; the registry
(`model_registry.py`) owns slugs, pricing, and `cost_fn`s. Adding a model to a
family is a one-line map edit here once its registry row exists.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .media_registry import MediaPricing, get as get_model

Verb = str   # "image" | "video" | "audio"
Mode = str   # image: t2i|edit ; video: t2v|i2v|r2v|lipsync ; audio: tts


class VerbError(ValueError):
    """Caller-facing media-verb error. The router renders it as a 400."""


# ---------------------------------------------------------------------------
# Family maps: family token -> {mode: concrete registry slug}
# ---------------------------------------------------------------------------
# Keyed by the provider-prefixed family the user picks; bare aliases (last path
# segment) are accepted too via _normalize_family. Every slug here MUST exist in
# the registry — asserted at import (see _assert_slugs_exist).

VIDEO_FAMILIES: dict[str, dict[Mode, str]] = {
    "bytedance/seedance": {
        "t2v": "bytedance/seedance-2-t2v",
        "i2v": "bytedance/seedance-2-i2v",
        "r2v": "bytedance/seedance-2-r2v",
    },
    "bytedance/seedance-fast": {
        "t2v": "bytedance/seedance-2-fast-t2v",
        "i2v": "bytedance/seedance-2-fast-i2v",
        "r2v": "bytedance/seedance-2-fast-r2v",
    },
    "kuaishou/kling": {
        "t2v": "kuaishou/kling-v3-t2v",
        "i2v": "kuaishou/kling-v3-i2v",
        # r2v lives on the Kling O3 line — v3→o3 fallback, by design.
        "r2v": "kuaishou/kling-o3-r2v",
        "lipsync": "kuaishou/kling-avatar-v2",
    },
    "google/veo": {
        "t2v": "google/veo-3-t2v",
        "i2v": "google/veo-3-i2v",
        # r2v lives on the Veo 3.1 line.
        "r2v": "google/veo-3.1-r2v",
    },
    "google/veo-fast": {
        "t2v": "google/veo-3-fast-t2v",
        "i2v": "google/veo-3-fast-i2v",
    },
}
VIDEO_DEFAULT_FAMILY = "bytedance/seedance-fast"
# For "auto" when the default family can't serve the mode (e.g. lip-sync).
VIDEO_MODE_DEFAULT_FAMILY: dict[Mode, str] = {"lipsync": "kuaishou/kling"}

IMAGE_FAMILIES: dict[str, dict[Mode, str]] = {
    "google/nano-banana": {"t2i": "google/nano-banana-pro", "edit": "google/nano-banana-pro-edit"},
    "openai/gpt-image": {"t2i": "openai/gpt-image-2", "edit": "openai/gpt-image-2-edit"},
    "bytedance/seedream": {"t2i": "bytedance/seedream-v4", "edit": "bytedance/seedream-v4-edit"},
    "google/imagen": {"t2i": "google/imagen-4"},
    "kuaishou/kling-image": {
        "t2i": "kuaishou/kling-v3-image",
        "edit": "kuaishou/kling-v3-image-edit",
    },
}
IMAGE_DEFAULT_FAMILY = "google/nano-banana"

AUDIO_FAMILIES: dict[str, dict[Mode, str]] = {
    # Default to v3 (the expressive tier that reads audio tags). The
    # style/speed-knob tier (multilingual v2) stays selectable via the
    # `elevenlabs/tts-v2` family token or its concrete slug.
    "elevenlabs/tts": {"tts": "elevenlabs/tts-v3"},
    "elevenlabs/tts-v2": {"tts": "elevenlabs/tts-multilingual-v2"},
}
AUDIO_DEFAULT_FAMILY = "elevenlabs/tts"

_FAMILIES: dict[Verb, dict[str, dict[Mode, str]]] = {
    "image": IMAGE_FAMILIES,
    "video": VIDEO_FAMILIES,
    "audio": AUDIO_FAMILIES,
}
_DEFAULT_FAMILY: dict[Verb, str] = {
    "image": IMAGE_DEFAULT_FAMILY,
    "video": VIDEO_DEFAULT_FAMILY,
    "audio": AUDIO_DEFAULT_FAMILY,
}
_VERB_KIND: dict[Verb, str] = {"image": "image", "video": "video", "audio": "audio"}


def _bare_aliases(fams: dict[str, dict[Mode, str]]) -> dict[str, str]:
    """Map the last path segment (and a couple of nicknames) → full family key,
    so `seedance` resolves to `bytedance/seedance`, `veo` to `google/veo`, etc."""
    out: dict[str, str] = {}
    for full in fams:
        out[full.split("/")[-1]] = full
    return out


_ALIASES: dict[Verb, dict[str, str]] = {v: _bare_aliases(f) for v, f in _FAMILIES.items()}
# Extra nicknames.
_ALIASES["image"].update({"nano": "google/nano-banana", "banana": "google/nano-banana"})

# Reverse: concrete slug -> (family, mode), for when the caller passes a concrete
# slug as `model` so we can validate it serves the inferred mode.
_SLUG_TO_FAMILY_MODE: dict[Verb, dict[str, tuple[str, Mode]]] = {}
for _verb, _fams in _FAMILIES.items():
    _rev: dict[str, tuple[str, Mode]] = {}
    for _fam, _modes in _fams.items():
        for _mode, _slug in _modes.items():
            _rev[_slug] = (_fam, _mode)
    _SLUG_TO_FAMILY_MODE[_verb] = _rev


def _assert_slugs_exist() -> None:
    """Every slug referenced by a family map must be a registered media model.
    Catches a typo or a family pointing at a not-yet-added slug at import time."""
    missing: list[str] = []
    for fams in _FAMILIES.values():
        for modes in fams.values():
            for slug in modes.values():
                m = get_model(slug)
                if m is None or not isinstance(m.pricing, MediaPricing):
                    missing.append(slug)
    if missing:
        raise RuntimeError(
            f"media_verbs: family maps reference unknown media slugs: {sorted(set(missing))}"
        )


_assert_slugs_exist()


# ---------------------------------------------------------------------------
# Mode inference — the present canonical inputs fix the task topology.
# ---------------------------------------------------------------------------

def _has(inputs: dict, key: str) -> bool:
    v = inputs.get(key)
    if v is None:
        return False
    if isinstance(v, (list, str)) and len(v) == 0:
        return False
    return True


def infer_mode(verb: Verb, inputs: dict) -> Mode:
    if verb == "image":
        return "edit" if _has(inputs, "refs") else "t2i"
    if verb == "video":
        if _has(inputs, "lipsync_audio"):
            return "lipsync"
        if _has(inputs, "refs"):
            return "r2v"
        if _has(inputs, "image"):
            return "i2v"
        return "t2v"
    if verb == "audio":
        return "tts"
    raise VerbError(f"unknown media verb '{verb}' (use image | video | audio)")


# ---------------------------------------------------------------------------
# Resolution — (verb, model_arg, mode) -> concrete slug, or a clean error.
# ---------------------------------------------------------------------------

def _normalize_family(verb: Verb, model_arg: str) -> str | None:
    """Return the canonical family key for a model_arg, or None if it isn't a
    family (e.g. it's a concrete slug)."""
    fams = _FAMILIES[verb]
    if model_arg in fams:
        return model_arg
    alias = _ALIASES[verb].get(model_arg)
    if alias:
        return alias
    return None


def _families_supporting(verb: Verb, mode: Mode) -> list[str]:
    return [fam for fam, modes in _FAMILIES[verb].items() if mode in modes]


def resolve_slug(verb: Verb, model_arg: str | None, mode: Mode) -> str:
    arg = (model_arg or "auto").strip()

    # 1) auto — the verb's default family for this mode.
    if arg == "auto":
        fam = VIDEO_MODE_DEFAULT_FAMILY.get(mode) if verb == "video" else None
        fam = fam or _DEFAULT_FAMILY[verb]
        modes = _FAMILIES[verb][fam]
        if mode in modes:
            return modes[mode]
        # default family can't do it — pick the first family that can.
        candidates = _families_supporting(verb, mode)
        if not candidates:
            raise VerbError(f"no registered {verb} model supports mode '{mode}'")
        return _FAMILIES[verb][candidates[0]][mode]

    # 2) a family token.
    fam = _normalize_family(verb, arg)
    if fam is not None:
        modes = _FAMILIES[verb][fam]
        if mode not in modes:
            supported = ", ".join(sorted(modes)) or "(none)"
            others = ", ".join(_families_supporting(verb, mode)) or "(none)"
            raise VerbError(
                f"the '{fam}' family doesn't support {mode} "
                f"(it does: {supported}). Families that do {mode}: {others}. "
                f"Pick one of those, or pass `model: \"auto\"`."
            )
        return modes[mode]

    # 3) a concrete registered slug — must be a model of this verb's kind AND
    #    serve the inferred mode (else the caller passed the wrong inputs).
    known = _SLUG_TO_FAMILY_MODE[verb].get(arg)
    if known is not None:
        _, slug_mode = known
        if slug_mode != mode:
            raise VerbError(
                f"model '{arg}' performs the '{slug_mode}' task, but the inputs "
                f"you passed imply '{mode}'. Drop the conflicting input or pick a "
                f"'{mode}'-capable model/family."
            )
        return arg
    # Unknown to this verb's family maps. It may still be a real registry slug
    # of another kind (e.g. an STT model, which has its own `transcribe` verb).
    m = get_model(arg)
    if m is not None:
        hint = (
            " For speech-to-text, use `transcribe` instead."
            if getattr(m, "returns_text", False)
            else ""
        )
        raise VerbError(
            f"'{arg}' isn't a {verb} model usable via generate_{verb}.{hint}"
        )
    fam_list = ", ".join(sorted(_FAMILIES[verb])) or "(none)"
    raise VerbError(
        f"unknown {verb} model/family '{arg}'. Use a family ({fam_list}), a "
        f"registered slug, or 'auto'."
    )


# ---------------------------------------------------------------------------
# Caps — per-slug constraints used to coerce canonical inputs (duration snap).
# ---------------------------------------------------------------------------

# Models with a fixed duration set; a requested duration is snapped to the
# nearest allowed value (clamp at the ends) with a warning.
_ALLOWED_DURATIONS: dict[str, tuple[int, ...]] = {
    "google/veo-3-t2v": (4, 6, 8),
    "google/veo-3-i2v": (4, 6, 8),
    "google/veo-3-fast-t2v": (4, 6, 8),
    "google/veo-3-fast-i2v": (4, 6, 8),
    "google/veo-3.1-r2v": (4, 6, 8),
}
# Seedance expects `duration` as a STRING ("8"); others take an int.
_DURATION_AS_STRING = {
    s for fam in ("bytedance/seedance", "bytedance/seedance-fast")
    for s in VIDEO_FAMILIES[fam].values()
}


# fal's Kling endpoints reject input images below a hard per-side minimum
# ("image_too_small": min 300×300) — a small logo/icon passed as a reference
# would 502 the whole render. The media endpoint treats this as a platform
# concern: it upscales any undersized input image to this minimum before the
# upstream call (see routers/media.py). Keyed by provider-family prefix so new
# Kling slugs (v3 image/i2v, o3 r2v, avatar, …) are covered automatically.
_MIN_INPUT_DIM_BY_FAMILY_PREFIX: tuple[tuple[str, int], ...] = (("kuaishou/kling", 300),)


def min_input_dim(slug: str) -> int | None:
    """Per-side minimum (px) the model requires for INPUT images, or None when
    it has no such floor. Kling (image edit / i2v / r2v / tail-frame / avatar)
    needs ≥300×300; everything else returns None (no resize)."""
    if not isinstance(slug, str):
        return None
    for prefix, dim in _MIN_INPUT_DIM_BY_FAMILY_PREFIX:
        if slug.startswith(prefix):
            return dim
    return None


# Image models reject very long prompts (e.g. nano-banana-pro caps at 2500
# chars). We used to silently truncate over-cap prompts at a word boundary and
# warn — but that still RENDERS (and bills) a card built from a prompt whose
# tail (CTA / anti-frame / quality blocks) was dropped, so the result is almost
# always wrong and the caller re-renders: two paid renders for one card. Fail
# fast instead — reject the call BEFORE the upstream/billing step so nothing is
# generated or charged, and the caller shortens the prompt and resends once.
# Conservative single cap across image models; a model with a smaller real cap
# still gets a clean error mapped at the endpoint instead of an opaque 502.
_IMAGE_PROMPT_MAX_CHARS = 2500


def _check_prompt_len(prompt: str, cap: int) -> None:
    """Raise VerbError (→ 400, pre-flight, no render/charge) when an image
    prompt exceeds the cap. Do NOT truncate — a truncated prompt silently drops
    its trailing blocks and produces a wrong render the caller pays for."""
    if not isinstance(prompt, str) or len(prompt) <= cap:
        return
    raise VerbError(
        f"prompt is {len(prompt)} chars but this model accepts at most {cap}. "
        f"Nothing was generated or billed. Shorten the prompt below {cap} "
        f"(aim for under ~1900) and resend — do not retry the same prompt, and "
        f"do not let the platform truncate it: cut scene/atmosphere prose, keep "
        f"every required element (logo, headline, CTA)."
    )


def _snap_duration(slug: str, duration: int) -> tuple[int, str | None]:
    allowed = _ALLOWED_DURATIONS.get(slug)
    if not allowed or duration in allowed:
        return duration, None
    lo, hi = allowed[0], allowed[-1]
    if duration < lo:
        snapped = lo
    elif duration > hi:
        snapped = hi
    else:
        snapped = min(allowed, key=lambda a: abs(a - duration))
    warn = (
        f"{slug} only renders {'/'.join(map(str, allowed))}s — requested "
        f"{duration}s snapped to {snapped}s."
    )
    return snapped, warn


# ---------------------------------------------------------------------------
# Adapters — canonical bag -> the model's native input shape.
# ---------------------------------------------------------------------------

@dataclass
class Resolved:
    slug: str
    inputs: dict[str, Any]
    kind: str
    mode: Mode
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# In-prompt reference tags — one portable author convention, normalized per model.
# ---------------------------------------------------------------------------
# Authors bind a description to a specific reference by writing `@Image1`,
# `@Image2`, … (1-indexed, in the SAME order as the `refs` list). Only some models
# read those tags natively — Seedance reference-to-video and Kling O3
# reference-to-video address `image_urls` as `@ImageN`. Every other model (Veo's
# "ingredients", and ALL image models — Nano Banana / Seedream / GPT-Image /
# Imagen / Kling-Image) wants the reference described in prose instead. Since the
# platform passes the prompt to the model UNCHANGED, a stray `@Image1` would ship
# literally and confuse a non-tag model — so here we keep the tags for the models
# that read them and rewrite them to prose ("the first reference image") for the
# rest. `@ElementN` (Kling's named-element syntax) needs a separate
# `kling_elements` payload we do NOT send, so it's rewritten to prose everywhere
# with a warning — use `@ImageN` instead.
_REF_IMAGE_RE = re.compile(r"@image\s*0*(\d+)", re.IGNORECASE)
_REF_ELEMENT_RE = re.compile(r"@element\s*0*(\d+)", re.IGNORECASE)
_REF_ORDINALS = (
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
)


def _ref_tags_native(slug: str) -> bool:
    """True for the models that read `@ImageN` reference tags in the prompt:
    Seedance (reference-to-video) and Kling O3 (reference-to-video)."""
    return slug.startswith(("bytedance/seedance", "seedance")) or "kling-o3" in slug


def _ref_phrase(i: int) -> str:
    if 1 <= i <= len(_REF_ORDINALS):
        return f"the {_REF_ORDINALS[i - 1]} reference image"
    return f"reference image {i}"


def _normalize_ref_tags(prompt: str, slug: str) -> tuple[str, list[str]]:
    """Make the `@ImageN` author convention portable across models. Returns the
    (possibly rewritten) prompt + any warnings. No-op when the prompt has no tags."""
    if not prompt or "@" not in prompt:
        return prompt, []
    warns: list[str] = []
    out = prompt
    # `@ElementN` is never wired (no kling_elements payload) → always prose.
    if _REF_ELEMENT_RE.search(out):
        out = _REF_ELEMENT_RE.sub(lambda m: _ref_phrase(int(m.group(1))), out)
        warns.append(
            "named `@ElementN` references aren't supported — write `@ImageN` (the "
            "Nth image in `refs`) instead. Rewrote them as prose for this render."
        )
    # `@ImageN`: kept for the models that read it, prose for everyone else.
    if not _ref_tags_native(slug) and _REF_IMAGE_RE.search(out):
        out = _REF_IMAGE_RE.sub(lambda m: _ref_phrase(int(m.group(1))), out)
        warns.append(
            f"{slug} doesn't read `@ImageN` reference tags — rewrote them as prose "
            "(\"the first reference image\", …); the references still apply in order."
        )
    return out, warns


_GPT_IMAGE_SLUGS = {"openai/gpt-image-2", "openai/gpt-image-2-edit"}
_NANO_SLUGS = {"google/nano-banana-pro", "google/nano-banana-pro-edit"}
# Kling's image-to-image endpoint takes a SINGLE base `image_url` (+ optional
# `elements`), unlike the `image_urls` list every other edit model accepts.
_KLING_IMAGE_EDIT_SLUGS = {"kuaishou/kling-v3-image-edit"}

# GPT Image 2 (fal) takes fal's `image_size` — a preset name or explicit
# {width, height} — NOT OpenAI's `size`: an unknown key is silently ignored and
# every render comes back at the endpoint default (landscape_4_3, 1024×768)
# regardless of the requested ratio. Explicit dims so 4:5 and true 9:16 exist:
# multiples of 16, total pixels 655,360–8,294,400, max edge 3840.
_GPT_IMAGE_SIZE: dict[str, dict[str, tuple[int, int]]] = {
    "1:1":  {"1K": (1024, 1024), "2K": (2048, 2048), "4K": (2880, 2880)},
    "4:3":  {"1K": (1024, 768),  "2K": (2048, 1536), "4K": (3264, 2448)},
    "3:4":  {"1K": (768, 1024),  "2K": (1536, 2048), "4K": (2448, 3264)},
    "4:5":  {"1K": (1024, 1280), "2K": (2048, 2560), "4K": (2560, 3200)},
    "5:4":  {"1K": (1280, 1024), "2K": (2560, 2048), "4K": (3200, 2560)},
    "16:9": {"1K": (1280, 720),  "2K": (2560, 1440), "4K": (3840, 2160)},
    "9:16": {"1K": (720, 1280),  "2K": (1440, 2560), "4K": (2160, 3840)},
}

_SEEDREAM_SLUGS = {"bytedance/seedream-v4", "bytedance/seedream-v4-edit"}

# Seedream v4 (fal) has the same trap: NO `aspect_ratio` input — it takes fal's
# `image_size` ({width, height} or preset; default 2048×2048), so a passed-
# through aspect was silently ignored and every render came back square.
# Constraint per the endpoint schema: total pixels between 960×960 and
# 4096×4096.
_SEEDREAM_SIZE: dict[str, dict[str, tuple[int, int]]] = {
    "1:1":  {"1K": (1024, 1024), "2K": (2048, 2048), "4K": (4096, 4096)},
    "4:3":  {"1K": (1152, 864),  "2K": (2048, 1536), "4K": (4096, 3072)},
    "3:4":  {"1K": (864, 1152),  "2K": (1536, 2048), "4K": (3072, 4096)},
    "4:5":  {"1K": (1024, 1280), "2K": (2048, 2560), "4K": (3072, 3840)},
    "5:4":  {"1K": (1280, 1024), "2K": (2560, 2048), "4K": (3840, 3072)},
    "16:9": {"1K": (1536, 864),  "2K": (2560, 1440), "4K": (3840, 2160)},
    "9:16": {"1K": (864, 1536),  "2K": (1440, 2560), "4K": (2160, 3840)},
}

# Models that take a literal `aspect_ratio` ENUM — fal validates it, so an
# unsupported ratio (e.g. 4:5 on imagen/kling) fails the whole render with a
# 422 instead of being ignored. Coerce to the nearest supported ratio and warn.
# Both families cap resolution at 2K.
_ASPECT_ENUM_SLUGS: dict[str, set[str]] = {
    "google/imagen-4": {"1:1", "16:9", "9:16", "4:3", "3:4"},
    "kuaishou/kling-v3-image": {"16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"},
    "kuaishou/kling-v3-image-edit": {"16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"},
}


def _nearest_aspect(aspect: str, allowed: set[str]) -> str | None:
    """The allowed ratio closest in shape to `aspect` (log-scale, so 4:5→3:4
    beats 4:5→1:1). None if `aspect` isn't parseable as `W:H`."""
    def _ratio(a: str) -> float:
        w, h = a.split(":")
        return float(w) / float(h)

    try:
        target = _ratio(aspect)
    except (ValueError, ZeroDivisionError):
        return None
    return min(allowed, key=lambda a: abs(math.log(_ratio(a) / target)))


def _adapt_image(slug: str, mode: Mode, c: dict) -> tuple[dict, list[str]]:
    prompt = c.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise VerbError("generate_image requires a non-empty `prompt`.")
    prompt, warnings = _normalize_ref_tags(prompt, slug)
    _check_prompt_len(prompt, _IMAGE_PROMPT_MAX_CHARS)
    native: dict[str, Any] = {"prompt": prompt}
    n = c.get("n")
    aspect = c.get("aspect_ratio")
    resolution = (c.get("resolution") or "1K") if c.get("resolution") else None

    if mode == "edit":
        refs = c.get("refs")
        if not isinstance(refs, list) or not refs:
            raise VerbError("image edit mode needs `refs` (a list of image URLs).")
        if slug in _KLING_IMAGE_EDIT_SLUGS:
            # Kling i2i edits a single base `image_url`; any further refs ride
            # along as one `element` whose `reference_image_urls` carries them
            # (the endpoint accepts up to 3, and rejects an element shaped any
            # other way — `frontal_image_url` / empty lists fail to resolve).
            native["image_url"] = refs[0]
            extra = [r for r in refs[1:] if isinstance(r, str) and r]
            if extra:
                native["elements"] = [{"reference_image_urls": extra[:3]}]
            if len(extra) > 3:
                warnings.append(
                    f"{slug} composes a base image + up to 3 extra references; "
                    f"used the first 4 of {len(refs)} refs."
                )
        else:
            native["image_urls"] = refs

    if slug in _GPT_IMAGE_SLUGS:
        if aspect:
            dims = _GPT_IMAGE_SIZE.get(aspect, {}).get(resolution or "1K")
            if dims:
                native["image_size"] = {"width": dims[0], "height": dims[1]}
            else:
                native["image_size"] = "auto"
                warnings.append(
                    f"aspect_ratio {aspect!r} has no GPT Image size mapping — "
                    "sent image_size='auto' (the model picks)."
                )
        native.setdefault("quality", "high")
        if n:
            native["num_images"] = int(n)
    elif slug in _NANO_SLUGS:
        if aspect:
            native["aspect_ratio"] = aspect
        if resolution:
            native["resolution"] = resolution
        if n:
            native["num_images"] = int(n)
    elif slug in _SEEDREAM_SLUGS:
        if aspect:
            dims = _SEEDREAM_SIZE.get(aspect, {}).get(resolution or "1K")
            if dims:
                native["image_size"] = {"width": dims[0], "height": dims[1]}
            else:
                native["image_size"] = "auto"
                warnings.append(
                    f"aspect_ratio {aspect!r} has no Seedream size mapping — "
                    "sent image_size='auto' (the model picks)."
                )
        if n:
            native["num_images"] = int(n)
    elif slug in _ASPECT_ENUM_SLUGS:
        allowed = _ASPECT_ENUM_SLUGS[slug]
        if aspect:
            if aspect in allowed:
                native["aspect_ratio"] = aspect
            else:
                nearest = _nearest_aspect(aspect, allowed)
                if nearest:
                    native["aspect_ratio"] = nearest
                    warnings.append(
                        f"{slug} doesn't support aspect_ratio {aspect!r} — "
                        f"rendered the nearest supported ratio {nearest!r} instead."
                    )
        if resolution:
            if resolution == "4K":
                resolution = "2K"
                warnings.append(f"{slug} caps at 2K — rendered 2K instead of 4K.")
            native["resolution"] = resolution
        if n:
            native["num_images"] = int(n)
    else:
        # Any future image model: pass aspect through if given; takes a plain
        # prompt (+ image_urls for edit).
        if aspect:
            native["aspect_ratio"] = aspect
        if n:
            native["num_images"] = int(n)
    return native, warnings


# Tail-keyframe (`last_frame`) native input name, by provider. fal exposes an
# end-frame on Seedance (`end_image_url`) and Kling (`tail_image_url`) only; Veo
# and anything else has none, so we surface a warning instead of a silent no-op.
def _last_frame_native_key(slug: str) -> str | None:
    if slug.startswith(("bytedance/seedance", "seedance")):
        return "end_image_url"
    if slug.startswith(("kuaishou/kling", "kling")):
        return "tail_image_url"
    return None


def _adapt_video(slug: str, mode: Mode, c: dict) -> tuple[dict, list[str]]:
    warnings: list[str] = []

    if mode == "lipsync":
        image = c.get("image")
        audio = c.get("lipsync_audio")
        if not image:
            raise VerbError("lip-sync needs a portrait `image`.")
        if not audio:
            raise VerbError("lip-sync needs `lipsync_audio` (a narration audio URL).")
        native: dict[str, Any] = {"image_url": image, "audio_url": audio}
        if c.get("prompt"):
            native["prompt"] = c["prompt"]   # short delivery note, not a script
        # No duration / aspect_ratio: output length follows the audio, framing
        # follows the portrait. Any duration/aspect_ratio passed is ignored.
        for ignored in ("duration", "aspect_ratio"):
            if c.get(ignored) is not None:
                warnings.append(f"lip-sync ignores `{ignored}` (it follows the audio/portrait).")
        return native, warnings

    prompt = c.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise VerbError(f"generate_video ({mode}) requires a non-empty `prompt`.")
    prompt, tag_warnings = _normalize_ref_tags(prompt, slug)
    warnings.extend(tag_warnings)
    native = {"prompt": prompt}

    if mode == "i2v":
        if not c.get("image"):
            raise VerbError("i2v needs an `image` URL.")
        native["image_url"] = c["image"]
    elif mode == "r2v":
        refs = c.get("refs")
        if not isinstance(refs, list) or not refs:
            raise VerbError("r2v needs `refs` (a list of reference image URLs).")
        native["image_urls"] = refs

    if c.get("aspect_ratio"):
        native["aspect_ratio"] = c["aspect_ratio"]
    if c.get("resolution"):
        native["resolution"] = c["resolution"]
    # Optional tail keyframe: end the clip on a specific still (e.g. an end card,
    # or a gameplay first frame so the clip flows into the real footage). The
    # native key differs by provider; only Seedance and Kling expose one, so for
    # other families we drop it with a warning rather than send a key fal rejects.
    last_frame = c.get("last_frame")
    if last_frame:
        tail_key = _last_frame_native_key(slug)
        if tail_key:
            native[tail_key] = last_frame
        else:
            warnings.append(
                f"{slug} has no last-frame keyframe input — `last_frame` was "
                f"ignored. (Seedance / Kling support it; for others, append the "
                f"frame deterministically instead.)"
            )
    # Native audio: every video family reads `generate_audio`.
    native["generate_audio"] = bool(c.get("audio", False))

    duration = c.get("duration")
    if duration is not None:
        try:
            d = int(duration)
        except (TypeError, ValueError):
            raise VerbError("`duration` must be a number of seconds.") from None
        d, warn = _snap_duration(slug, d)
        if warn:
            warnings.append(warn)
        native["duration"] = str(d) if slug in _DURATION_AS_STRING else d

    return native, warnings


# v3 reads audio tags but tunes delivery through tags + `stability` only; the
# `style`/`speed`/`similarity_boost` knobs are a multilingual-v2 feature. fal's
# v3 endpoint silently ignores them, so we drop them for v3 and warn — the
# caller learns they had no effect and can switch tiers if they need them.
_V3_AUDIO_SLUG = "elevenlabs/tts-v3"
_V2_ONLY_AUDIO_KNOBS = ("similarity_boost", "style", "speed")


def _adapt_audio(slug: str, mode: Mode, c: dict) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    text = c.get("text")
    if not isinstance(text, str) or not text.strip():
        raise VerbError("generate_audio requires non-empty `text`.")
    native: dict[str, Any] = {"text": text}
    if c.get("voice"):
        native["voice"] = c["voice"]
    if c.get("language"):
        native["language_code"] = c["language"]
    if c.get("stability") is not None:
        native["stability"] = c["stability"]

    v2_only = [k for k in _V2_ONLY_AUDIO_KNOBS if c.get(k) is not None]
    if slug == _V3_AUDIO_SLUG:
        if v2_only:
            knobs = ", ".join(f"`{k}`" for k in v2_only)
            warnings.append(
                f"ElevenLabs v3 ignores {knobs} — shape v3 delivery with audio "
                "tags ([excited]/[whispers]/…) plus `stability`, or pick "
                "`elevenlabs/tts-multilingual-v2` for those knobs."
            )
    else:
        for k in v2_only:
            native[k] = c[k]
    return native, warnings


# ---------------------------------------------------------------------------
# Public entry point — called by the media endpoint when `verb` is set.
# ---------------------------------------------------------------------------

DEFAULT_FAMILY_BY_KIND: dict[Verb, str] = dict(_DEFAULT_FAMILY)  # image/video/audio → default family


def family_catalog() -> dict[Verb, list[tuple[str, str]]]:
    """Per verb-kind: `[(family_token, representative_slug)]` for the model
    picker. The representative slug is the family's default-mode model (used to
    surface a price). Order matches the family-map definition order."""
    rep_modes: dict[Verb, tuple[Mode, ...]] = {
        "image": ("t2i", "edit"),
        "video": ("i2v", "t2v", "r2v", "lipsync"),
        "audio": ("tts",),
    }
    out: dict[Verb, list[tuple[str, str]]] = {}
    for verb, fams in _FAMILIES.items():
        rows: list[tuple[str, str]] = []
        for fam, modes in fams.items():
            slug = next(
                (modes[m] for m in rep_modes.get(verb, ()) if m in modes),
                next(iter(modes.values())),
            )
            rows.append((fam, slug))
        out[verb] = rows
    return out


def is_valid_media_model(verb: Verb, token: str) -> bool:
    """True if `token` is a usable model for generate_<verb>: a family token
    (incl. a bare alias), a concrete registered slug in a family map, `"auto"`,
    or any registered media slug whose kind matches the verb. Used by the
    manifest to validate skill.yaml's `image_model` / `video_model` /
    `audio_model` defaults at deploy time."""
    if verb not in _FAMILIES or not isinstance(token, str) or not token:
        return False
    if token == "auto":
        return True
    if _normalize_family(verb, token) is not None:
        return True
    if token in _SLUG_TO_FAMILY_MODE[verb]:
        return True
    m = get_model(token)
    return m is not None and getattr(m, "kind", None) == _VERB_KIND.get(verb)


def resolve_and_adapt(verb: Verb, model_arg: str | None, inputs: dict | None) -> Resolved:
    """Resolve a verb call to (concrete slug, native inputs, kind, warnings).

    Raises VerbError (→ 400) on an unsupported family/mode, a wrong-kind slug,
    or missing required canonical inputs.
    """
    if verb not in _FAMILIES:
        raise VerbError(f"unknown media verb '{verb}' (use image | video | audio).")
    c = dict(inputs or {})
    mode = infer_mode(verb, c)
    slug = resolve_slug(verb, model_arg, mode)

    warnings: list[str] = []
    if verb == "image":
        native, warnings = _adapt_image(slug, mode, c)
    elif verb == "video":
        native, warnings = _adapt_video(slug, mode, c)
    else:
        native, warnings = _adapt_audio(slug, mode, c)

    return Resolved(slug=slug, inputs=native, kind=_VERB_KIND[verb], mode=mode, warnings=warnings)
