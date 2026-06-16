"""Media model registry — the subset of `api/app/model_registry.py` the local
runner needs to drive Fal directly.

Hosted media goes through the platform API, which owns the full registry
(pricing, per-model `cost_fn`s, the public price page). A local run (`puras run
--local` / `puras serve`) bills nothing — it calls Fal on the user's own key —
so all it needs from the registry is the mapping every media verb resolves to:

    public slug  ->  upstream Fal endpoint id  (+ kind, + returns_text)

This module mirrors that mapping for exactly the slugs the verb family maps in
`media_verbs.py` reference (asserted at import there). It deliberately carries NO
prices: a local run is the user's own Fal bill. Keep this table in sync with the
media rows of `api/app/model_registry.py` when a model is added or its upstream
endpoint id changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["image", "video", "audio"]


@dataclass(frozen=True)
class MediaPricing:
    """Marker only — local runs don't bill. Present so the verb layer's
    `isinstance(model.pricing, MediaPricing)` registry checks pass unchanged."""


@dataclass(frozen=True)
class MediaModel:
    slug: str
    kind: Kind
    upstream: str          # the Fal endpoint id fal_client.subscribe() is called with
    returns_text: bool = False
    pricing: MediaPricing = MediaPricing()


# slug -> (kind, upstream endpoint id, returns_text). Mirrors the media rows of
# the canonical registry; upstream ids are the exact strings the hosted media
# router passes to fal_client.subscribe.
_MEDIA: tuple[tuple[str, Kind, str, bool], ...] = (
    # ---- Image -------------------------------------------------------------
    ("openai/gpt-image-2", "image", "openai/gpt-image-2", False),
    ("openai/gpt-image-2-edit", "image", "openai/gpt-image-2/edit", False),
    ("bytedance/seedream-v4", "image", "fal-ai/bytedance/seedream/v4/text-to-image", False),
    ("bytedance/seedream-v4-edit", "image", "fal-ai/bytedance/seedream/v4/edit", False),
    ("google/imagen-4", "image", "fal-ai/imagen4/preview", False),
    ("google/nano-banana-pro", "image", "fal-ai/nano-banana-pro", False),
    ("google/nano-banana-pro-edit", "image", "fal-ai/nano-banana-pro/edit", False),
    ("kuaishou/kling-v3-image", "image", "fal-ai/kling-image/v3/text-to-image", False),
    ("kuaishou/kling-v3-image-edit", "image", "fal-ai/kling-image/v3/image-to-image", False),
    # ---- Video -------------------------------------------------------------
    ("bytedance/seedance-2-t2v", "video", "bytedance/seedance-2.0/text-to-video", False),
    ("bytedance/seedance-2-i2v", "video", "bytedance/seedance-2.0/image-to-video", False),
    ("bytedance/seedance-2-r2v", "video", "bytedance/seedance-2.0/reference-to-video", False),
    ("bytedance/seedance-2-fast-t2v", "video", "bytedance/seedance-2.0/fast/text-to-video", False),
    ("bytedance/seedance-2-fast-i2v", "video", "bytedance/seedance-2.0/fast/image-to-video", False),
    ("bytedance/seedance-2-fast-r2v", "video", "bytedance/seedance-2.0/fast/reference-to-video", False),
    ("kuaishou/kling-v3-t2v", "video", "fal-ai/kling-video/v3/pro/text-to-video", False),
    ("kuaishou/kling-v3-i2v", "video", "fal-ai/kling-video/v3/pro/image-to-video", False),
    ("kuaishou/kling-o3-r2v", "video", "fal-ai/kling-video/o3/pro/reference-to-video", False),
    ("kuaishou/kling-avatar-v2", "video", "fal-ai/kling-video/ai-avatar/v2/pro", False),
    ("google/veo-3-t2v", "video", "fal-ai/veo3", False),
    ("google/veo-3-i2v", "video", "fal-ai/veo3/image-to-video", False),
    ("google/veo-3-fast-t2v", "video", "fal-ai/veo3/fast", False),
    ("google/veo-3-fast-i2v", "video", "fal-ai/veo3/fast/image-to-video", False),
    ("google/veo-3.1-r2v", "video", "fal-ai/veo3.1/reference-to-video", False),
    # ---- Audio -------------------------------------------------------------
    ("elevenlabs/scribe-v2", "audio", "fal-ai/elevenlabs/speech-to-text/scribe-v2", True),
    ("elevenlabs/tts-v3", "audio", "fal-ai/elevenlabs/tts/eleven-v3", False),
    ("elevenlabs/tts-multilingual-v2", "audio", "fal-ai/elevenlabs/tts/multilingual-v2", False),
)

_REGISTRY: dict[str, MediaModel] = {
    slug: MediaModel(slug=slug, kind=kind, upstream=upstream, returns_text=returns_text)
    for slug, kind, upstream, returns_text in _MEDIA
}


def get(slug: str) -> MediaModel | None:
    """Look up a media model by its public slug. None when unknown."""
    return _REGISTRY.get(slug)
