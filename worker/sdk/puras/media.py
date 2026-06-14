"""puras.media — create media (image / video / audio) and transcribe speech,
billed to your workspace.

Credentials are managed by the platform. Each call is debited from your
workspace's credit balance at the registered rate. Generated files are saved
into your workspace drive at the returned `drive_path` (a signed URL is in the
response too); `transcribe` returns text inline (no file).

High-level, model-portable surface — you pick WHAT to make and a `model`
family, not an upstream slug:

    from puras import media

    img  = media.generate_image("a cat in a hat", aspect_ratio="1:1")
    clip = media.generate_video(
        "make it spin slowly", image=portrait_url, duration=8, audio=True,
        model="bytedance/seedance",
    )
    vo   = media.generate_audio("Hello world", voice="Rachel", language="en")
    text = media.transcribe(audio_url, keyterms=["Puras"])

`model` is a family token (`bytedance/seedance`, `kuaishou/kling`,
`google/veo`, `google/nano-banana`, `openai/gpt-image`, `elevenlabs/tts`, …)
or `"auto"`. The inputs you pass fix the task: a video call with
`lipsync_audio` lip-syncs, with `refs` does reference-to-video, with `image`
animates a still, otherwise text-to-video. If the chosen family can't do the
inferred task you get a clear error — never a silently-wrong render.

Each generate_* call returns
`{model, kind, drive_path, request_id, billed_micros, billed_usd, meta}`. Turn
`drive_path` into a URL with `puras.drive.url(drive_path)` when you need one.
"""

from __future__ import annotations

from typing import Any

from ._client import media_generate


def _compact(d: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is None so we only send inputs the caller set."""
    return {k: v for k, v in d.items() if v is not None}


def generate_image(
    prompt: str,
    *,
    model: str = "auto",
    refs: list[str] | None = None,
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    n: int | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate an image from a prompt — model-portable.

    Pass `refs` (image URLs) to run an edit/compose instead of text-to-image.
    `model` is a family (`google/nano-banana`, `openai/gpt-image`,
    `bytedance/seedream`, `google/imagen`, …) or `"auto"`; the platform picks
    the concrete model and adapts the inputs. Returns the same shape as `run`.

    To bind part of the prompt to a specific reference, write `@Image1`,
    `@Image2`, … (1-indexed, in `refs` order). The platform normalizes them per
    model — kept verbatim for models that read them, rewritten to prose ("the
    first reference image") for the rest — so the same prompt is portable.

    Args:
      prompt: what to draw / how to edit. Use `@Image1`/`@Image2` to point at a
        specific `refs` entry.
      refs: optional reference image URLs → edit mode (must be URLs, not drive
        paths — sign drive paths with `puras.drive.url` first). Address them in
        the prompt as `@Image1`, `@Image2`, … in this order.
      aspect_ratio: e.g. "1:1", "16:9", "9:16".
      resolution: "1K" | "2K" | "4K" (honored where the model supports it).
      n: number of images.
    """
    inputs = _compact(
        {"prompt": prompt, "refs": refs, "aspect_ratio": aspect_ratio,
         "resolution": resolution, "n": n}
    )
    return media_generate(model=model, inputs=inputs, verb="image", output_path=output_path)


def generate_video(
    prompt: str = "",
    *,
    model: str = "auto",
    image: str | None = None,
    last_frame: str | None = None,
    refs: list[str] | None = None,
    lipsync_audio: str | None = None,
    duration: int | None = None,
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    audio: bool = False,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate a video — model-portable. The inputs you pass fix the mode:

      - `lipsync_audio` (+ `image`) → lip-synced talking head (avatar);
      - `refs`  → reference-to-video (keep referenced subjects on-model);
      - `image` → image-to-video (animate a still);
      - neither → text-to-video.

    `model` is a family (`bytedance/seedance`, `kuaishou/kling`, `google/veo`,
    `bytedance/seedance-fast`, `google/veo-fast`) or `"auto"`. If the chosen
    family can't do the inferred mode you get a clear error (e.g. only Kling
    does lip-sync). Reference/image inputs must be URLs (sign drive paths
    first). Returns the same shape as `run`.

    Reference images: bind a description to one with `@Image1`, `@Image2`, …
    (1-indexed, in `refs` order). Portable — kept verbatim for models that read
    the tags (Seedance / Kling r2v), rewritten to prose for the rest (Veo).

    Native audio: with `audio=True` the chosen model voices dialogue, SFX and
    ambience in the SAME pass (all families support it). Write a spoken line in
    the prompt as `Speaker says: "<line>"` (a colon, not quotes around the
    speaker) and append `(no subtitles)` so it isn't stamped on the frame. Only
    reach for `lipsync_audio` for a fixed verbatim read — it looks static.

    Args:
      prompt: the scene/action (a short delivery note for lip-sync). Label
        multi-shot beats `Shot 1:` / `Cut to:`; use `@Image1`/`@Image2` for refs.
      image: a still to animate (i2v) or the portrait (lip-sync).
      last_frame: optional URL of a still the clip should END on (tail keyframe).
        Pair with `image` for first→last interpolation, or use alone to land a
        t2v/r2v clip on a set frame (e.g. an end card). Seedance / Kling only;
        ignored with a warning on other families.
      refs: reference image URLs (r2v); address them as `@Image1`, `@Image2`, ….
      lipsync_audio: narration audio URL → lip-sync mode.
      duration: seconds; snapped to the model's allowed set (e.g. Veo 4/6/8) with
        a warning. Ignored for lip-sync (length follows the audio).
      aspect_ratio / resolution: honored where the model supports them.
      audio: generate native audio (t2v/i2v/r2v) — dialogue + SFX in one pass.
    """
    inputs = _compact(
        {"prompt": prompt or None, "image": image, "last_frame": last_frame,
         "refs": refs, "lipsync_audio": lipsync_audio, "duration": duration,
         "aspect_ratio": aspect_ratio, "resolution": resolution,
         "audio": True if audio else None}
    )
    return media_generate(model=model, inputs=inputs, verb="video", output_path=output_path)


def generate_audio(
    text: str,
    *,
    model: str = "auto",
    voice: str | None = None,
    language: str | None = None,
    stability: float | None = None,
    similarity_boost: float | None = None,
    style: float | None = None,
    speed: float | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate speech from text (text-to-speech) — model-portable.

    `model` is a family (`elevenlabs/tts`) or `"auto"`, both of which resolve to
    **ElevenLabs v3** — the expressive tier that reads inline **audio tags**.
    Wrap a delivery cue in square brackets and v3 acts it out without speaking
    it: ``"[excited] It's finally here! [whispers] Don't tell anyone."`` Common
    tags: `[excited]` `[sad]` `[angry]` `[whispers]` `[laughs]` `[sighs]`
    `[sarcastic]` `[British accent]`. Lower `stability` (~0.3) makes tags land
    harder; near 1.0 they stop landing.

    v3 shapes delivery through audio tags + `stability` only. The
    `style` / `speed` / `similarity_boost` knobs are **v2-only** and are ignored
    on v3 — pass ``model="elevenlabs/tts-v2"`` if you need them (you then lose
    audio-tag support). Returns the same shape as `run`.

    Args:
      text: the words to speak (billed per character). May contain v3 audio
        tags in [brackets].
      voice: a voice name/persona (e.g. "Aria", "Roger") or a voice id.
      language: ISO 639-1 code to lock pronunciation; omit to auto-detect.
      stability: 0–1. Lower = more expressive / tag-responsive, higher =
        steadier. The primary v3 delivery knob.
      similarity_boost / style / speed: extra delivery controls — **v2 only**
        (ignored on the default v3 model).
    """
    inputs = _compact(
        {"text": text, "voice": voice, "language": language,
         "stability": stability, "similarity_boost": similarity_boost,
         "style": style, "speed": speed}
    )
    return media_generate(model=model, inputs=inputs, verb="audio", output_path=output_path)


def transcribe(
    audio: str,
    *,
    keyterms: list[str] | None = None,
    language_code: str | None = None,
    model: str = "elevenlabs/scribe-v2",
) -> dict[str, Any]:
    """Transcribe speech to text. Returns `{text, words, language_code, ...}` —
    `words` carries per-word `start`/`end` seconds. There is NO file
    (`drive_path` is empty). Billed per second of audio.

    Args:
      audio: an audio URL the model can fetch (https:// or a data: URI).
      keyterms: optional brand / proper-noun spellings to bias transcription.
      language_code: optional ISO 639-1 hint; omit to auto-detect.
    """
    inputs: dict[str, Any] = {"audio_url": audio}
    if keyterms:
        inputs["keyterms"] = keyterms
    if language_code:
        inputs["language_code"] = language_code
    return media_generate(model=model, inputs=inputs)
