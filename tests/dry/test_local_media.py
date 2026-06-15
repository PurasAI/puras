"""Local media generate_* path (FAL_KEY → call Fal directly, no platform).

Covers the three seams the feature adds:
  - `_build_tools` keeps the media verbs offline ONLY when media is enabled;
  - `_local_media_enabled` is the local+FAL_KEY gate;
  - `media_local.generate` resolves a verb, inlines a local drive ref as a
    data URI, calls Fal, and returns the hosted result shape.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from worker import agent_runner, media_local
from worker.config import get_settings


@pytest.fixture(autouse=True)
def _hosted_env_placeholders(monkeypatch):
    """`WorkerSettings` hard-requires the hosted env; fill harmless placeholders
    for the bits a local run never touches (mirrors local_run._prepare_env) so
    `get_settings()` is loadable in these dry tests."""
    for name in (
        "DATABASE_URL", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
        "PURAS_SERVICE_TOKEN", "ANTHROPIC_API_KEY",
    ):
        monkeypatch.setenv(name, "x")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@dataclass
class _Skill:
    name: str = "demo"
    tools: list = field(default_factory=list)
    disable_bash: bool = False
    is_adhoc: bool = False
    output_schema: Any = None
    allowed_tools: Any = None


_MEDIA = {"generate_image", "generate_video", "generate_audio", "transcribe"}


def _tool_names(skill, **kw):
    tools, _ = agent_runner._build_tools(skill, **kw)
    return {t["name"] for t in tools}


def test_media_verbs_dropped_offline_without_fal():
    names = _tool_names(_Skill(), platform_enabled=False, media_enabled=False)
    assert not (names & _MEDIA), names


def test_media_verbs_kept_offline_with_fal():
    names = _tool_names(_Skill(), platform_enabled=False, media_enabled=True)
    assert _MEDIA <= names, names
    # The OTHER platform-only tools stay off — only media is re-enabled.
    assert "web_search" not in names
    assert "memory_search" not in names


def test_media_verbs_present_hosted():
    names = _tool_names(_Skill(), platform_enabled=True)
    assert _MEDIA <= names


class _LocalCtx:
    platform_enabled = False


class _HostedCtx:
    platform_enabled = True


def test_local_media_enabled_gate(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("FAL_KEY", "fal-xyz")
    assert agent_runner._local_media_enabled(_LocalCtx()) is True
    # Hosted never takes the local Fal path regardless of the env.
    assert agent_runner._local_media_enabled(_HostedCtx()) is False
    get_settings.cache_clear()
    monkeypatch.delenv("FAL_KEY", raising=False)
    assert agent_runner._local_media_enabled(_LocalCtx()) is False
    get_settings.cache_clear()


def _fake_fal(captured: dict):
    mod = types.ModuleType("fal_client")

    def subscribe(endpoint, arguments=None, with_logs=False):
        captured["endpoint"] = endpoint
        captured["arguments"] = arguments
        return {"images": [{"url": "https://fal.example/out.png", "width": 1024}]}

    mod.subscribe = subscribe
    return mod


def test_generate_image_resolves_and_calls_fal(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("FAL_KEY", "fal-xyz")
    captured: dict = {}
    monkeypatch.setitem(sys.modules, "fal_client", _fake_fal(captured))

    out = media_local.generate(
        "auto", {"prompt": "a cat", "aspect_ratio": "1:1"},
        "ws-local", "job-1", verb="image", output_dir="demo/abc",
    )
    # auto image → nano-banana-pro; native prompt passed through.
    assert captured["endpoint"] == "fal-ai/nano-banana-pro"
    assert captured["arguments"]["prompt"] == "a cat"
    assert out["model"] == "google/nano-banana-pro"
    assert out["kind"] == "image"
    assert out["drive_path"].startswith("demo/abc/")
    assert out["output_url"] == "https://fal.example/out.png"
    get_settings.cache_clear()


def test_local_drive_ref_inlined_as_data_uri(monkeypatch, tmp_path):
    get_settings.cache_clear()
    monkeypatch.setenv("FAL_KEY", "fal-xyz")
    monkeypatch.setenv("LOCAL_DRIVE_PATH", str(tmp_path))
    from worker import drive
    drive.setup_drive()
    # A reference image sitting in the local drive (no bucket to sign against).
    ref = drive.workspace_drive("ws-local") / "brand" / "logo.png"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"\x89PNG\r\n\x1a\nDATA")

    captured: dict = {}
    monkeypatch.setitem(sys.modules, "fal_client", _fake_fal(captured))

    media_local.generate(
        "auto", {"prompt": "edit @Image1", "refs": ["drive/brand/logo.png"]},
        "ws-local", "job-2", verb="image",
    )
    urls = captured["arguments"]["image_urls"]
    assert urls[0].startswith("data:image/png;base64,"), urls
    get_settings.cache_clear()


def test_missing_drive_ref_is_a_clean_error(monkeypatch, tmp_path):
    get_settings.cache_clear()
    monkeypatch.setenv("FAL_KEY", "fal-xyz")
    monkeypatch.setenv("LOCAL_DRIVE_PATH", str(tmp_path))
    from worker import drive
    drive.setup_drive()
    monkeypatch.setitem(sys.modules, "fal_client", _fake_fal({}))
    with pytest.raises(media_local.LocalMediaError):
        media_local.generate(
            "auto", {"prompt": "edit", "refs": ["brand/missing.png"]},
            "ws-local", "job-3", verb="image",
        )
    get_settings.cache_clear()


def test_generate_without_fal_key_errors(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(media_local.LocalMediaError):
        media_local.generate("auto", {"prompt": "x"}, "ws", "job", verb="image")
    get_settings.cache_clear()
