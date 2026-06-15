"""puras — bundled Python SDK for user functions running on the platform.

Usage from a deployed function:

    from puras import media, secret

    def render(inputs: dict) -> dict:
        api_key = secret("APP_SUPABASE_KEY")   # skillpack secret as env var
        img = media.generate_image(inputs["prompt"])
        return {"drive_path": img["drive_path"], "billed_usd": img["billed_usd"]}

These helpers read the per-job context the worker injects:
    PURAS_API_BASE          # http://localhost:8000 (or prod URL)
    PURAS_SERVICE_TOKEN     # shared with API for internal-only endpoints
    PURAS_WORKSPACE_ID      # the calling workspace (drive + billing tenant)
    PURAS_JOB_ID            # the current job
"""

__version__ = "0.3.3"

from . import drive, inputs, media, subagent  # noqa: F401  — usable as `puras.media.generate_image(...)`
from .client import Client, JobError, PurasAPIError  # noqa: F401  — external skill calls
from .inputs import load_bytes, load_path  # noqa: F401
from .media import (  # noqa: F401  — also importable directly
    generate_audio,
    generate_image,
    generate_video,
    transcribe,
)
from .secrets import secret  # noqa: F401
from .subagent import SubagentRunError  # noqa: F401

__all__ = [
    "media", "secret", "inputs", "load_bytes", "load_path",
    "generate_image", "generate_video", "generate_audio", "transcribe",
    "drive", "subagent", "SubagentRunError",
    "Client", "JobError", "PurasAPIError",
]
