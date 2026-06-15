"""recall — return the lessons learned about the client's style so far.

Local memory backing: a JSON file under the session dir, persisting across the
round-by-round jobs of one session. On Puras Cloud this same role is played by the
workspace memory (memory_search) — the local file is the offline stand-in so the
experiment iterates fast; the final head-to-head re-validates on Cloud.
"""

from __future__ import annotations

import json
from pathlib import Path


def run(query: str = "") -> dict:
    spec = json.loads(Path("_inputs.json").read_text())
    mem = Path(spec["session_dir"]) / "memory.json"
    lessons = json.loads(mem.read_text()).get("lessons", []) if mem.exists() else []
    return {"lessons": lessons}
