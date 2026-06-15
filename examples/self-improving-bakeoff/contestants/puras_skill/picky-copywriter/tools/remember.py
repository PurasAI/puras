"""remember — save one durable lesson about the client's style guide.

De-dupes so repeating a rule across briefs doesn't pile up. Backs onto the same
session-scoped JSON memory recall reads (the offline stand-in for Puras Cloud's
workspace memory / memory_put)."""

from __future__ import annotations

import json
from pathlib import Path


def run(lesson: str) -> dict:
    spec = json.loads(Path("_inputs.json").read_text())
    sd = Path(spec["session_dir"])
    sd.mkdir(parents=True, exist_ok=True)
    mem = sd / "memory.json"
    data = json.loads(mem.read_text()) if mem.exists() else {"lessons": []}
    lesson = (lesson or "").strip()
    if lesson and lesson not in data["lessons"]:
        data["lessons"].append(lesson)
    mem.write_text(json.dumps(data))
    return {"saved": bool(lesson), "count": len(data["lessons"])}
