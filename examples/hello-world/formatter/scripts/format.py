"""`format` — the deterministic greeting-card layout for the hello-world skillpack.

A plain `scripts/<file>.py:<func>` skill (no LLM): it takes a name plus an
optional shout and couplet and assembles a small text "card" framed in a box.
The `greeter` agentic skill calls this as a subagent; it's also runnable on its
own. Pure stdlib, so it needs no requirements.txt / venv.

The worker calls the entrypoint as `run(**inputs)` and validates the returned
dict against the skill's `output_schema`.
"""

from __future__ import annotations


def run(name: str, shout: str = "", poem: str = "") -> dict:
    headline = (shout or f"Hello, {name}!").strip()
    poem_lines = [ln.strip() for ln in (poem or "").splitlines() if ln.strip()]

    # Inner width fits the longest line; the box adds one space of padding each
    # side, so a row is `║ <centered(inner)> ║`.
    inner = max([len(headline), *(len(ln) for ln in poem_lines), 22])
    top = "╔" + "═" * (inner + 2) + "╗"
    bottom = "╚" + "═" * (inner + 2) + "╝"
    sep = "╟" + "─" * (inner + 2) + "╢"

    rows = [top, "║ " + headline.center(inner) + " ║"]
    if poem_lines:
        rows.append(sep)
        rows.extend("║ " + ln.center(inner) + " ║" for ln in poem_lines)
    rows.append(bottom)

    return {"card": "\n".join(rows)}
