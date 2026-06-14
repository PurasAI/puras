"""`emphasize` — a trivial custom tool for the hello-world greeter.

Uppercases a short string and adds emphatic punctuation. Pure stdlib, no deps,
so there's no requirements.txt and it runs on the worker's base interpreter. It
exists only to show how a skill declares (in skill.yaml `tools:`) and the agent
calls its own Python tool.

A tool entrypoint is `<file>:<func>`; the function's params are the tool's
declared `input_schema` properties, and it returns a dict shaped like the
tool's `output_schema`.
"""

from __future__ import annotations


def run(text: str) -> dict:
    cleaned = " ".join((text or "").split())  # collapse whitespace
    loud = cleaned.upper().rstrip("!?. ")
    return {"loud": f"{loud}!!!" if loud else "!!!"}
