"""Starter files for `puras init`.

Two template sources:

  blank        — `BLANK_FILES` below, written straight to disk. Embedded in
                 the package so `init` works offline and the scaffold always
                 matches the installed CLI. The same bytes are committed at
                 `examples/skillpack-template/` (mirrored to
                 github.com/PurasAI/skillpack-template); a dry test and the
                 sync workflow keep the two in lockstep.
  hello-world  — the full worked example (agentic + deterministic skill, a
                 custom tool, two subagents), fetched at init time from its
                 public mirror. Skills are parsed server-side at deploy, so
                 fetching latest tracks the live platform rather than the
                 installed CLI — exactly what a fresh example should do.

Module-level imports stay stdlib-minimal on purpose: the monorepo's dry tests
load this file standalone (no package init) to diff `BLANK_FILES` against the
committed template.
"""

from __future__ import annotations

from pathlib import Path

HELLO_WORLD_REPO = "PurasAI/hello-world"
_TARBALL_URL = "https://codeload.github.com/{repo}/tar.gz/refs/heads/main"

# The remote binding is per-workspace: `init` writes the caller's own.
_FETCH_SKIP = {"puras.yaml"}

_BLANK_SKILL_YAML = """\
title: My Skill
description: >
  One line on what this skill does — shown on the skill page and read by
  agents deciding when to call it.

# A `SKILL.md` entrypoint makes this skill AGENTIC: the markdown is the system
# prompt of an agent run with your declared tools. A `scripts/x.py:func`
# entrypoint makes it deterministic instead — plain Python, no LLM.
entrypoint: SKILL.md

# Optional `family/variant` model for agentic skills — omit for the platform
# default. Media slots exist too: image_model / video_model / audio_model.
# text_model: claude/haiku-4-5

input_schema:
  type: object
  required: [prompt]
  properties:
    prompt:
      type: string
      description: What the user wants, in their words.

# Declare output fields only — the platform auto-requires every field and
# prunes anything extra. `type: text` renders multi-line in the playground;
# richer types (image, video, color) render uploads and pickers.
output_schema:
  type: object
  properties:
    answer:
      type: text
      description: The skill's result.

# Real inputs the playground offers as one-click starting points.
examples:
  - title: Try it
    inputs:
      prompt: Introduce yourself in one short paragraph.
"""

_BLANK_SKILL_MD = """\
You are <what this skill is — its job, its voice, its limits>. This file is
the system prompt your agent runs with; replace the placeholders with the
skill's real instructions.

## Inputs

Your inputs are in the first message:
- `prompt` — what the user asked for.

## Steps

1. <The actual work. Call your declared tools here, or hand a stage to an
   isolated subagent with `run_subagent`.>
2. Call `set_output` once with exactly the fields in `output_schema`:
   `{ "answer": <the result> }`.

## Guardrails

- <What this skill must never do; when to return an error instead of
  pushing on.>
"""

_AGENTS_MD = """\
# Working on a Puras skillpack

This directory is a **Puras skillpack**: a bundle of AI skills deployed and
run on [Puras](https://puras.co). Each top-level folder containing a
`skill.yaml` is one skill — the folder name is the skill name. There is no
root manifest; `puras.yaml` (written by `puras init` / `puras deploy`) binds
the directory to its remote skillpack and carries the pack page's
title/description.

## The contract

- `skill.yaml` declares the contract: title, description, `input_schema`,
  `output_schema`, optional `tools:`, optional `examples:`.
- The **entrypoint suffix decides the kind**: `entrypoint: SKILL.md`
  (+ optional `text_model: family/variant`) = agentic — the markdown is the
  agent's system prompt. `entrypoint: scripts/x.py:func` = deterministic —
  plain Python; the worker calls `func(**inputs)` and validates the returned
  dict.
- Schemas use the Puras dialect (JSON Schema + extras like `type: text`,
  `image`, `video`). Input schemas keep an explicit `required`; output
  schemas declare fields only (the platform auto-requires and prunes).
- Every skill ends by producing exactly the `output_schema` fields — agentic
  skills call the auto-injected `set_output` tool once; deterministic skills
  return the dict.
- Custom tools under `tools:` are deterministic Python (`file.py:func`)
  callable by this skill's agent. `run_subagent` runs a bundle `*.md` prompt
  or a sibling skill as an isolated subagent.

## Dev loop

```sh
pip install puras
puras login                      # or set PURAS_API_KEY
puras deploy                     # zip this dir + push a deployment
                                 # (creates the skillpack on first push)
puras run <skill> -i key=value   # submit a job and wait
puras logs <job_id>              # stream a job's events
puras feedback <job_id> --up     # rate a result (👍/👎 + --comment)
```

## References

- Building a skillpack: https://puras.co/docs/building-a-skillpack
- skill.yaml, field by field: https://puras.co/docs/skill-yaml-reference
- Agent runtime tools (`run_subagent`, `set_output`, media verbs):
  https://puras.co/docs/agent-tools-reference
- CLI: https://puras.co/docs/cli-reference
"""

BLANK_FILES: dict[str, str] = {
    "my-skill/skill.yaml": _BLANK_SKILL_YAML,
    "my-skill/SKILL.md": _BLANK_SKILL_MD,
    "AGENTS.md": _AGENTS_MD,
}


def scaffold_blank(root: Path) -> list[str]:
    """Write the blank starter into `root`. Never overwrites: existing files
    are skipped, so it's safe in a dir that already has e.g. an AGENTS.md."""
    written: list[str] = []
    for rel, content in BLANK_FILES.items():
        dest = root / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        written.append(rel)
    return written


def scaffold_hello_world(root: Path) -> list[str]:
    """Fetch the hello-world example from its public mirror into `root`.

    Plain codeload tarball — no git and no GitHub token required. The repo's
    `puras.yaml` is skipped (the binding is per-workspace; `init` writes the
    caller's own) and existing files are never overwritten."""
    import io
    import tarfile

    import httpx

    from .commands import CliError  # function-level: commands imports us too

    url = _TARBALL_URL.format(repo=HELLO_WORLD_REPO)
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=30)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001 — any fetch failure gets the same exit
        raise CliError(
            f"couldn't fetch the hello-world template ({e}) — clone it "
            f"instead: https://github.com/{HELLO_WORLD_REPO}"
        ) from None

    written: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = Path(m.name).parts[1:]  # drop the tarball's top-level dir
            if not parts:
                continue
            rel = Path(*parts)
            if rel.is_absolute() or ".." in rel.parts or str(rel) in _FETCH_SKIP:
                continue
            dest = root / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            f = tf.extractfile(m)
            dest.write_bytes(f.read() if f else b"")
            written.append(str(rel))
    return written
