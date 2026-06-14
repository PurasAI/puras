# hello-world

The smallest [Puras](https://puras.co) skillpack that still exercises the
whole surface — a good first read, and the smoke test for in-process
subagents. No external APIs and no media, so it's cheap to run: the only cost
is the `greeter`'s few Haiku turns. This is the pack the
[Building a skillpack](https://puras.co/docs/building-a-skillpack) guide
builds from scratch.

## Run it yourself

```sh
pip install puras
puras login                                  # paste a workspace API key
puras init --template hello-world            # …or clone this repo
puras deploy                                 # creates the pack in YOUR workspace on first push
puras run greeter -i name=Ada -i style=playful
```

## What's in it

Two skills in one skillpack:

| Skill | Kind | Entrypoint | What it shows |
|-------|------|-----------|---------------|
| `greeter` | **agentic** | `SKILL.md` | a custom tool, a `.md` subagent, and calling a sibling skill as a subagent |
| `formatter` | **deterministic** | `scripts/format.py:run` | a plain Python skill (no LLM) in the same skillpack |

## The flow

Run `greeter` with `{ "name": "Ada", "style": "playful" }`. It:

1. **Custom tool** — calls `emphasize` (`tools/emphasize.py`) to shout the name → `"ADA!!!"`.
2. **`.md` subagent** — `run_subagent({ target: "references/poet.md", ... })` runs the
   poet prompt as an isolated subagent that writes a two-line couplet.
3. **Sibling-skill subagent** — `run_subagent({ target: "formatter", ... })` calls the
   deterministic `formatter` skill (same skillpack) to lay out a greeting card.
4. **`set_output`** — returns `{ card, shout, poem }`.

```
greeter (agentic)
├── emphasize            ← custom Python tool
├── run_subagent("references/poet.md")   ← .md subagent  (in-process)
└── run_subagent("formatter")            ← sibling skill (in-process, deterministic)
```

## Why it's the subagent smoke test

Both subagent calls resolve inside this skillpack's own deployment, so the
worker runs them **in-process** as nested agents (no queued child job, no second
worker slot → no pipeline deadlock). This example covers both local subagent
shapes at once:

- a bundle `*.md` prompt (`references/poet.md`), and
- another manifest skill by bare name (`formatter`).

## Layout

```
hello-world/
├── greeter/
│   ├── skill.yaml            # manifest: schemas, model, the `emphasize` tool
│   ├── SKILL.md              # the agent's system prompt (the 4 steps above)
│   ├── tools/emphasize.py    # custom tool — run(text) -> {loud}
│   └── references/poet.md    # the .md subagent's prompt
└── formatter/
    ├── skill.yaml            # manifest: schemas (deterministic skill)
    └── scripts/format.py     # run(name, shout, poem) -> {card}
```

A skill's **name is its directory name**; there's no root manifest — every
`<name>/skill.yaml` is discovered automatically. The entrypoint suffix decides
the kind: `SKILL.md` → agentic, `scripts/x.py:func` → deterministic.

## Learn more

- [Building a skillpack](https://puras.co/docs/building-a-skillpack) — this pack, built step by step
- [skill.yaml reference](https://puras.co/docs/skill-yaml-reference) — every field
- [CLI reference](https://puras.co/docs/cli-reference) — deploy / run / logs / secrets
- Starting your own pack? [skillpack-template](https://github.com/PurasAI/skillpack-template)
  is the blank version of this layout.

---

This repo is a **read-only mirror** synced from the Puras monorepo. Issues
are welcome here; pull requests will be overwritten by the next sync. MIT
licensed.
