# skillpack-template

The minimal starting point for a [Puras](https://puras.co) skillpack — one
agentic skill with placeholders where your logic goes. Use it for a clean
slate; for a worked example of every moving part (custom tools, subagents, a
deterministic skill) start from
[hello-world](https://github.com/PurasAI/hello-world) instead.

## Get it

Three equivalent ways:

```sh
pip install puras
puras init            # scaffolds these files into the current directory
```

…or click **Use this template** on GitHub, or clone this repo.

## Ship it

```sh
puras login                              # paste a workspace API key
puras deploy                             # creates the skillpack on first push
puras run my-skill -i prompt="Say hello"
```

`puras init` (or the first `puras deploy`) writes `puras.yaml` — the pack
manifest that binds this directory to your skillpack and carries the pack
page's title and description.

## What's here

```
my-skill/
├── skill.yaml   # the contract: schemas, model, tools
└── SKILL.md     # the agent's system prompt (the entrypoint)
AGENTS.md        # orientation for coding agents working in this directory
```

Rename `my-skill/` to name the skill; add more `<name>/skill.yaml` folders
for more skills — every top-level folder with a `skill.yaml` is one skill.

## Learn more

- [Building a skillpack](https://puras.co/docs/building-a-skillpack) — the guided tour
- [skill.yaml reference](https://puras.co/docs/skill-yaml-reference) — every field
- [CLI reference](https://puras.co/docs/cli-reference) — deploy / run / logs / secrets

---

This repo is a **read-only mirror** synced from the Puras monorepo. Issues
are welcome here; pull requests will be overwritten by the next sync. MIT
licensed.
