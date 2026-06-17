<div align="center">

# Puras — local skill runner

**Run AI skills on your own machine, on your own LLM key — no account.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![PyPI](https://img.shields.io/pypi/v/puras.svg)](https://pypi.org/project/puras/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

[Docs](https://puras.co/docs) · [Examples](./examples) · [What's a skill?](#whats-a-skill) · [Build a skill](#build-your-own-skill)

</div>

---

This is the open-source runner that executes a Puras **skill** entirely on your
laptop: no Postgres, no bucket, no platform API, no sign-up. It's the *same*
agent loop the hosted platform runs (one loop, two environments), so a skill
behaves identically locally and in prod.

```bash
pip install "puras[local]"
puras run --local greeter --dir ./examples/hello-world -i name=Ada
```

- 🧑‍💻 **Local-first** — run and iterate on a skill before deploying anything.
- 🔌 **Local API** — `puras serve` exposes the hosted job API on `localhost`, so you build and test your app offline.
- 🔑 **Bring your own key** — your provider, your bill. If a key isn't set, the CLI asks for it.
- 🪶 **Dependency-light** — the offline path needs no DB/bucket/openai stack.
- 🔁 **Prod parity** — the same loop and contracts as the hosted platform.

## What's a skill?

A **skill** is a small folder that an agent runs end to end. At its simplest
it's two files — a prompt and an input/output contract:

```
summarize/
  SKILL.md      # the prompt — the agent's instructions
  skill.yaml    # the contract — input/output schema (+ optional model, tools, evals)
```

`skill.yaml` declares what goes in and what comes out:

```yaml
title: Summarizer
description: Condense a block of text into two plain sentences.
entrypoint: SKILL.md          # markdown entrypoint = agentic (the file is the system prompt)

input_schema:
  type: object
  required: [text]
  properties:
    text:
      type: string
      description: The text to condense.

output_schema:
  type: object
  properties:
    summary:
      type: text
      description: A two-sentence summary.
```

`SKILL.md` is the system prompt the agent runs with:

```markdown
You summarize text. Read the `text` input, then call `set_output` once with a
`summary` of at most two plain sentences. Don't add opinions or extra detail.
```

That's a complete skill. Run it locally:

```bash
puras run --local summarize --dir ./my-skillpack -i text="...the article..."
```

Events stream to your console as the agent works; the final JSON output prints
at the end. A skill can grow from here — declare Python `tools:` the agent can
call, hand stages to isolated subagents, add `evals:` — but two files is the
floor.

### From skill to product pipeline

A skill is one stage; a **pipeline** is your app orchestrating skills behind a
real API. You don't rewrite anything to get there — `puras serve` stands up the
exact job API your app will call in production, backed by the local runner:

```bash
puras serve --dir ./my-skillpack          # → http://127.0.0.1:8787
```

Now your app calls the skill over HTTP, the same way every stage of a pipeline
does. Chain a few skills (`extract` → `summarize` → `publish`) and you have a
pipeline — each one a folder, each one independently runnable and testable:

```python
import puras

client = puras.Client(api_key="local", api_base="http://127.0.0.1:8787", skillpack="local")

text = fetch_article(url)                       # your code
summary = client.run("summarize", {"text": text})["summary"]
post = client.run("write_post", {"summary": summary})
```

The only thing that changes when you ship is the base URL: point the client at
`https://api.puras.co`, `puras deploy`, and the **same app code** runs against
the managed platform.

## Run a skill

```bash
puras run --local <skill> --dir <skillpack> -i KEY=VALUE [-i KEY2=VALUE2 ...]
```

- `<skill>` — the skill to run; omit it when the bundle has exactly one.
- `--dir` — the skillpack bundle root (a folder of `<skill>/skill.yaml`). Defaults to `.`.
- `-i KEY=VALUE` — an input, repeatable; validated against the skill's `input_schema`.
- `--model claude/sonnet-4-6` — override the skill's model for this run.

The bundled `hello-world` skillpack has two skills to start from — `greeter`
(agentic) and `formatter` (deterministic):

```bash
puras run --local greeter --dir ./examples/hello-world -i name="the Puras team"
```

From a checkout of this repo instead of PyPI:

```bash
pip install -e .             # puras-runner (the runtime)
pip install -e worker/sdk    # the puras CLI + SDK
```

Programmatic use is the same loop:

```python
from worker.local_run import run_local

res = run_local("./examples/hello-world", {"name": "Ada"}, skill="greeter")
print(res["output"])
```

## Run a skill's evals

Evals are to a skill what unit tests are to code. If a skill declares an `evals:`
block, run its suite locally and gate on it:

```bash
puras eval --local content-repurposer --dir ./examples/content-studio --threshold 80
```

`check` / `exact_match` / `schema` graders run free; a `rubric` (LLM-as-judge)
grader runs on your key. `--threshold N` is a CI gate — non-zero exit if the
pass-rate is below `N`.

## Build your app against a local API

`puras run --local` answers *"does my skill work?"*. When you're building the
**app** that calls the skill, you want the other half: a local server that speaks
the same API your app will hit in production. That's `puras serve`:

```bash
puras serve --dir ./examples/hello-world          # → http://127.0.0.1:8787
```

It mirrors the hosted **job API** (`POST /v1/jobs`, `GET /v1/jobs/{id}`,
`…/events`, `…/spans`) backed by the offline runner — in-memory, zero extra
dependencies. Point any Puras SDK at it by changing one thing — the base URL —
and your app runs unchanged, offline:

```python
import puras

# api_base is the only thing that differs between local and prod
client = puras.Client(api_key="local", api_base="http://127.0.0.1:8787", skillpack="local")
print(client.run("greeter", {"name": "Ada"}))
```

```ts
import { Puras } from "puras";
const puras = new Puras({ apiKey: "local", apiBase: "http://127.0.0.1:8787", skillpack: "local" });
console.log(await puras.run("greeter", { name: "Ada" }));
```

…or just curl it:

```bash
curl -s "http://127.0.0.1:8787/v1/jobs?wait=true" \
  -H "content-type: application/json" \
  -d '{"skill": "greeter", "inputs": {"name": "Ada"}}'
```

Auth is open locally; `--require-key <token>` emulates API-key auth, and
`--host` / `--port` change where it binds. (The Python and React-Native SDKs
poll, so they work as-is; live SSE streaming is a hosted feature.)

## Build your own skill

```bash
cp -r examples/skillpack-template my-skillpack
$EDITOR my-skillpack/my-skill/SKILL.md      # the prompt
$EDITOR my-skillpack/my-skill/skill.yaml    # schema, model, tools, evals
puras run --local --dir ./my-skillpack -i topic=otters
```

```
my-skillpack/
  my-skill/
    SKILL.md          # the agent's instructions (system prompt)
    skill.yaml        # input/output schema + model + tools + evals
    tools/...         # optional deterministic Python tools the agent can call
```

When you're happy, the same bundle deploys to the hosted platform unchanged.
See the [docs](https://puras.co/docs) to deploy and call skills over the API.

## Open-source vs Cloud

Puras is **open-core**: the runner — the agent loop and the local tool surface —
is MIT-licensed and runs fully offline, forever. The hosted platform at
[puras.co](https://puras.co) is how the project is sustainably funded, and it
*adds* the managed surface that can't exist on a single laptop. Premium isn't a
crippled core — it's the capabilities that need real infrastructure.

|                              | **Local runner** (this repo, MIT)         | **Puras Cloud** (hosted)                       |
| ---------------------------- | ----------------------------------------- | ---------------------------------------------- |
| Setup & maintenance          | `pip install`, you run it                 | Fully managed, nothing to install              |
| LLM key & billing            | Bring your own key, you pay the provider  | Managed, usage-based, transparent pricing      |
| Agent loop & local tools     | ✓ text, `bash`, file tools, your Python tools, in-process subagents | ✓ same loop                |
| Job API for your app         | ✓ `puras serve` — the job API on localhost | ✓ api.puras.co — managed, scaled, durable     |
| Evals (`check`/`schema`/`rubric`) | ✓ per run + offline suites           | ✓ + suites at scale, CI gating, version diffs  |
| Media (image/video/audio)    | ✓ `generate_*` direct to the provider     | ✓ generation + persistence (bucket-backed)     |
| Web search / fetch / browser | ✓ search + fetch                          | ✓ search / fetch / browser screenshots         |
| Shared memory                | ✓ persistent, local SQLite                | ✓ persistent, workspace-scoped + semantic (pgvector) |
| Persistent storage           | —                                         | ✓ bucket-backed drive                          |
| Durable resume               | —                                         | ✓ checkpointed, survives worker restarts       |
| Hindsight (retrospectives)   | —                                         | ✓ mines stored run traces for recurring inefficiencies, surfaces fixes |
| Budgets, tracing, dashboard  | console events + a token tally            | ✓ spend budgets, OTel spans, run timelines     |
| Marketplace & sharing        | —                                         | ✓                                              |
| Support                      | [Issues](../../issues) & [Discussions](../../discussions) | priority / SLA              |

The local runner gives you the free, offline core; the hosted platform adds the
managed surface — persistence, scale, durable resume, retrospectives — for when
you ship. Most tools work the same in both places: media and web run on your own
keys locally and through the managed services in the cloud, and workspace memory
persists across `puras run --local` invocations in a local SQLite store. The
included examples (`hello-world`, `skillpack-template`, `content-studio`) use
only the local surface and run end-to-end offline.

## How it works

The runner runs the agent on a `LocalRunContext` with `platform_enabled=False`
(but `memory_enabled=True` — its `memory_backend()` returns the SQLite store).
That `RunContext` seam is the whole trick: one agent loop, two environments. The
offline import path is kept **dependency-light** — no Postgres/bucket/openai at
import time — and that's enforced by
`tests/dry/test_local_import_isolation.py`.

| Path | What |
| --- | --- |
| `worker/` | The `puras-runner` runtime — the agent loop (`agent_runner.py`), the `RunContext` seam, the local entrypoint (`local_run.py`), skill loading, the eval runner. |
| `worker/sdk/` | The `puras` package — the CLI and the SDK skills import at runtime. Ships as its own wheel. |
| `examples/` | Runnable, offline-capable skillpacks. |
| `tests/` | The dependency-light import-isolation guard, a CLI smoke test, and the `puras serve` API tests. |

## Community & contributing

Questions, bugs, and skill ideas are welcome in
[Issues](../../issues) and [Discussions](../../discussions). PRs that improve the
runner, the docs, or the examples are appreciated.

## License

[MIT](./LICENSE). The hosted platform's server-side code is separate and
commercial — this runner, the SDK, and the examples are MIT and yours to use,
modify, and self-run.
