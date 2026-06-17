<div align="center">

# Puras — local skill runner

**Run AI skills end to end on your own machine.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![PyPI](https://img.shields.io/pypi/v/puras.svg)](https://pypi.org/project/puras/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

[Why Puras?](#why-puras) · [Getting started](#getting-started) · [Open-source vs Cloud](#open-source-vs-cloud) · [Community](#community) · [License](#license)

</div>

---

Puras turns a prompt into a typed, testable, deployable **skill** — and runs it
on your own machine.

## Why Puras?

You can already fire a prompt at an LLM SDK and get a string back. The problem is
everything around it: validating inputs, wiring up tool calls, multi-step
orchestration, forcing structured output, retries, and tests — glue code you
write and maintain per prompt. A skill is that prompt promoted to a real unit:

- a typed **input/output contract** — schema-validated in, schema-valid JSON out, every time;
- the **tools and sub-steps** the agent may use, declared in one place;
- **evals** that test the prompt like code, with a CI gate;
- **one loop, two environments** — the same agent loop runs offline and hosted, so you build against a local API and **ship the identical bundle to production** with `puras deploy`.

If all you need is a single completion, reach for the SDK — Puras earns its place
the moment a prompt becomes something you run repeatedly, test, and ship.

## Getting started

Install the CLI and the offline runner:

```bash
pip install "puras[local]"
```

A skill is a folder — a prompt and an input/output contract. Create two files.
`skill.yaml` declares what goes in and what comes out:

```yaml
# summarize/skill.yaml
title: Summarizer
description: Condense a block of text into two plain sentences.
entrypoint: SKILL.md          # markdown entrypoint = agentic; the file is the system prompt

input_schema:
  type: object
  required: [text]
  properties:
    text:
      type: string

output_schema:
  type: object
  properties:
    summary:
      type: text
```

`SKILL.md` is the system prompt the agent runs with:

```markdown
<!-- summarize/SKILL.md -->
You summarize text. Read the `text` input and call `set_output` once with a
`summary` of at most two plain sentences — no opinions, no extra detail.
```

Serve it. `puras serve` exposes the same job API your app will hit in
production, backed by the local runner on your own LLM key (if no key is set,
the CLI asks for one):

```bash
puras serve          # serves ./ → http://127.0.0.1:8787
```

Now call the skill from your app — point any Puras SDK at the local base URL:

```python
import puras

client = puras.Client(api_key="local", api_base="http://127.0.0.1:8787", skillpack="local")
out = client.run("summarize", {"text": "...a long article..."})
print(out["summary"])
```

```ts
import { Puras } from "puras";

const puras = new Puras({ apiKey: "local", apiBase: "http://127.0.0.1:8787", skillpack: "local" });
const { summary } = await puras.run("summarize", { text: "...a long article..." });
console.log(summary);
```

That's the whole loop. When you ship, change the base URL to
`https://api.puras.co`, run `puras deploy`, and the **same app code** runs
against the managed platform — nothing else changes.

> Want to iterate faster? `puras run --local summarize -i text="…"` runs a skill
> straight from the CLI, `puras eval --local` gates it on evals, and the
> [`examples/`](./examples) folder has ready-to-run skillpacks.

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
you ship the same bundle unchanged.

## Community

Questions, bugs, and skill ideas are welcome in
[Issues](../../issues) and [Discussions](../../discussions). PRs that improve the
runner, the docs, or the examples are appreciated.

Working from a checkout instead of PyPI:

```bash
pip install -e .             # puras-runner (the runtime)
pip install -e worker/sdk    # the puras CLI + SDK
```

## License

[MIT](./LICENSE). The hosted platform's server-side code is separate and
commercial — this runner, the SDK, and the examples are MIT and yours to use,
modify, and self-run.
