<div align="center">

# Puras — local skill runner

**Run [Puras](https://puras.co) AI skills on your own machine, on your own LLM key — no account.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![PyPI](https://img.shields.io/pypi/v/puras.svg)](https://pypi.org/project/puras/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

[Docs](https://puras.co/docs) · [Cloud](https://puras.co) · [Examples](./examples) · [Build a skill](#build-your-own-skill)

</div>

---

A **skill** is a small folder — a prompt, an input/output schema, optional Python
tools, and evals — that an agent runs end to end. This is the open-source runner
that executes one entirely **on your laptop**: no Postgres, no bucket, no platform
API, no sign-up. It's the *same* agent loop the hosted platform runs (one loop,
two environments), so a skill behaves identically locally and in prod.

```bash
pip install "puras[local]"
puras run --local greeter --dir ./examples/hello-world -i name=Ada
```

- 🧑‍💻 **Local-first** — run and iterate on a skill before deploying anything.
- 🔌 **Local API** — `puras serve` exposes the hosted job API on `localhost`, so you build and test your app offline before you deploy.
- 🔑 **Bring your own key** — your provider, your bill, nothing billed by a platform.
- 🪶 **Dependency-light** — the offline path needs no DB/bucket/openai stack.
- 🔁 **Prod parity** — the same loop and contracts as [Puras Cloud](https://puras.co).

## Getting started

### Run it locally

The whole point of this repo — a skill's agent loop on your machine, on your key:

```bash
pip install "puras[local]"        # the puras CLI + the offline runner

# the bundled "hello world" skillpack has two skills: greeter + formatter
puras run --local greeter --dir ./examples/hello-world -i name="the Puras team"
```

It's **BYO key**: you call the provider directly and pay your own bill. The CLI
reads your key from `$ANTHROPIC_API_KEY` (or the `--api-key` flag) and, if
neither is set, prompts for it in the terminal.

From a checkout of this repo instead of PyPI:

```bash
pip install -e .             # puras-runner (the runtime)
pip install -e worker/sdk    # the puras CLI + SDK
```

### …or use Puras Cloud (recommended for production)

The fastest way to run a skill with the **full tool surface** — media generation,
web search/fetch, shared memory, persistent storage, durable resume, and
eval suites at scale — is to [sign up free at puras.co](https://puras.co). No
infrastructure to manage, automatic scaling, and a one-line MCP connect for
Claude Code. The local runner here gives you the free, offline core; Cloud adds
the managed, hosted surface for when you ship. The [comparison below](#open-source-vs-cloud)
spells out exactly which is which.

## Run a skill

```bash
puras run --local <skill> --dir <skillpack> -i KEY=VALUE [-i KEY2=VALUE2 ...]
```

- `<skill>` — the skill to run; omit it when the bundle has exactly one.
- `--dir` — the skillpack bundle root (a folder of `<skill>/skill.yaml`). Defaults to `.`.
- `-i KEY=VALUE` — an input, repeatable; validated against the skill's `input_schema`.
- `--model claude/sonnet-4-6` — override the skill's model for this run.
- `--api-key sk-...` — your LLM key, if it isn't already in the environment.

Events stream to your console as the agent works; the final JSON output and a
token tally (informational — you paid your provider, not Puras) print at the end.

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
grader runs on your BYO key. `--threshold N` is a CI gate — non-zero exit if the
pass-rate is below `N`.

A suite runs in **suite mode**: the agent's side-effecting tools are
short-circuited with stubs, so testing a skill never renders media, sends email,
or writes for real. Built-in side-effecting verbs (`generate_image`/`video`/
`audio`, `transcribe`, `web_*`, `download_url`, `send_email`, `memory_put`/
`forget`) get a safe default stub; declare `evals.mocks: { <tool>: <response> }`
in `skill.yaml` to feed a realistic value or to mock a **custom** tool, and a
dataset case may carry its own `mocks: {...}` to override per case. Pure/local
tools (`bash`, `file_*`, `todo_write`) always run for real.

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
and your app runs unchanged, offline, on your own key:

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

When you ship, change the base URL to `https://api.puras.co` and `puras deploy` —
the **same app code** now runs against the managed platform. Auth is open
locally; `--require-key <token>` emulates API-key auth, and `--host` / `--port`
change where it binds. (The Python and React-Native SDKs poll, so they work
as-is; live SSE streaming is a Cloud feature.)

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

When you're happy, the same bundle deploys to [Puras Cloud](https://puras.co)
unchanged. See the [docs](https://puras.co/docs) to deploy and call skills over
the API.

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
| Media (image/video/audio)    | —                                         | ✓ generation + persistence                     |
| Web search / fetch / browser | —                                         | ✓                                              |
| Shared memory & storage      | —                                         | ✓ persistent, workspace-scoped                 |
| Durable resume               | —                                         | ✓ checkpointed, survives worker restarts       |
| Budgets, tracing, dashboard  | console events + a token tally            | ✓ spend budgets, OTel spans, run timelines     |
| Marketplace & sharing        | —                                         | ✓                                              |
| Support                      | [Issues](../../issues) & [Discussions](../../discussions) | priority / SLA              |

The hosted-only tools (`worker/agent_runner.py:PLATFORM_ONLY_TOOLS`) are simply
not offered to the model offline, so a skill that needs them still runs — it just
won't see those tools locally. The included examples (`hello-world`,
`skillpack-template`, `content-studio`) use only the local surface and run
end-to-end offline.

## How it works

The runner runs the agent on a `LocalRunContext` with `platform_enabled=False`.
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
