# puras

The Python SDK + CLI for [Puras](https://puras.co) — deploy long-running,
multi-step agentic skills and call them from anywhere.

```bash
pip install puras        # or: uv tool install puras  /  uvx puras
```

You get two things in one package:

- **`import puras`** — the runtime SDK your deployed skills use
  (`media.generate_image/video/audio`, `media.transcribe`, `secret`,
  `load_path`, `subagent.run`).
- **`puras`** — a CLI to scaffold, deploy, run, and tail skills from your
  terminal and CI.

## Quickstart

```bash
puras login                       # paste a workspace API key (created in the dashboard)
puras init --name "My Skillpack"  # creates the skillpack + writes puras.yaml + scaffolds my-skill/
puras deploy                      # zips ./ and pushes an active deployment
puras run my-skill -i prompt="hi" # submit a job and wait for the result
```

Want a worked example instead of a blank slate? `puras init --template
hello-world` scaffolds the full
[hello-world](https://github.com/PurasAI/hello-world) pack — an agentic
skill, a deterministic skill, a custom tool, and two subagents.

### Auth

- Interactive: `puras login` stores your key in `~/.puras/config.json`.
- CI: set `PURAS_API_KEY` (and optionally `PURAS_API_BASE`) in the environment —
  no `login` step, no browser.

```bash
PURAS_API_KEY=puras_live_… puras deploy --skillpack <id>
```

## Commands

| Command | What it does |
|---|---|
| `puras login` / `logout` / `whoami` | manage stored credentials |
| `puras init` | create a skillpack, write `puras.yaml`, scaffold a starter (`--template hello-world` for the full example) |
| `puras skillpacks` | list your skillpacks |
| `puras deploy [path]` | bundle a dir and push a deployment (`--no-activate`) |
| `puras deployments` | list deployments for the current skillpack |
| `puras activate <version\|id>` | make a deployment the active one |
| `puras run <skill> -i k=v` | submit a job and wait (`--async`, `--json`) |
| `puras logs <job_id>` | stream a job's events until it finishes |
| `puras feedback <job_id> --up\|--down` | rate a result (👍/👎 + `--comment`) so you can see which skills run well |
| `puras secrets set/ls/rm` | manage skillpack secrets |
| `puras pull` | download the active bundle |

The skillpack is resolved from `--skillpack <id|slug>`, else the nearest `puras.yaml`.
The CLI is a convenience over the public HTTP API (`POST /v1/jobs`,
`/v1/skillpacks/{id}/deployments`, …) — anything it does, you can do with curl
and a Bearer key.
