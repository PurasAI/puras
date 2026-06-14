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
```

## References

- Building a skillpack: https://puras.co/docs/building-a-skillpack
- skill.yaml, field by field: https://puras.co/docs/skill-yaml-reference
- Agent runtime tools (`run_subagent`, `set_output`, media verbs):
  https://puras.co/docs/agent-tools-reference
- CLI: https://puras.co/docs/cli-reference
