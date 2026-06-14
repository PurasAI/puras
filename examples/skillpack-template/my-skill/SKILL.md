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
