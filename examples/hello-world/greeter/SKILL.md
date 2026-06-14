You write a warm greeting for a person. You do it in three small moves — a
custom tool and two subagents — then return a tidy card. Don't write the card or
the poem yourself; that's what the tool and subagents are for.

## Inputs

Your inputs are in the first message:
- `name` — who to greet.
- `style` — the tone for the poem (`friendly`, `formal`, or `playful`).

## Step 1 — Shout the name (custom tool)

Call the `emphasize` tool to get a loud version of the name:

```
emphasize({ "text": <name> })
```

Keep the returned `loud` string — that's your `shout`.

## Step 2 — Ask the poet (a `.md` subagent)

Hand the name and style to the poet subagent, which writes a two-line couplet.
This is a bundle prompt file run as an isolated subagent:

```
run_subagent({
  "target": "references/poet.md",
  "inputs": { "name": <name>, "style": <style> }
})
```

It returns `{ "poem": "<two lines>" }`. Keep the `poem`.

## Step 3 — Assemble the card (a sibling skill as a subagent)

Call the deterministic `formatter` skill in this skillpack — it lays out a
greeting card from the pieces. A bare name targets a skill in this same
skillpack:

```
run_subagent({
  "target": "formatter",
  "inputs": { "name": <name>, "shout": <shout>, "poem": <poem> }
})
```

It returns `{ "card": "<formatted card>" }`.

## Step 4 — Return

Call `set_output` once with:
- `card` — the formatter's card.
- `shout` — the loud name from Step 1.
- `poem` — the couplet from Step 2.

## Guardrails

- Use the tool and subagents above; don't write the card or the poem yourself.
- One call to each. If a subagent returns an error, put what it said into your
  output instead of retrying forever.
