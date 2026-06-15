# Orchestration Bake-off — Puras agent vs LangGraph determinism

A head-to-head, run in the terminal: the same game of Wordle, hundreds of times,
played by two kinds of orchestration.

- **LangGraph (deterministic):** a real `StateGraph` wrapping a steelman entropy
  solver. The flow is drawn up front. On a clean game it's near-optimal.
- **Puras (agentic):** a single skill that plays through one `make_guess` tool
  plus the agent loop — it decides what to do each turn instead of following a
  fixed graph.

A **"mischievous host"** injects unforeseen edge cases (format shifts, noisy
clues, new constraints, temporary lies, silent rule changes). The thesis:
fixing the orchestration up front means not trusting the model to handle the
unexpected — so as the edge-case rate rises, the deterministic side falls off a
cliff while the agent degrades gracefully. The robustness curve is the proof.

See [`DESIGN.md`](./DESIGN.md) for the full design and the fairness guarantees.

## Install

```bash
pip install -e .                       # the Puras runner (from the repo root)
pip install langgraph                  # the deterministic contestant
```

The Wordle word lists (official 2315 answers + 10657 allowed guesses) are
bundled under `engine/data/`, so the engine runs fully offline.

## Run

```bash
cd examples/orchestration-bakeoff

# Free, fast: the deterministic side only, at high N — the baseline cliff.
python -m harness.run_bakeoff --players det --det-n 300

# Both contestants (the agent side spends your ANTHROPIC_API_KEY — keep N small).
ANTHROPIC_API_KEY=sk-... python -m harness.run_bakeoff --n 20

# Stronger agent:
python -m harness.run_bakeoff --players puras --n 20 --model claude/sonnet-4-6
```

Both contestants face the **identical** (seed, config) games, so the curves are
directly comparable. Results are written to `results/` as JSON for reproduction.

## Layout

```
engine/        player-agnostic Wordle engine + the perturbation injector
               (the shared, independent environment; word lists in data/)
players/
  langgraph_player/   deterministic StateGraph solver (the steelman)
  puras_player/       the codebreaker skillpack (agent + make_guess tool)
harness/       runs the grid, renders the table + ASCII robustness curve
tests/         engine truth, fairness invariants, both contestants
```

## Tests

```bash
python tests/test_engine.py
python tests/test_langgraph_player.py
python tests/test_puras_tool.py
```
