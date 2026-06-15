# Self-Improving Skill Bake-off — Puras vs a steelman LangChain/LangGraph

The public experiment: build a self-improving, memory-backed agent for a hidden
business rulebook two ways — a **Puras skill** and a **steelman LangGraph agent**
(given real memory, built to win) — and compare honestly.

> Honest footing: LangChain/LangGraph already have memory (BaseStore + LangMem),
> learning (LangMem procedural memory), and eval-driven improvement (LangSmith).
> So "we learn, they can't" is false. We steelman the LangGraph side and look for
> where Puras *honestly* wins: same outcome with far less code + ops, cheaper by
> default, and a built-in eval→memory loop. See [`DESIGN.md`](./DESIGN.md).

## The task — "Picky Client"

A client enforces a hidden, consistent style guide (length ≤ 140, must mention the
product, must mention the warranty, no "best", must end with a question, must
include a number). Each round the agent writes copy for a brief; an **objective
grader** returns a score in [0,1] and which rules broke. The only way to score
well is to *learn the rulebook* — across rounds via memory, and (for Puras) via the
prompt the optimizer evolves.

## Run

```bash
pip install -e .                                    # Puras runner (repo root)
pip install langgraph langchain langchain-anthropic # the steelman + optimizer
export ANTHROPIC_API_KEY=sk-...

cd examples/self-improving-bakeoff
python -m harness.bakeoff --n 8 --seed 42           # head-to-head, learning curves + cost
python -m optimizer --iterations 4 --target 1.0     # eval-reward prompt optimizer (Puras side)
```

## Early results (haiku, n=8, seed=42 — preliminary, single seed)

| | learning curve (per round) | mean | cost |
|---|---|---|---|
| LangGraph (steelman, BaseStore) | 0.67 → plateau ~0.73–0.83 | 0.73 | ~$0.068 |
| Puras skill (memory) | climbs to repeated 1.0 by round 3 | 0.88 | ~$0.014 |

Both have memory; both get identical instructions. Puras converges higher and runs
cheaper here. **Caveats before any public claim:** single seed (needs multi-seed),
cost bases must be made apples-to-apples, and the LangGraph steelman should be
hardened further. The eval-reward optimizer then pushes the Puras prompt toward a
perfect mean and reports *what* helped — fed back into Puras's roadmap (the
optimizer is an external R&D instrument, not shipped inside Puras).

## Layout

```
task/          the Picky-Client rulebook + objective grader + brief generator
contestants/
  langgraph_agent/   create_react_agent + BaseStore memory (steelman)
  puras_skill/       picky-copywriter skillpack + a per-session memory tool
  shared.py          the round protocol + base instructions (identical for both)
optimizer/     eval-reward prompt hill-climber (external R&D instrument)
harness/       head-to-head runner + learning curves + cost
tests/         the objective grader
```
