# Self-Improving Skill Bake-off — Puras vs a steelman LangChain/LangGraph

The public (HN/Reddit) experiment. **Claim we want to earn, not fake:** building a
self-improving, eval-gated, memory-backed agent for your business process is
*better, cheaper, and far less work* with a Puras skill than with LangGraph +
LangMem + LangSmith — even when the LangGraph side is built to win.

> Status: DESIGN + building. Branch `claude/agentic-orchestration-demo-s94k29`.
> Supersedes the Wordle `orchestration-bakeoff` (that proved agentic > deterministic;
> this proves the real, harder, honest point).

## 0. Why this, and the honest footing

Research finding (see chat): LangChain/LangGraph already have all three pillars —
long-term memory (LangGraph `BaseStore` + LangMem's semantic/episodic/**procedural**
memory), learning (LangMem prompt/behavior optimization), and eval-driven
self-improvement (LangSmith). So a naive "we learn, they don't" claim is false and
HN would shred it. We therefore **steelman LangGraph** (give it memory, build it to
win) and find the dimensions Puras *honestly* wins:

1. **Same outcome, a fraction of the code + zero infra** (a skill folder vs
   StateGraph + a Store backend + LangMem + LangSmith + glue).
2. **Cheaper by default** (built-in model `routing`: cheap model, escalate only on
   eval fail; eval-gating avoids wasted calls).
3. **Cheaper/faster to *get good*:** an external optimizer (part of THIS
   experiment harness, not shipped inside Puras) iterates the skill's prompt from
   eval scores until it beats the hand-built LangGraph. The point isn't to ship an
   optimizer in Puras — it's that this script is an **R&D instrument**: by pitting
   Puras against a steelman and hill-climbing, it tells us *which Puras features to
   build and how to tune the existing ones* (memory, routing, eval-gating).

We show the learning curves side by side **including where they tie** — admitting
parity is what makes the rest credible.

## 1. The task — "Picky Client" (learn a hidden business rulebook)

A repetitive content task standing in for "your business process". The client has
a **hidden but consistent style guide** (a rulebook). Each round the agent writes a
short deliverable for a brief; an **objective grader** scores it in [0,1] and says
which rules failed. The only way to score well is to *learn the rulebook* — from
eval feedback (across rounds, via memory) and from a better base prompt (via the
optimizer).

- Deterministic rules (objective, free grader — no LLM-judge noise), e.g. length
  ceiling, must mention the product, must include "warranty", must avoid "best",
  must end with a question, must contain a number. The rulebook is *fixed* for a
  run; instances differ (different products/facts).
- "Business process" framing: the rulebook == a brand/client style guide the agent
  must internalize — exactly Puras's real content-skill use case.

## 2. Contestants (both LLM agents with memory — fair)

- **LangGraph (steelman):** a real `create_react_agent` + a long-term `BaseStore`
  (InMemoryStore for the run). After each round it stores the rules it inferred
  from eval feedback and retrieves them next round. Same model (Haiku), same single
  `write_copy` tool + grader, same instructions. Built to win.
- **Puras skill:** `picky-copywriter` skillpack — the agent loop + the same tool;
  memory via `memory_put/search`. On the local runner we back memory with a small
  persistent tool (offline, fast); the **final head-to-head is re-validated on
  Puras Cloud** with the real workspace brain.

Both start *blind* to the rulebook. Fairness mirrors the Wordle design's §3:
shared task+grader module, identical model/tool, seeded instance order.

## 3. The RL-like optimizer (an R&D instrument, not a Puras feature)

This lives in the experiment harness. We are **not** writing an optimizer into
Puras; we use this loop to (a) push the Puras skill up so the head-to-head is a
real fight, and (b) observe *what* makes it improve — which is the signal for
Puras's roadmap and feature tuning. Eval score = reward; we hill-climb the
**skill's prompt** (`SKILL.md`):

```
score(prompt) = mean grader score over a validation batch of briefs
loop:
  run skill with current prompt → collect failing rounds + eval feedback
  optimizer-LLM proposes a revised prompt from {prompt, failures, feedback}
  if score(new) > score(best): best = new        # accept improvement
stop when best beats the LangGraph score (or K iters / budget)
```

This is LLM-as-optimizer (APE/OPRO/DSPy-style) driven by Puras eval scores. Output:
a trajectory (e.g. 60 → 72 → 85) crossing the LangGraph line (80), plus the
optimized prompt. **Fairness note:** LangMem can self-optimize too; the honest
point is Puras ships this loop *built-in* — you don't assemble it.

## 4. Metrics (all reproducible, saved to results/)

- **Quality:** grader score per round; learning curve per contestant (likely a tie
  → shown honestly).
- **Optimizer trajectory:** Puras score vs iteration, vs the LangGraph baseline.
- **Cost:** tokens / USD per round and to-convergence (Puras cheaper by default).
- **Build/ops cost (the headline):** lines of code, # components/services, setup
  steps to reach the same self-improving behavior. Counted honestly from each repo.

## 5. What this script feeds back into Puras (the real point)

This harness is an R&D instrument for Puras, **not** a feature to embed in it.
Each run produces a "findings" report: where Puras lost to the steelman, what the
optimizer had to change to win, and what that implies for the product. Expected
outputs:
- **Feature candidates** — gaps the experiment exposes (e.g. better memory
  retrieval ergonomics, eval-feedback surfaced to the agent, a cheaper default
  routing policy). Logged as roadmap signal, not shipped here.
- **Tuning of existing features** — concrete settings for `routing` /
  eval-gating / memory write-back that measurably helped, fed back into Puras's
  defaults.
- The optimizer + harness stay external — a repeatable lab we re-run as Puras
  changes, to keep proving (and improving) the claim.

## 6. Layout

```
self-improving-bakeoff/
  task/            picky-client rulebook + objective grader + brief generator
  contestants/
    langgraph_agent/   create_react_agent + BaseStore memory (steelman)
    puras_skill/       picky-copywriter skillpack (+ local memory tool)
  optimizer/       eval-reward prompt hill-climber (the RL-like loop + feature proto)
  harness/         head-to-head runner, learning curves, build-cost accounting
  results/         saved runs (gitignored)
```

## 7. Plan

- **P0** task + grader + brief generator (objective, tested). ← building now
- **P1** both contestants on the local runner; blind learning curves.
- **P2** the optimizer loop; push Puras past the LangGraph baseline.
- **P3** re-validate the head-to-head on Puras Cloud (real memory); cost + build-cost
  accounting; write-up.
