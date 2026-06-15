"""Eval-reward hill-climber over the Puras skill's prompt.

reward(prompt) = mean grader score over a fixed validation batch (run through the
real skill, with memory). Each iteration: run the skill, collect the rules it most
often broke (grader feedback — the legitimate learning signal), ask an optimizer
LLM to rewrite the prompt to fix them, accept the new prompt only if its reward
improves. We log the trajectory (the score climbing past the LangGraph baseline)
and, crucially, *what changed* — that's the signal for Puras's roadmap.

This is LLM-as-optimizer (APE/OPRO/DSPy-style). It lives here, in the experiment,
not inside Puras. The optimizer model can differ from the contestant model — it's
the experimenter's tool, not a contestant.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

from task import generate_briefs
from contestants.shared import BASE_INSTRUCTIONS
from contestants.puras_skill.runner import run_puras_session


@dataclass
class Step:
    iteration: int
    mean_score: float
    cost_usd: float
    accepted: bool
    top_failures: list[tuple[str, int]]
    instructions: str = ""


@dataclass
class Result:
    baseline_mean: float
    best_mean: float
    best_instructions: str
    trajectory: list[Step] = field(default_factory=list)


_OPTIMIZER_SYS = """\
You optimize the system prompt of a copywriting agent. The agent writes copy that a \
client grades against a hidden, consistent style guide. You're given the agent's \
CURRENT prompt, its mean score in [0,1] over a validation batch, and the client \
rules it most often BROKE (verbatim grader feedback). Rewrite the prompt so the \
agent reliably satisfies those rules: you MAY bake the discovered rules in as \
explicit, concrete guidance. Keep the agent's workflow intact — it must still call \
recall first, write copy, call submit_copy exactly once, then remember lessons. \
Output ONLY the new prompt text, nothing else."""


def _reward(instructions: str, briefs, contestant_model: str):
    from contestants.shared import usd
    scores, tin, tout, err, rounds = run_puras_session(
        briefs, model=contestant_model, instructions=instructions, details=True)
    mean = sum(scores) / len(scores) if scores else 0.0
    fails = collections.Counter()
    for r in rounds:
        for fb in r.get("broken_rules", []):
            fails[fb] += 1
    return mean, usd(tin, tout), fails, err


def _propose(current: str, mean: float, fails, optimizer_model: str) -> str:
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model=optimizer_model, temperature=0.4, max_tokens=1500)
    fail_lines = "\n".join(f"- {fb} (broken {n}×)" for fb, n in fails.most_common()) or "- (none)"
    msg = (f"CURRENT PROMPT:\n\"\"\"\n{current}\n\"\"\"\n\n"
           f"MEAN SCORE: {mean:.2f}/1.0\n\nMOST-BROKEN RULES (grader feedback):\n{fail_lines}\n\n"
           "Rewrite the prompt now.")
    resp = llm.invoke([("system", _OPTIMIZER_SYS), ("user", msg)])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def optimize(
    *,
    iterations: int = 4,
    n_val: int = 6,
    val_seed: int = 7,
    contestant_model: str = "claude/haiku-4-5",
    optimizer_model: str = "claude-sonnet-4-6",
    target: float | None = None,
    on_step=None,
) -> Result:
    briefs = generate_briefs(seed=val_seed, n=n_val)

    best = BASE_INSTRUCTIONS
    base_mean, base_cost, fails, _ = _reward(best, briefs, contestant_model)
    best_mean = base_mean
    res = Result(baseline_mean=base_mean, best_mean=base_mean, best_instructions=best)
    res.trajectory.append(Step(0, base_mean, base_cost, True, fails.most_common(), best))
    if on_step:
        on_step(res.trajectory[-1])

    for it in range(1, iterations + 1):
        if target is not None and best_mean >= target:
            break
        candidate = _propose(best, best_mean, fails, optimizer_model)
        mean, cost, cfails, _ = _reward(candidate, briefs, contestant_model)
        accepted = mean > best_mean
        if accepted:
            best, best_mean, fails = candidate, mean, cfails
            res.best_instructions, res.best_mean = best, best_mean
        res.trajectory.append(Step(it, mean, cost, accepted, cfails.most_common(), candidate))
        if on_step:
            on_step(res.trajectory[-1])

    return res
