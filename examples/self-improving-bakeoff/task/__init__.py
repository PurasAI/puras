"""The "Picky Client" task: a hidden, consistent style rulebook a contestant must
learn (from eval feedback + memory) to score well. Objective, deterministic
grading — no LLM-judge noise."""

from .client import Brief, Rule, RULEBOOK, generate_briefs, grade

__all__ = ["Brief", "Rule", "RULEBOOK", "generate_briefs", "grade"]
