"""Eval-reward prompt optimizer — an EXTERNAL R&D instrument (not shipped in Puras).

Hill-climbs the Puras skill's prompt using grader scores as reward, to (a) make the
head-to-head a real fight and (b) reveal what helps the skill — feeding Puras's
roadmap and tuning."""

from .optimize import optimize

__all__ = ["optimize"]
