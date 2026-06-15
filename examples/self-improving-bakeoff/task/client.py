"""The Picky Client's hidden style rulebook + an objective grader.

This stands in for "your business process": a fixed, consistent set of rules the
client silently enforces. A contestant only ever sees *which rules failed* (the
grader's feedback) — never the rulebook itself — so the only way to a high score
is to learn it: across rounds via memory, and via a better base prompt (the
optimizer). Every rule is a deterministic check, so grading is objective and free.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Brief:
    product: str   # the product to write copy for
    fact: str      # a concrete feature to work in
    number: str    # a concrete figure (price, %, count) — lets copy satisfy "has a digit"


@dataclass(frozen=True)
class Rule:
    id: str
    feedback: str                         # what the grader tells a failing contestant
    check: Callable[[str, Brief], bool]   # True == satisfied


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


# The hidden rulebook — fixed for a run. Feedback strings are the ONLY signal a
# contestant gets about a failure; the rule's intent must be inferable from them.
RULEBOOK: list[Rule] = [
    Rule("length", "Copy must be 140 characters or fewer.",
         lambda c, b: len(c.strip()) <= 140),
    Rule("mentions_product", "Copy must mention the product by name.",
         lambda c, b: b.product.lower() in c.lower()),
    Rule("warranty", "Copy must mention the warranty.",
         lambda c, b: _has_word(c, "warranty")),
    Rule("no_superlative", "Copy must not use the word 'best'.",
         lambda c, b: not _has_word(c, "best")),
    Rule("ends_question", "Copy must end with a question.",
         lambda c, b: c.strip().endswith("?")),
    Rule("has_number", "Copy must include a specific number.",
         lambda c, b: any(ch.isdigit() for ch in c)),
]


def grade(copy: str, brief: Brief) -> tuple[float, list[str]]:
    """Score copy in [0,1] = fraction of rules satisfied; return the feedback for
    every failed rule (the contestant's only window into the rulebook)."""
    copy = copy or ""
    failed = [r.feedback for r in RULEBOOK if not r.check(copy, brief)]
    score = (len(RULEBOOK) - len(failed)) / len(RULEBOOK)
    return score, failed


_PRODUCTS = [
    ("the Nimbus 7 backpack", "a rain-proof zip", "30L"),
    ("the Halcyon desk lamp", "a warm-dim mode", "1800K"),
    ("the Trailfin water bottle", "a leak-proof cap", "750ml"),
    ("the Aera standing desk", "a one-tap height memory", "120cm"),
    ("the Lumen e-reader", "a glare-free screen", "300ppi"),
    ("the Cove bluetooth speaker", "a 20-hour battery", "20h"),
    ("the Pace running shoe", "a carbon midsole", "200g"),
    ("the Vela office chair", "lumbar support", "5yr"),
    ("the Onyx mechanical keyboard", "hot-swap switches", "87 keys"),
    ("the Mistral fan", "a whisper mode", "25dB"),
]


def generate_briefs(seed: int, n: int) -> list[Brief]:
    """A reproducible batch of briefs. The rulebook is constant; only the briefs
    vary, so learning the rulebook transfers across the whole batch."""
    rng = random.Random(seed)
    pool = list(_PRODUCTS)
    rng.shuffle(pool)
    out = []
    for i in range(n):
        product, fact, number = pool[i % len(pool)]
        out.append(Brief(product=product, fact=fact, number=number))
    return out
