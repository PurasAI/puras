"""Guard the objective grader: a fully-compliant copy scores 1.0, and each
broken rule is detected with its feedback. Runs under pytest or standalone."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task import RULEBOOK, Brief, generate_briefs, grade  # noqa: E402

B = Brief(product="the Nimbus 7 backpack", fact="a rain-proof zip", number="30L")


def test_perfect_copy_scores_one():
    copy = "Pack the Nimbus 7 backpack for 30 days — its warranty has you covered. Ready to roll?"
    score, failed = grade(copy, B)
    assert score == 1.0, (score, failed)
    assert failed == []


def test_each_rule_can_fail_independently():
    # Missing warranty + no question + uses 'best' + no product + no number + too long.
    bad = "best " * 40
    score, failed = grade(bad, B)
    assert score < 1.0
    fb = " ".join(failed).lower()
    assert "warranty" in fb and "question" in fb and "best" in fb and "140" in fb


def test_specific_violations():
    # Satisfies everything except the banned superlative.
    copy = "The Nimbus 7 backpack is the best — 30L, with warranty. Want one?"
    score, failed = grade(copy, B)
    assert any("best" in f for f in failed)
    assert not any("warranty" in f for f in failed)


def test_briefs_reproducible():
    a = generate_briefs(7, 5)
    b = generate_briefs(7, 5)
    assert a == b and len(a) == 5
    assert generate_briefs(8, 5) != a  # different seed → different order


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
