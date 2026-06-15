"""Unit tests for the prompt-optimizer search core (worker.optimizer_core).

Fully offline: a fake ScoringBackend returns scripted ScoreResults and a fake
proposer emits scripted candidates, so we assert the search's selection rules —
successive-halving prune, early-stop on no improvement, budget exhaustion, and
winner selection vs the baseline — with no LLM and no eval runs.
"""

from __future__ import annotations

import asyncio

from worker.optimizer_core import (
    Candidate,
    ScoreResult,
    SearchConfig,
    beats_baseline,
    keep_count,
    run_search,
    select_survivors,
    worst_case_runs,
)


def _seed() -> Candidate:
    return Candidate(system_prompt="base", source="baseline", round_index=0)


class FakeBackend:
    """Scores by a fixed map id->ScoreResult (default = a low baseline score)."""

    def __init__(self, scores: dict[str, ScoreResult], default: ScoreResult):
        self.scores = scores
        self.default = default
        self.calls: list[tuple[str, int]] = []  # (candidate_id, n_cases-ish)

    async def score(self, candidate, *, case_ids, repeat):
        self.calls.append((candidate.id, len(case_ids) if case_ids else -1))
        return self.scores.get(candidate.id, self.default)


def _make_proposer(batches: list[list[Candidate]]):
    """Return a proposer that yields the next scripted batch each round, then []."""
    seq = list(batches)

    async def _propose(*, n, parent, round_index):
        return seq.pop(0) if seq else []

    return _propose


def test_select_survivors_keeps_top_by_score():
    a = Candidate(system_prompt="a", id="a")
    b = Candidate(system_prompt="b", id="b")
    c = Candidate(system_prompt="c", id="c")
    scored = [
        (a, ScoreResult(mean_score=50.0, pass_rate=0.5, n=1)),
        (b, ScoreResult(mean_score=90.0, pass_rate=0.9, n=1)),
        (c, ScoreResult(mean_score=70.0, pass_rate=0.7, n=1)),
    ]
    survivors = select_survivors(scored, keep_count(3, 0.5))  # keep ceil(1.5)=2
    assert [s.id for s in survivors] == ["b", "c"]


def test_beats_baseline_requires_threshold_and_no_passrate_regression():
    base = ScoreResult(mean_score=60.0, pass_rate=0.8, n=4)
    better = ScoreResult(mean_score=75.0, pass_rate=0.85, n=4)
    thin = ScoreResult(mean_score=61.0, pass_rate=0.8, n=4)
    regressed = ScoreResult(mean_score=90.0, pass_rate=0.5, n=4)
    assert beats_baseline(better, base, improvement_threshold=5.0)
    assert not beats_baseline(thin, base, improvement_threshold=5.0)
    assert not beats_baseline(regressed, base, improvement_threshold=5.0)


def test_run_search_picks_improving_winner():
    seed = _seed()
    cand = Candidate(system_prompt="better", id="win")
    backend = FakeBackend(
        scores={
            seed.id: ScoreResult(mean_score=50.0, pass_rate=0.5, n=2),
            "win": ScoreResult(mean_score=90.0, pass_rate=0.9, n=2),
        },
        default=ScoreResult(mean_score=10.0, pass_rate=0.1, n=2),
    )
    proposer = _make_proposer([[cand]])
    config = SearchConfig(max_candidates=1, max_rounds=2, minibatch_size=2, repeat=1)
    res = asyncio.run(
        run_search(backend, seed=seed, propose_fn=proposer, config=config,
                   all_case_ids=["c1", "c2"])
    )
    assert res.winner.id == "win"


def test_run_search_keeps_baseline_when_no_candidate_improves():
    seed = _seed()
    weak = Candidate(system_prompt="weak", id="weak")
    backend = FakeBackend(
        scores={
            seed.id: ScoreResult(mean_score=80.0, pass_rate=0.9, n=2),
            "weak": ScoreResult(mean_score=40.0, pass_rate=0.4, n=2),
        },
        default=ScoreResult(mean_score=40.0, pass_rate=0.4, n=2),
    )
    proposer = _make_proposer([[weak]])
    config = SearchConfig(max_candidates=1, max_rounds=3, minibatch_size=2)
    res = asyncio.run(
        run_search(backend, seed=seed, propose_fn=proposer, config=config,
                   all_case_ids=["c1", "c2"])
    )
    assert res.winner.id == seed.id
    assert res.stop_reason == "no_improvement"


def test_run_search_stops_on_budget_exhaustion():
    seed = _seed()
    cand = Candidate(system_prompt="x", id="x")
    # Each score reports a per-run cost; the baseline alone exceeds a tiny budget.
    backend = FakeBackend(
        scores={},
        default=ScoreResult(mean_score=50.0, pass_rate=0.5, mean_cost_micros=1000, n=5),
    )
    proposer = _make_proposer([[cand]])
    config = SearchConfig(max_candidates=1, max_rounds=3, minibatch_size=2,
                          budget_micros=100)
    res = asyncio.run(
        run_search(backend, seed=seed, propose_fn=proposer, config=config,
                   all_case_ids=["c1", "c2"])
    )
    # Baseline cost (5 * 1000) already blew the 100-micro budget → no rounds ran.
    assert res.stop_reason == "budget_exhausted"
    assert res.winner.id == seed.id


def test_worst_case_runs_is_bounded():
    config = SearchConfig(max_candidates=4, max_rounds=2, minibatch_size=3, repeat=1,
                          keep_fraction=0.5)
    # baseline(10) + 2 rounds * (4*3 minibatch + 2 survivors*10 full) = 10 + 2*32 = 74
    assert worst_case_runs(config, n_cases=10) == 74
