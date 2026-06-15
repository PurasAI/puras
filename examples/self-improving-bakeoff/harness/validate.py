"""Multi-seed validation — is the gap real or seed noise?

Runs both contestants over several seeds (each a fresh session with its own briefs
and memory), same model, same instructions, and reports mean-of-means ± stdev,
how often Puras ≥ LangGraph, and cost computed on ONE shared pricing basis.

    python -m harness.validate --seeds 5 --n 6

This is the credibility gate before any public claim: a single seed proves nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from task import generate_briefs  # noqa: E402
from contestants.shared import usd  # noqa: E402

RESULTS = ROOT / "results"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Multi-seed validation")
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds")
    ap.add_argument("--n", type=int, default=6, help="rounds per session")
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--puras-model", default="claude/haiku-4-5")
    ap.add_argument("--lg-model", default="claude-haiku-4-5")
    args = ap.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY") and Path("/tmp/.akey").exists():
        os.environ["ANTHROPIC_API_KEY"] = Path("/tmp/.akey").read_text().strip()

    from contestants.puras_skill.runner import run_puras_session
    from contestants.langgraph_agent import run_langgraph_session

    seeds = [args.seed0 + i for i in range(args.seeds)]
    rows = []
    p_means, l_means = [], []
    p_cost = l_cost = 0.0
    p_wins = 0

    for sd in seeds:
        briefs = generate_briefs(seed=sd, n=args.n)
        ps, pin, pout, pe = run_puras_session(briefs, model=args.puras_model)
        ls, lin, lout, le = run_langgraph_session(briefs, model=args.lg_model)
        pm, lm = sum(ps) / len(ps), sum(ls) / len(ls)
        pc, lc = usd(pin, pout), usd(lin, lout)
        p_means.append(pm); l_means.append(lm)
        p_cost += pc; l_cost += lc
        p_wins += pm >= lm
        rows.append((sd, pm, lm, pc, lc, pe or le or ""))
        print(f"  seed {sd}: puras {pm:.3f} (${pc:.3f})  vs  langgraph {lm:.3f} (${lc:.3f})",
              flush=True)

    def agg(xs):
        return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)

    pmean, pstd = agg(p_means)
    lmean, lstd = agg(l_means)

    print("\nMulti-seed validation  (Picky-Client, identical model/prompt/memory)")
    print("=" * 70)
    print(f"{'seed':<6}{'puras':<18}{'langgraph':<18}{'winner':<8}")
    print("-" * 70)
    for sd, pm, lm, pc, lc, err in rows:
        w = "puras" if pm > lm else ("tie" if pm == lm else "langgraph")
        print(f"{sd:<6}{pm:.3f} (${pc:.3f})    {lm:.3f} (${lc:.3f})    {w:<8}{'  ERR' if err else ''}")
    print("-" * 70)
    print(f"Puras     : mean {pmean:.3f} ± {pstd:.3f}   total ${p_cost:.3f}")
    print(f"LangGraph : mean {lmean:.3f} ± {lstd:.3f}   total ${l_cost:.3f}")
    print(f"Puras ≥ LangGraph in {p_wins}/{len(seeds)} seeds")
    if l_cost:
        print(f"Cost ratio (langgraph / puras): {l_cost / p_cost:.1f}x" if p_cost else "")

    RESULTS.mkdir(exist_ok=True)
    f = RESULTS / f"validate-{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps({
        "args": vars(args), "seeds": seeds,
        "puras": {"means": p_means, "mean": pmean, "std": pstd, "cost_usd": round(p_cost, 4)},
        "langgraph": {"means": l_means, "mean": lmean, "std": lstd, "cost_usd": round(l_cost, 4)},
        "puras_wins": p_wins,
    }, indent=2))
    print(f"saved → {f}")


if __name__ == "__main__":
    main()
