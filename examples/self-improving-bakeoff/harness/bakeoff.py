"""Head-to-head: the Puras skill vs the steelman LangGraph agent, over one session
of briefs, with learning curves and cost.

    python -m harness.bakeoff --n 8 --seed 42

Both face the same briefs and the same hidden rulebook; both have memory. We print
each round's score, the running mean (the learning curve), and total cost — and we
show parity honestly where it ties.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from task import generate_briefs  # noqa: E402

RESULTS = ROOT / "results"


def _curve(label: str, scores: list[float]) -> str:
    cum = []
    run = 0.0
    for i, s in enumerate(scores):
        run += s
        cum.append(run / (i + 1))
    cells = "  ".join(f"{s:.2f}" for s in scores)
    means = "  ".join(f"{m:.2f}" for m in cum)
    return f"{label}\n  round score : {cells}\n  running mean: {means}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Picky-Client head-to-head")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--puras-model", default="claude/haiku-4-5")
    ap.add_argument("--lg-model", default="claude-haiku-4-5")
    ap.add_argument("--only", choices=["puras", "langgraph"], default=None)
    args = ap.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY") and Path("/tmp/.akey").exists():
        os.environ["ANTHROPIC_API_KEY"] = Path("/tmp/.akey").read_text().strip()

    briefs = generate_briefs(seed=args.seed, n=args.n)
    out = {"args": vars(args)}
    t0 = time.perf_counter()

    from contestants.shared import usd

    if args.only != "langgraph":
        from contestants.puras_skill.runner import run_puras_session
        ps, pin, pout, pe = run_puras_session(briefs, model=args.puras_model)
        out["puras"] = {"scores": ps, "cost_usd": round(usd(pin, pout), 4), "error": pe,
                        "tokens": [pin, pout], "mean": round(sum(ps) / len(ps), 3)}
    if args.only != "puras":
        from contestants.langgraph_agent import run_langgraph_session
        ls, lin, lout, le = run_langgraph_session(briefs, model=args.lg_model)
        out["langgraph"] = {"scores": ls, "cost_usd": round(usd(lin, lout), 4), "error": le,
                            "tokens": [lin, lout], "mean": round(sum(ls) / len(ls), 3)}

    print("\nPicky-Client head-to-head  (score = fraction of the client's rules satisfied)")
    print("=" * 72)
    if "langgraph" in out:
        print(_curve("LangGraph (steelman, BaseStore memory)", out["langgraph"]["scores"]))
        print(f"  mean {out['langgraph']['mean']}   cost ~${out['langgraph']['cost_usd']}\n")
    if "puras" in out:
        print(_curve("Puras skill (memory + eval)", out["puras"]["scores"]))
        print(f"  mean {out['puras']['mean']}   cost ~${out['puras']['cost_usd']}\n")
    print(f"wall {time.perf_counter()-t0:.1f}s")

    RESULTS.mkdir(exist_ok=True)
    f = RESULTS / f"headtohead-{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"saved → {f}")


if __name__ == "__main__":
    main()
