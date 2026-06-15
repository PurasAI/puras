"""CLI for the eval-reward prompt optimizer.

    python -m optimizer --iterations 4 --n-val 6 --target 1.0

Prints the trajectory (mean score per iteration, accepted/rejected, top failures)
and the best prompt found. Saves the run to results/."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimizer import optimize  # noqa: E402

RESULTS = ROOT / "results"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Eval-reward prompt optimizer")
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--n-val", type=int, default=6)
    ap.add_argument("--val-seed", type=int, default=7)
    ap.add_argument("--target", type=float, default=1.0)
    ap.add_argument("--contestant-model", default="claude/haiku-4-5")
    ap.add_argument("--optimizer-model", default="claude-sonnet-4-6")
    args = ap.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY") and Path("/tmp/.akey").exists():
        os.environ["ANTHROPIC_API_KEY"] = Path("/tmp/.akey").read_text().strip()

    def on_step(s):
        tag = "ACCEPT" if s.accepted else "reject"
        tops = ", ".join(f"{k}×{n}" for k, n in s.top_failures[:3]) or "(none)"
        print(f"iter {s.iteration}: mean={s.mean_score:.3f} cost=${s.cost_usd:.3f} "
              f"[{tag}]  top-failures: {tops}", flush=True)

    res = optimize(iterations=args.iterations, n_val=args.n_val, val_seed=args.val_seed,
                   target=args.target, contestant_model=args.contestant_model,
                   optimizer_model=args.optimizer_model, on_step=on_step)

    print(f"\nbaseline mean {res.baseline_mean:.3f}  ->  best mean {res.best_mean:.3f}")
    RESULTS.mkdir(exist_ok=True)
    f = RESULTS / f"optimize-{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps({
        "baseline_mean": res.baseline_mean, "best_mean": res.best_mean,
        "best_instructions": res.best_instructions,
        "trajectory": [{"iteration": s.iteration, "mean": s.mean_score, "cost_usd": s.cost_usd,
                        "accepted": s.accepted, "top_failures": s.top_failures}
                       for s in res.trajectory],
    }, indent=2))
    print(f"saved → {f}")


if __name__ == "__main__":
    main()
