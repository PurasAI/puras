"""Run the orchestration bake-off and render it in the terminal.

    python -m harness.run_bakeoff --n 20                 # both contestants, 20 games/cell
    python -m harness.run_bakeoff --players det --n 500  # free, fast deterministic curve
    python -m harness.run_bakeoff --n 10 --model claude/sonnet-4-6

Output: a per-cell summary table and an ASCII robustness curve (win-rate vs
perturbation rate). The deterministic side is pure algorithm (free, fast), so
you can run it at high N; the agent side spends your LLM key, so keep its N
modest. Both face identical (seed, config) games, so the curves are comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.experiment import CONFIGS  # noqa: E402
from engine.scoring import GameResult, play, summarize  # noqa: E402
from engine.game import new_game  # noqa: E402
from engine.experiment import build_game  # noqa: E402
from players.langgraph_player import LangGraphSolver  # noqa: E402

RESULTS_DIR = ROOT / "results"


def _puras_game_subprocess(seed: int, config_id: int, max_guesses: int,
                           model: str | None) -> GameResult:
    """Run one agent game in an isolated subprocess (clean env/settings each time,
    so games can run concurrently). Returns a GameResult parsed from its JSON."""
    cmd = [sys.executable, "-m", "harness.one_game", str(seed), str(config_id),
           str(max_guesses), model or ""]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          env={**os.environ})
    line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith("{")), None)
    if not line:
        return GameResult("puras-agent", "", seed, CONFIGS[config_id]["rate"],
                          False, 0, 0, config_id != 0,
                          error=f"subprocess produced no result: {proc.stderr[-300:]}")
    d = json.loads(line)
    return GameResult(
        player="puras-agent", secret="", seed=seed, rate=d["rate"], won=d["won"],
        guesses_used=d["guesses_used"], attempts=d["attempts"],
        perturbed=config_id != 0, elapsed_s=d.get("elapsed_s", 0.0),
        cost_micros=d.get("cost_micros", 0), error=d.get("error"),
    )


def _det_game(seed: int, config_id: int, max_guesses: int) -> GameResult:
    game = build_game(seed, config_id, max_guesses=max_guesses)
    r = play(game, LangGraphSolver(), seed=seed, rate=CONFIGS[config_id]["rate"])
    r.player = "langgraph"
    return r


def _bar(frac: float, width: int = 30) -> str:
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


def _curve(by_player: dict[str, dict[int, dict]]) -> str:
    """ASCII robustness curve: win-rate per config, one row per (player, config)."""
    lines = ["", "Robustness curve — win-rate vs perturbation rate", "=" * 64]
    for player, per_cfg in by_player.items():
        lines.append(f"\n{player}")
        for cfg in CONFIGS:
            s = per_cfg.get(cfg["id"])
            if not s or not s.get("n"):
                continue
            wr = s["win_rate"]
            label = f'  rate {cfg["rate"]:.2f} {cfg["label"]:<9}'
            lines.append(f"{label} |{_bar(wr)}| {wr*100:5.1f}%  (n={s['n']})")
    return "\n".join(lines)


def _table(rows: list[tuple]) -> str:
    head = ("player", "config", "rate", "n", "win%", "g/win", "att", "err", "cost$")
    widths = [11, 9, 5, 4, 6, 6, 5, 4, 8]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*head), "-" * 70]
    for r in rows:
        out.append(fmt.format(*r))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Orchestration bake-off: Puras agent vs LangGraph")
    ap.add_argument("--n", type=int, default=10, help="games per (player,config) cell")
    ap.add_argument("--det-n", type=int, default=None,
                    help="games per cell for the deterministic side (defaults to --n)")
    ap.add_argument("--players", choices=["det", "puras", "both"], default="both")
    ap.add_argument("--configs", type=str, default=None,
                    help="comma-separated config ids (default: all)")
    ap.add_argument("--max-guesses", type=int, default=6)
    ap.add_argument("--model", type=str, default=None, help="override the agent model")
    ap.add_argument("--workers", type=int, default=4, help="concurrent agent games")
    ap.add_argument("--seed0", type=int, default=1000, help="first seed (games use seed0..seed0+n)")
    args = ap.parse_args(argv)

    cfg_ids = [int(x) for x in args.configs.split(",")] if args.configs else [c["id"] for c in CONFIGS]
    det_n = args.det_n if args.det_n is not None else args.n
    run_det = args.players in ("det", "both")
    run_puras = args.players in ("puras", "both")

    if run_puras and not os.environ.get("ANTHROPIC_API_KEY"):
        keyfile = Path("/tmp/.akey")
        if keyfile.exists():
            os.environ["ANTHROPIC_API_KEY"] = keyfile.read_text().strip()
        else:
            print("! no ANTHROPIC_API_KEY — running deterministic side only")
            run_puras = False

    by_player: dict[str, dict[int, dict]] = {}
    all_results: list[GameResult] = []
    rows: list[tuple] = []
    t_start = time.perf_counter()

    for cid in cfg_ids:
        if run_det:
            res = [_det_game(args.seed0 + i, cid, args.max_guesses) for i in range(det_n)]
            all_results += res
            by_player.setdefault("langgraph", {})[cid] = summarize(res)
        if run_puras:
            seeds = [args.seed0 + i for i in range(args.n)]

            def _one(seed, cid=cid):
                gr = _puras_game_subprocess(seed, cid, args.max_guesses, args.model)
                print(f"  puras cfg{cid} seed{seed}: "
                      f"{'WON' if gr.won else 'lost'} in {gr.guesses_used} "
                      f"(${gr.cost_micros/1e6:.3f}){' ERR' if gr.error else ''}", flush=True)
                return gr

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                res = list(pool.map(_one, seeds))
            all_results += res
            by_player.setdefault("puras-agent", {})[cid] = summarize(res)

    # --- tables ---
    for player, per_cfg in by_player.items():
        for cid, s in sorted(per_cfg.items()):
            if not s.get("n"):
                continue
            cfg = CONFIGS[cid]
            rows.append((
                player, cfg["label"], f'{cfg["rate"]:.2f}', s["n"],
                f'{s["win_rate"]*100:.0f}', s["avg_guesses_on_win"] or "-",
                s["avg_attempts"], s["errors"], f'{s["total_cost_usd"]:.3f}',
            ))

    print("\n" + _table(rows))
    print(_curve(by_player))
    print(f"\nTotal wall time: {time.perf_counter()-t_start:.1f}s")

    # --- persist ---
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "args": vars(args),
        "summary": {p: {str(k): v for k, v in d.items()} for p, d in by_player.items()},
        "games": [
            {"player": r.player, "seed": r.seed, "rate": r.rate, "won": r.won,
             "guesses_used": r.guesses_used, "attempts": r.attempts,
             "cost_usd": round(r.cost_micros / 1e6, 4), "error": r.error}
            for r in all_results
        ],
    }
    out = RESULTS_DIR / f"bakeoff-{stamp}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Saved results → {out}")


if __name__ == "__main__":
    main()
