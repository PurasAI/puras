"""Drive the Puras picky-copywriter skill over a session of briefs.

One job per round (via the local runner), sharing a session_dir so memory persists
across rounds — the offline mirror of a Puras Cloud workspace where the brain
carries lessons between jobs. Reads the engine-truth score the submit_copy tool
recorded; never trusts the agent's self-report.

`instructions` lets the optimizer (P2) swap in a candidate SKILL.md: we copy the
skillpack to a temp dir and overwrite the prompt, so the on-disk skill is untouched.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../self-improving-bakeoff
REPO = ROOT.parents[1]                               # .../puras
SKILLPACK = Path(__file__).resolve().parent          # .../puras_skill


def _ensure_paths() -> None:
    extra = f"{ROOT}:{REPO}"
    cur = os.environ.get("PYTHONPATH", "")
    if extra not in cur:
        os.environ["PYTHONPATH"] = f"{extra}:{cur}" if cur else extra
    for p in (str(REPO), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _skillpack_with_prompt(instructions: str | None) -> tuple[Path, Path | None]:
    """Return (skillpack_dir, tmp_to_cleanup). If instructions are given, copy the
    pack to a temp dir and overwrite SKILL.md with them."""
    if not instructions:
        return SKILLPACK, None
    tmp = Path(tempfile.mkdtemp(prefix="picky-pack-"))
    dst = tmp / "puras_skill"
    shutil.copytree(SKILLPACK, dst)
    md = dst / "picky-copywriter/SKILL.md"
    front = md.read_text().split("---", 2)
    header = f"---{front[1]}---\n" if len(front) >= 3 else ""
    md.write_text(header + "\n" + instructions + _FINISH)
    return dst, tmp


_FINISH = (
    "\n\n## Finishing\n\nWhen you're done, call `set_output` once with the copy you "
    "submitted and the score it received:\n\n```\n"
    'set_output({ "copy": "<your submitted copy>", "score": <the score> })\n```\n'
)


def run_puras_session(briefs, *, model: str | None = None, instructions: str | None = None):
    """Run the skill over briefs. Returns (scores_per_round, cost_micros, error)."""
    _ensure_paths()
    from worker.local_run import run_local

    pack, tmp = _skillpack_with_prompt(instructions)
    sess = Path(tempfile.mkdtemp(prefix="picky-sess-"))
    cost_micros = 0
    error = None
    try:
        for i, b in enumerate(briefs):
            try:
                res = run_local(
                    str(pack),
                    {"product": b.product, "fact": b.fact, "number": b.number,
                     "round_index": i, "session_dir": str(sess)},
                    skill="picky-copywriter", model=model,
                    on_event=lambda *a, **k: None,
                )
                cost_micros += (res.get("usage") or {}).get("cost_micros", 0)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
        state = json.loads((sess / "state.json").read_text()) if (sess / "state.json").exists() else {"rounds": {}}
        scores = [state["rounds"].get(str(i), {}).get("score", 0.0) for i in range(len(briefs))]
    finally:
        shutil.rmtree(sess, ignore_errors=True)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return scores, cost_micros, error
