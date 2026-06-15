"""submit_copy — grade the copy against the client's hidden rulebook.

Returns the score + which rules broke (the agent's only window into the rulebook).
Records only the FIRST submission per round as the authoritative score, so the
round's score reflects what the agent knew before submitting — the cross-round
learning signal. The harness reads this state after the run; it never trusts the
agent's self-report.
"""

from __future__ import annotations

import json
from pathlib import Path

from task import Brief, grade


def run(copy: str) -> dict:
    spec = json.loads(Path("_inputs.json").read_text())
    brief = Brief(product=spec["product"], fact=spec["fact"], number=spec["number"])
    score, failed = grade(copy or "", brief)

    sd = Path(spec["session_dir"])
    sd.mkdir(parents=True, exist_ok=True)
    state_file = sd / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"rounds": {}}
    ri = str(spec["round_index"])
    if ri not in state["rounds"]:  # first submission this round is the scored one
        state["rounds"][ri] = {"score": score, "copy": copy}
    state_file.write_text(json.dumps(state))

    return {"score": round(score, 3), "broken_rules": failed, "perfect": not failed}
