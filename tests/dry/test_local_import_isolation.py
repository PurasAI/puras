"""Proof that the OFFLINE runner path is dependency-light (open-core PR 3b).

`puras run --local` must work from `pip install puras[local]` — a thin install
that does NOT carry the hosted DB/storage stack (sqlalchemy / asyncpg / storage3
/ posthog) or the openai SDK. This test spawns a fresh interpreter that BLOCKS
those modules at import, then imports the entire `worker.local_run` import graph.
If any module on the offline path pulls a blocked dep at import time, the
subprocess fails — catching a regression the moment someone adds a top-level
heavy import to a module the local runner touches.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# worker package root = .../worker (this file is worker/tests/dry/<f>.py)
_WORKER_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = r"""
import sys

BLOCKED = {"sqlalchemy", "asyncpg", "storage3", "posthog", "openai"}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"blocked for the local-runner import test: {name}")
        return None


sys.meta_path.insert(0, _Blocker())

# The offline entrypoint and everything run_local imports at module load.
import worker.local_run            # noqa: F401
import worker.agent_runner         # noqa: F401  (the crown-jewel loop)
import worker.media_local          # noqa: F401  (local Fal media; fal_client stays lazy)
import worker.media_verbs          # noqa: F401  (pure verb→model resolution)
import worker.media_registry       # noqa: F401  (pure media model table)
import worker.web_local            # noqa: F401  (local web fetch/search; httpx/anthropic ok)
import worker.deployment           # noqa: F401
import worker.drive                # noqa: F401
import worker.manifest             # noqa: F401
import worker.run_context          # noqa: F401
import worker.skill_loader         # noqa: F401
import worker.workdir              # noqa: F401
import worker.config               # noqa: F401
import worker.providers            # noqa: F401  (anthropic ok; openai must stay lazy)
import worker.prompt_cache         # noqa: F401  (made light)
import worker.storage              # noqa: F401  (made light)
import worker.eval_runner          # noqa: F401  (offline grading; made light)
import worker.eval_local           # noqa: F401  (puras eval --local)
import worker.local_server         # noqa: F401  (puras serve — stdlib http only)

# And the blocker really is active (guards against a no-op test).
try:
    import sqlalchemy  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    raise SystemExit("blocker inactive: sqlalchemy imported")

print("LOCAL_IMPORT_OK")
"""


def test_offline_import_path_needs_no_db_or_storage_stack():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=str(_WORKER_ROOT),
        env={"PYTHONPATH": str(_WORKER_ROOT), "PATH": ""},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "offline import path pulled a blocked heavy dep:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "LOCAL_IMPORT_OK" in proc.stdout
