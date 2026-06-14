"""Run a user-supplied Python function in a subprocess with a timeout.

The function is identified by a manifest entry like
    entrypoint = "functions.video_utils:main"
which we split into module_path + func_name. The deployment root is on
PYTHONPATH so `lib.colors` etc. import cleanly.

cwd = job workdir, so the function sees the same drive/ folder the agent uses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import get_settings, service_token
from .proc_limits import child_preexec

RUNNER_SOURCE = '''
import importlib
import json
import sys
import traceback


def main():
    module_name = sys.argv[1]
    func_name = sys.argv[2]
    inputs = json.loads(sys.stdin.read() or "{}")
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, func_name)
        out = fn(**inputs) if isinstance(inputs, dict) else fn(inputs)
        sys.stdout.write(json.dumps({"ok": True, "result": out}, default=str))
    except Exception as e:
        sys.stdout.write(json.dumps({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }))


if __name__ == "__main__":
    main()
'''


SDK_PATH = str(Path(__file__).parent / "sdk")


def run_function(
    entrypoint: str,
    inputs: dict,
    workdir: Path,
    deployment_root: Path,
    python_exe: str,
    secrets: dict[str, str] | None = None,
    workspace_id: str | None = None,
    job_id: str | None = None,
) -> dict:
    """entrypoint format: 'module.path:func'.

    `secrets` is a {NAME: VALUE} dict from the skillpack's skillpack_secrets
    table; each entry becomes an env var visible to the function subprocess.

    workspace_id / job_id are injected as PURAS_WORKSPACE_ID / PURAS_JOB_ID
    so the bundled `puras` SDK can attribute media calls back to the caller's
    workspace (drive + billing both live at workspace level).
    """
    module_name, _, func_name = entrypoint.partition(":")
    if not module_name or not func_name:
        return {"ok": False, "error": f"invalid entrypoint format: {entrypoint!r}"}

    s = get_settings()
    runner_dir = Path(tempfile.mkdtemp(prefix="fn-runner-"))
    runner_path = runner_dir / "runner.py"
    runner_path.write_text(RUNNER_SOURCE)

    puras_env: dict[str, str] = {
        "PURAS_API_BASE": s.api_base,
        "PURAS_SERVICE_TOKEN": service_token(),
    }
    if workspace_id:
        puras_env["PURAS_WORKSPACE_ID"] = workspace_id
    if job_id:
        puras_env["PURAS_JOB_ID"] = job_id

    # Allowlisted base env (P1-5): the function gets safe system/runtime vars,
    # the skillpack's OWN secrets, and the platform-injected PURAS_* it needs —
    # but NOT the worker's platform secrets (DATABASE_URL, Supabase keys, provider
    # keys, …). See proc_env.safe_base_env.
    from .proc_env import safe_base_env

    env = {
        **safe_base_env(s.skill_env_passthrough_list),
        **(secrets or {}),  # skillpack secrets override worker env if names collide
        **puras_env,        # platform-injected; always wins
        "PYTHONUNBUFFERED": "1",
        # Deployment root + workdir + bundled puras SDK on PYTHONPATH
        "PYTHONPATH": (
            f"{deployment_root}:{workdir}:{SDK_PATH}:" + os.environ.get("PYTHONPATH", "")
        ),
    }
    try:
        # Heavy deterministic tools (stitch, auto-caption burn) fork ffmpeg from
        # here; mark this child the kernel's OOM victim so an over-budget encode
        # dies instead of the worker process (see proc_limits.child_preexec).
        proc = subprocess.run(
            [python_exe, str(runner_path), module_name, func_name],
            input=json.dumps(inputs),
            capture_output=True,
            text=True,
            timeout=s.function_timeout_seconds,
            env=env,
            cwd=str(workdir),
            preexec_fn=child_preexec(),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"function exceeded {s.function_timeout_seconds}s timeout"}

    if proc.returncode != 0:
        return {
            "ok": False,
            "error": f"runner exit {proc.returncode}",
            "stderr": proc.stderr[-4000:],
        }
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "runner produced non-json output",
            "stdout": proc.stdout[-4000:],
        }
