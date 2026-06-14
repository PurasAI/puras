"""Per-job working directory.

Layout:
    <workdir_root>/<job_id>/
        SKILL.md, skill.yaml, references/, scripts/, requirements.txt, …
            # per-entry symlinks into the skill bundle (read-only)
        _inputs.json
            # serialized job inputs
        drive/
            # symlink → <drive_root>/<workspace_id>/

The agent's bash tool, the function runner, and the skill loader all use
this workdir as their cwd. Instead of mounting the skill bundle under a
nested directory, each top-level entry of the skill bundle is symlinked
straight into the workdir root, so the agent can `cat SKILL.md` or
`cat references/02-best-practices.md` without any path prefix.

`drive/` is the only place a skill should write things meant to outlast
the job — it's the workspace's persistent drive. The active workspace is
always the caller's; cross-workspace skillpack invocations (where the
caller runs a public skillpack owned by someone else) still bill and
persist to the caller's workspace.

`create_workdir` runs before the skill is resolved (it only needs the
job id + workspace), so it doesn't mount any skill files yet;
`attach_skill` adds the per-entry symlinks once the skill is loaded.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import get_settings
from .drive import workspace_drive

# Names we never overwrite when mounting a skill bundle. `drive` collides
# with the workspace-drive symlink we create in `create_workdir`; entries
# starting with `_` are reserved for worker-managed slots (e.g.
# `_inputs.json`).
_RESERVED_NAMES = {"drive"}


def create_workdir(job_id: str, workspace_id: str, inputs: dict) -> Path:
    s = get_settings()
    root = Path(s.workdir_root) / job_id
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "_inputs.json").write_text(json.dumps(inputs, indent=2, default=str))

    drive_link = root / "drive"
    target = workspace_drive(workspace_id)
    drive_link.symlink_to(target, target_is_directory=True)

    return root


def attach_skill(workdir: Path, skill_root: Path) -> None:
    """Symlink each top-level entry of the skill bundle into the workdir root.

    Called after `load_skill` so the agent's bash cwd (= workdir) exposes
    SKILL.md / references/ / scripts/ directly at relative paths. Idempotent:
    re-attaching the same skill (or a different one) replaces any prior
    symlinks of the same name.

    Skips collisions with reserved worker slots:
      - `drive` (the workspace-drive symlink)
      - any entry whose name starts with `_` (e.g. `_inputs.json`)
    """
    for entry in skill_root.iterdir():
        name = entry.name
        if name in _RESERVED_NAMES or name.startswith("_"):
            continue
        link = workdir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(entry, target_is_directory=entry.is_dir())


def cleanup_workdir(job_id: str) -> None:
    s = get_settings()
    root = Path(s.workdir_root) / job_id
    if root.exists():
        # `drive/` is a symlink into the real workspace drive — rmtree won't
        # follow it, so persistent data stays put. The per-entry skill
        # symlinks are also followed-as-symlinks and just removed.
        shutil.rmtree(root, ignore_errors=True)
