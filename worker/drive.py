"""Drive backend setup.

The worker's drive is a **local directory** — pure POSIX filesystem, no FUSE.
The Supabase `drive` bucket is the source of truth; this local dir is a
per-worker cache + scratch over it:

- **reads:** a declared input is materialized at job start (`ensure_input_files`);
  any other bucket object is pulled on demand (`ensure_local_drive_file` for
  tool-layer reads, the `drive_pull` tool for raw bash). A local hit never
  touches the network.
- **writes:** the skill writes plain local files. Whatever must outlive the job
  or be served by the API (generated media, declared outputs, files signed for
  an upstream) is pushed to the bucket explicitly (`upload_drive_file`,
  upload-on-sign, end-of-job sync-out). Intermediate scratch stays local-only.

This replaces the old s3fs-fuse mount, whose read-after-write cache races across
the FUSE boundary were a recurring source of "drive file not found" failures.
Local dev already ran this way (`LOCAL_DRIVE_PATH`); prod now matches it.

After `setup_drive()` returns, `get_drive_root()` gives the filesystem path where
`<workspace_id>/...` is the workspace's area. Workspaces are the tenancy
boundary — every job runs against the caller's workspace drive, regardless of
which skillpack provides the code.
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog

from .config import get_settings

log = structlog.get_logger()
_drive_root: Path | None = None


def _slugify(name: str) -> str:
    """Filesystem-safe folder slug from a skill name.

    `social-photo` → `social-photo`, `@inline` → `inline`, `Report.md` →
    `report-md`. Never empty — falls back to `skill`."""
    s = re.sub(r"[^a-z0-9_-]+", "-", str(name or "").strip().lower()).strip("-_")
    return s or "skill"


def resolve_output_dir(workspace_id: str, skill_name: str, job_id) -> str:
    """The per-run deliverables folder: `<skill-slug>/<jobshort>` under the
    workspace drive root, returned as a drive-relative path (no leading `drive/`).

    Every run gets one human-browsable folder grouped by skill, so the drive root
    holds one folder per skill rather than a flat pile of `media/`, `screenshots/`,
    `_jobs/…` scratch. `jobshort` is the first segment of the job UUID — already
    unique in practice; on the astronomically rare prefix collision we reserve the
    next free `…-1`, `…-2`, … instead of commingling two runs. The folder is
    created (reserved) here so the suffix is stable across the run."""
    slug = _slugify(skill_name)
    short = str(job_id).split("-", 1)[0] or str(job_id)[:8] or "run"
    root = workspace_drive(workspace_id)
    base = f"{slug}/{short}"
    rel = base
    n = 1
    while (root / rel).exists():
        rel = f"{base}-{n}"
        n += 1
    (root / rel).mkdir(parents=True, exist_ok=True)
    return rel


def setup_drive() -> Path:
    """Call once at worker startup. Returns the local drive root path.

    Uses `LOCAL_DRIVE_PATH` when set (dev points it at a host dir); otherwise
    falls back to `DRIVE_MOUNT_PATH` (a plain writable dir in the prod container
    now that nothing is mounted there). The bucket — not this dir — is the
    durable store, so the dir may be ephemeral container disk or a volume.
    """
    global _drive_root
    s = get_settings()
    root = Path(s.local_drive_path or s.drive_mount_path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    log.info("drive_local_root", path=str(root))
    _drive_root = root
    return root


def get_drive_root() -> Path:
    if _drive_root is None:
        raise RuntimeError("drive not set up — call setup_drive() first")
    return _drive_root


def workspace_drive(workspace_id: str) -> Path:
    """Get (and mkdir) the local drive path for a workspace."""
    p = get_drive_root() / workspace_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def teardown_drive() -> None:
    """Nothing to unmount in local-dir mode. Kept for call-site symmetry."""
    return
