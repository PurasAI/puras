"""Deployment resolution for the worker.

A "resolved deployment" gives the runtime the disk root + parsed manifest.
Per-skill venvs are built lazily by `build_skill_python()` based on each
skill's own `requirements.txt`. Multiple skills with identical requirements
share a single venv (keyed by the requirements file's sha).

Two sources:

1. LOCAL_PROJECT_PATH (dev mode): a real directory on the host. Default
   python = sys.executable; per-skill venvs are still honored if present.

2. A deployment row in the DB: download the zip from storage (cached by id),
   extract it. Per-skill venvs are then built on first job.

Layout under deployments_root:
    bundles/<deployment_id>/      # extracted bundle
    venvs/<reqs_sha256>/          # shared venv, keyed by requirements.txt sha
"""

from __future__ import annotations

import fcntl
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import structlog

from .config import get_settings
from .manifest import Manifest, parse_bundle_dir
from .storage import download

log = structlog.get_logger()


@dataclass
class ResolvedDeployment:
    root: Path
    manifest: Manifest
    deployment_id: str | None  # None in dev mode


# ----------------------------------------------------------------------- dev
def resolve_local() -> ResolvedDeployment:
    s = get_settings()
    root = Path(s.local_project_path).expanduser().resolve()  # type: ignore[arg-type]
    if not root.is_dir():
        raise RuntimeError(f"LOCAL_PROJECT_PATH does not exist: {root}")
    manifest = parse_bundle_dir(root)
    return ResolvedDeployment(
        root=root,
        manifest=manifest,
        deployment_id=None,
    )


# ----------------------------------------------------------------- prod cache
def _bundles_dir() -> Path:
    s = get_settings()
    p = Path(s.deployments_root) / "bundles"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _venvs_dir() -> Path:
    s = get_settings()
    p = Path(s.deployments_root) / "venvs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_bundle(deployment_id: str, storage_path: str) -> Path:
    """Download + extract the deployment zip once; subsequent calls reuse.

    Same concurrency hazard as `_build_venv`: jobs from one batch usually share
    a deployment, so extraction is serialized under a per-id flock.
    """
    target = _bundles_dir() / deployment_id
    sentinel = target / ".ready"
    if sentinel.exists():
        return target
    lock_path = _bundles_dir() / f".{deployment_id}.lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        if sentinel.exists():
            return target
        s = get_settings()
        tmp_zip = Path(tempfile.gettempdir()) / f"dep-{deployment_id}.zip"
        log.info("deployment_downloading", deployment_id=deployment_id, storage_path=storage_path)
        tmp_zip.write_bytes(download(s.deployments_bucket, storage_path))
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(target)
        sentinel.touch()
        tmp_zip.unlink(missing_ok=True)
        log.info("deployment_ready", deployment_id=deployment_id, root=str(target))
    return target


def _build_venv(requirements_file: Path) -> Path:
    """Build a venv keyed by sha256(requirements.txt); shared across deployments
    with identical deps. Idempotent.

    Concurrent jobs (asyncio.to_thread callers) can hit the same sha at once;
    without exclusion one job rmtree's the half-built venv out from under the
    other's `python -m venv`. An exclusive flock per sha serializes the build;
    losers wait, then return the now-ready venv via the sentinel re-check.
    """
    sha = hashlib.sha256(requirements_file.read_bytes()).hexdigest()[:16]
    venv = _venvs_dir() / sha
    sentinel = venv / ".ready"
    if sentinel.exists():
        return venv
    lock_path = _venvs_dir() / f".{sha}.lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        if sentinel.exists():
            return venv
        log.info("venv_building", sha=sha, requirements=str(requirements_file))
        if venv.exists():
            shutil.rmtree(venv)
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        pip = venv / "bin" / "pip"
        subprocess.run(
            [str(pip), "install", "--no-cache-dir", "-r", str(requirements_file)],
            check=True,
        )
        sentinel.touch()
        log.info("venv_ready", sha=sha)
    return venv


def resolve_deployment(deployment_id: str, storage_path: str) -> ResolvedDeployment:
    root = _ensure_bundle(deployment_id, storage_path)
    manifest = parse_bundle_dir(root)
    return ResolvedDeployment(
        root=root,
        manifest=manifest,
        deployment_id=deployment_id,
    )


def build_skill_python(skill_root: Path) -> tuple[str, Path | None]:
    """Return (python_exe, venv_dir) for a skill.

    If the skill has a non-empty `requirements.txt`, build (or reuse) a venv
    keyed by the file's sha and return its python. Otherwise fall back to the
    worker's own interpreter.
    """
    req = skill_root / "requirements.txt"
    if req.exists() and req.stat().st_size > 0:
        venv = _build_venv(req)
        return str(venv / "bin" / "python"), venv
    return sys.executable, None
