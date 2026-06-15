"""Zip a skillpack directory into an in-memory deployment bundle.

Mirrors what the server expects (see api/app/manifest.py `parse_bundle_zip`): a
*flat* layout with `<skill>/skill.yaml` at the archive root — no `skills/`
wrapper. Subskills live at `<skill>/subskills/<sub>/skill.yaml`. The root
`puras.yaml` (pack manifest) SHIPS in the bundle — the server reads the pack's
title/description/marketing from it. Dev/VCS cruft is excluded so the bundle
stays small.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {
    ".git", ".hg", ".svn",
    ".venv", "venv", "env",
    "node_modules",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".idea", ".vscode",
    "dist", "build", ".next",
}
EXCLUDE_FILES = {".DS_Store", ".env", ".env.local"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log"}


def skill_dirs(root: Path) -> list[Path]:
    """Top-level skill directories (each holds a `skill.yaml`) in `root`.

    Used to detect the single-skill case — `puras deploy` names the deployment
    after the lone skill so the end user never has to think about a pack."""
    return [
        d
        for d in sorted(root.iterdir())
        if d.is_dir() and d.name not in EXCLUDE_DIRS and (d / "skill.yaml").is_file()
    ]


def zip_skillpack(root: Path) -> bytes:
    root = root.resolve()
    if not skill_dirs(root):
        raise FileNotFoundError(
            f"no `<skill>/skill.yaml` found in {root} — run this from your "
            f"skill dir (or `puras init` to create one)"
        )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            if p.is_dir():
                continue
            if p.name in EXCLUDE_FILES or p.suffix in EXCLUDE_SUFFIXES:
                continue
            zf.write(p, str(rel))
    return buf.getvalue()
