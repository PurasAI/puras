"""Credential + project config.

Two layers:

  ~/.puras/config.json   {api_base, api_key}    — global auth, written by `login`
  ./puras.yaml           the pack manifest      — per-skillpack: the remote
                         binding (`skillpack_id`, `slug`, written by `init` /
                         cached by `deploy`) PLUS the authored pack-page
                         content (`title`, `description`, `marketing`). It
                         ships INSIDE the bundle, where the server parses the
                         content; the binding keys are server-ignored.

Env always wins: PURAS_API_KEY / PURAS_API_BASE override the global file.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_API_BASE = "https://api.puras.co"
GLOBAL_FILE = Path.home() / ".puras" / "config.json"
PROJECT_FILE = "puras.yaml"

# The binding keys the CLI owns inside puras.yaml. Everything else in the file
# (title/description/marketing) is authored by hand — `save_project` updates
# these lines surgically so it never reformats or drops the authored content.
_BINDING_KEYS = ("skillpack_id", "slug")


@dataclass
class Auth:
    api_base: str
    api_key: str | None


def _read_json(p: Path) -> dict:
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_auth() -> Auth:
    data = _read_json(GLOBAL_FILE)
    api_base = (
        os.environ.get("PURAS_API_BASE") or data.get("api_base") or DEFAULT_API_BASE
    ).rstrip("/")
    api_key = os.environ.get("PURAS_API_KEY") or data.get("api_key")
    return Auth(api_base=api_base, api_key=api_key)


def save_auth(api_base: str, api_key: str) -> None:
    GLOBAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_FILE.write_text(
        json.dumps({"api_base": api_base.rstrip("/"), "api_key": api_key}, indent=2) + "\n"
    )
    try:
        GLOBAL_FILE.chmod(0o600)
    except OSError:
        pass


def clear_auth() -> bool:
    try:
        GLOBAL_FILE.unlink()
        return True
    except OSError:
        return False


def _read_yaml(p: Path) -> dict:
    try:
        data = yaml.safe_load(p.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def find_project_file(start: Path | None = None) -> Path | None:
    """Nearest puras.yaml, walking up from `start`."""
    cur = (start or Path.cwd()).resolve()
    for d in [cur, *cur.parents]:
        f = d / PROJECT_FILE
        if f.is_file():
            return f
    return None


def load_project() -> dict:
    f = find_project_file()
    return _read_yaml(f) if f else {}


def save_project(directory: Path, data: dict) -> Path:
    """Persist the remote binding into `puras.yaml`, preserving any authored
    content (title/description/marketing) already in the file.

    Existing binding lines are replaced in place; missing ones are prepended.
    Only top-level scalar `key: value` lines are touched — never the authored
    blocks."""
    f = directory / PROJECT_FILE
    binding = {k: data[k] for k in _BINDING_KEYS if data.get(k)}

    text = f.read_text() if f.is_file() else ""
    lines = text.split("\n") if text else []
    for key, val in binding.items():
        pat = re.compile(rf"^{key}\s*:")
        line = f"{key}: {val}"
        for i, ln in enumerate(lines):
            if pat.match(ln):
                lines[i] = line
                break
        else:
            lines.insert(0, line)
    f.write_text("\n".join(lines).rstrip("\n") + "\n")
    return f
