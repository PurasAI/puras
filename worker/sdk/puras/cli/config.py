"""Credential + project config.

Two layers:

  ~/.puras/config.json   {api_base, api_key, llm_api_key}
                         — global auth (api_base/api_key, written by `login`)
                         PLUS the BYO LLM key a local run saves on first prompt
                         (llm_api_key) so `puras run --local` doesn't re-ask it
                         every time.
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
# The BYO LLM key field in the global file (see `save_llm_key`).
LLM_KEY_FIELD = "llm_api_key"

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


def _write_global(data: dict) -> None:
    """Write the global config dict, 0600 — it holds secrets."""
    GLOBAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_FILE.write_text(json.dumps(data, indent=2) + "\n")
    try:
        GLOBAL_FILE.chmod(0o600)
    except OSError:
        pass


def save_auth(api_base: str, api_key: str) -> None:
    # Merge, don't clobber: a saved BYO LLM key (llm_api_key) shares this file
    # and must survive a `login`.
    data = _read_json(GLOBAL_FILE)
    data["api_base"] = api_base.rstrip("/")
    data["api_key"] = api_key
    _write_global(data)


def clear_auth() -> bool:
    try:
        GLOBAL_FILE.unlink()
        return True
    except OSError:
        return False


def load_llm_key() -> str | None:
    """The BYO LLM key saved by a prior `puras run --local`, if any. Stored next
    to the workspace auth in ~/.puras/config.json so the one-time prompt isn't
    re-asked every run. The ANTHROPIC_API_KEY env var still wins upstream (see
    cli._resolve_llm_key); this is only the fallback when it's unset."""
    return _read_json(GLOBAL_FILE).get(LLM_KEY_FIELD)


def save_llm_key(key: str) -> None:
    """Persist the BYO LLM key without disturbing the workspace auth fields."""
    data = _read_json(GLOBAL_FILE)
    data[LLM_KEY_FIELD] = key
    _write_global(data)


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
