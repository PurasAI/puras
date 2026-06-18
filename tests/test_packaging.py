"""`pip install puras` must pull the offline runner — no `[local]` extra needed.

`puras-runner` (the `worker` agent loop) is a CORE dependency of the `puras`
SDK so a plain `pip install puras` gives a working `puras run --local`. This
guards against it slipping back to optional-only, which is exactly the install
papercut that motivated the merge (`pip install puras` → `worker` missing).
"""

from __future__ import annotations

from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "worker" / "sdk" / "pyproject.toml"


def _core_deps_block() -> str:
    """The text before [project.optional-dependencies] — i.e. core `[project]`
    deps, never the extras. Avoids tomllib (added in 3.11; we target 3.10)."""
    text = _PYPROJECT.read_text()
    core, sep, _extras = text.partition("[project.optional-dependencies]")
    assert sep, "expected an [project.optional-dependencies] section to split on"
    return core


def test_runner_is_a_core_dependency():
    assert "puras-runner" in _core_deps_block(), (
        "puras-runner must be a core dependency so `pip install puras` "
        "installs the offline runner without the [local] extra"
    )
