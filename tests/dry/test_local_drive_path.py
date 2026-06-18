"""`puras run --local` honors an exported LOCAL_DRIVE_PATH.

Regression: `_prepare_env` assigned LOCAL_DRIVE_PATH with plain `=` (not
`setdefault` like its siblings), so without an explicit `--drive-path` it
clobbered a user's exported value back to the ephemeral $TMPDIR default — the
drive silently landed somewhere other than where the user pointed it.

Precedence is now: `--drive-path` (the `drive_path` arg) → an exported
`$LOCAL_DRIVE_PATH` → the local scratch default. The flag must still win, so
this isn't a pure `setdefault`.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from worker import local_run


@pytest.fixture()
def _isolated_env(monkeypatch):
    """`_prepare_env` mutates os.environ directly; snapshot + restore so a call
    never leaks env into sibling tests. A real-looking key satisfies the
    BYO-key guard; LOCAL_DRIVE_PATH starts unset per test (each sets its own)."""
    snapshot = dict(os.environ)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("LOCAL_DRIVE_PATH", raising=False)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


def test_exported_local_drive_path_is_honored(_isolated_env, monkeypatch):
    monkeypatch.setenv("LOCAL_DRIVE_PATH", "/my/drive")
    local_run._prepare_env(api_key=None, drive_path=None)
    assert os.environ["LOCAL_DRIVE_PATH"] == "/my/drive"


def test_drive_path_flag_wins_over_export(_isolated_env, monkeypatch):
    monkeypatch.setenv("LOCAL_DRIVE_PATH", "/my/drive")
    local_run._prepare_env(api_key=None, drive_path="/flag/drive")
    assert os.environ["LOCAL_DRIVE_PATH"] == "/flag/drive"


def test_defaults_to_local_scratch_when_unset(_isolated_env):
    local_run._prepare_env(api_key=None, drive_path=None)
    assert os.environ["LOCAL_DRIVE_PATH"] == str(
        Path(tempfile.gettempdir()) / "puras-local" / "drive"
    )
