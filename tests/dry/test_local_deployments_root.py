"""`puras run --local` must pick a writable deployments_root without root.

Regression: `_prepare_env` redirected the drive + workdir to a local scratch
dir but forgot deployments_root, so it stayed at the hosted default
(/var/puras/deployments) — root-owned, unwritable for a plain `pip install`
user. `build_skill_python()` then died on `mkdir` before the agent loop ever
started. `_prepare_env` now setdefaults DEPLOYMENTS_ROOT under the same local
scratch folder as the drive/workdir, while still honoring a user override.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from worker import local_run


@pytest.fixture()
def _isolated_env(monkeypatch):
    """`_prepare_env` mutates os.environ directly (setdefault / assignment),
    which monkeypatch can't auto-revert. Snapshot + restore so a call never
    leaks env into sibling tests. A real-looking key satisfies the BYO-key
    guard; the paths under test start unset so the setdefault branch runs."""
    snapshot = dict(os.environ)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    for name in ("DEPLOYMENTS_ROOT", "WORKDIR_ROOT", "LOCAL_DRIVE_PATH"):
        monkeypatch.delenv(name, raising=False)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


def test_deployments_root_defaults_under_local_scratch(_isolated_env):
    local_run._prepare_env(api_key=None, drive_path=None)
    root = os.environ["DEPLOYMENTS_ROOT"]
    # Never the root-owned hosted default — that's the bug this guards against.
    assert not root.startswith("/var/puras")
    # Same local scratch folder as the drive/workdir.
    assert root == str(Path(tempfile.gettempdir()) / "puras-local" / "deployments")


def test_deployments_root_user_override_is_honored(_isolated_env, monkeypatch):
    monkeypatch.setenv("DEPLOYMENTS_ROOT", "/custom/dep/root")
    local_run._prepare_env(api_key=None, drive_path=None)
    assert os.environ["DEPLOYMENTS_ROOT"] == "/custom/dep/root"
