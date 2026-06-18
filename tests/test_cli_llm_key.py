"""`puras run --local` persists the BYO LLM key instead of re-prompting.

Regression: `_resolve_llm_key` asked for ANTHROPIC_API_KEY on EVERY local run —
the entered key was used once and never stored. It now saves the first prompt
into ~/.puras/config.json (0600) and reads it back on later runs, while keeping
the env var as the higher-priority source and never clobbering the workspace
login that shares the same file.
"""

from __future__ import annotations

import json
import types

import pytest

from puras.cli import commands, config


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    """Redirect the global config file to a throwaway path so a test never reads
    or writes the real ~/.puras/config.json. ANTHROPIC_API_KEY starts unset so
    the saved/prompt branches are the ones exercised."""
    f = tmp_path / "config.json"
    monkeypatch.setattr(config, "GLOBAL_FILE", f)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return f


def _args(api_key=None):
    return types.SimpleNamespace(api_key=api_key)


def test_save_then_load_llm_key_roundtrips(cfg):
    assert config.load_llm_key() is None
    config.save_llm_key("sk-abc")
    assert config.load_llm_key() == "sk-abc"
    assert json.loads(cfg.read_text())["llm_api_key"] == "sk-abc"
    assert (cfg.stat().st_mode & 0o777) == 0o600  # secret → 0600


def test_login_preserves_saved_llm_key(cfg):
    config.save_llm_key("sk-byo")
    config.save_auth("https://api.test", "wk-123")  # a later `puras login`
    data = json.loads(cfg.read_text())
    assert data["api_key"] == "wk-123"      # workspace key written
    assert data["llm_api_key"] == "sk-byo"  # BYO key survives the merge


def test_resolve_env_wins_over_saved(cfg, monkeypatch):
    config.save_llm_key("sk-saved")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert commands._resolve_llm_key(_args()) == "sk-env"


def test_resolve_returns_saved_without_prompting(cfg, monkeypatch):
    config.save_llm_key("sk-saved")
    # Any prompt attempt fails the test — proving the saved key short-circuits it.
    monkeypatch.setattr(
        commands.getpass, "getpass",
        lambda *_a, **_k: pytest.fail("should not prompt when a key is saved"),
    )
    assert commands._resolve_llm_key(_args()) == "sk-saved"


def test_resolve_prompts_once_then_persists(cfg, monkeypatch):
    assert config.load_llm_key() is None
    monkeypatch.setattr(commands.sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(commands.getpass, "getpass", lambda *_a, **_k: "  sk-typed  ")
    out = commands._resolve_llm_key(_args())
    assert out == "sk-typed"                    # whitespace stripped
    assert config.load_llm_key() == "sk-typed"  # persisted → no prompt next run
