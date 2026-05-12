"""Shared pytest fixtures for the deep-research-mcp test suite.

These fixtures intentionally avoid any network or LLM calls — every test in
this suite must be runnable without API keys. Tests that exercise real
external APIs live under ``tests/integration/`` and are gated behind a
``-m integration`` marker (not collected by default).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_report_dir(monkeypatch):
    """Point DEEP_RESEARCH_REPORT_DIR at a fresh temp dir for the test.

    Most of the server's filesystem state (task registry, checkpoint sqlite,
    worker logs) lives under this directory, so isolating it per-test means
    tests don't see each other's data and the developer's real reports
    are untouched.
    """
    with tempfile.TemporaryDirectory() as path:
        monkeypatch.setenv("DEEP_RESEARCH_REPORT_DIR", path)
        yield Path(path)


@pytest.fixture
def settings(tmp_report_dir):
    """A freshly-constructed Settings instance bound to the tmp report dir.

    Settings field defaults are evaluated at class-definition time (when
    settings.py is first imported), so simply setenv-ing here is too late.
    We instead override the relevant attributes on the instance — this gives
    tests proper isolation without forcing a settings-module reload between
    every test.
    """
    from deep_research_runtime.settings import Settings

    s = Settings()
    s.report_dir = str(tmp_report_dir)
    return s


@pytest.fixture
async def store(settings):
    """An empty TaskRegistryStore backed by an isolated SQLite file."""
    from deep_research_runtime.storage import TaskRegistryStore

    registry = TaskRegistryStore(settings)
    try:
        yield registry
    finally:
        await registry.close()
