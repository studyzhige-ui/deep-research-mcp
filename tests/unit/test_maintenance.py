"""Tests for the prune maintenance utility.

These exercise the full path (registry + checkpoint scan, dry-run, and a
real deletion) against synthetic SQLite databases assembled to look like the
runtime's layout. We avoid importing langgraph directly — the pruner is
designed to discover the checkpoint schema at runtime, so it suffices to
create any sqlite file with a ``thread_id`` column.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _now_iso(offset_days: float = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=offset_days)).isoformat()


@pytest.fixture
async def populated_runtime(settings):
    """Populate the registry with a mix of fresh/stale, terminal/in-flight tasks.

    Also creates a fake checkpoint DB with a ``thread_id`` column so the
    pruner has something to delete on the checkpoint side.
    """
    from deep_research_runtime.storage import TaskRegistryStore

    store = TaskRegistryStore(settings)
    # Fresh terminal — must NOT be pruned (within retention window).
    await store.save_task_meta("fresh1", {
        "lifecycle": "completed",
        "completed_at": _now_iso(offset_days=-1),
    })
    # Stale terminal — must be pruned.
    await store.save_task_meta("stale1", {
        "lifecycle": "completed",
        "completed_at": _now_iso(offset_days=-100),
    })
    await store.save_task_meta("stale2", {
        "lifecycle": "failed",
        "failed_at": _now_iso(offset_days=-100),
    })
    # In-flight — must NEVER be pruned regardless of age.
    await store.save_task_meta("running1", {
        "lifecycle": "running",
        "created_at": _now_iso(offset_days=-100),
    })
    # Sprinkle some events for stale1 so we can assert they're cleaned.
    for i in range(5):
        await store.append_task_event("stale1", "work", f"msg-{i}")
    await store.save_report_version("stale1", "# old report", change_summary="v1")
    await store.close()

    # Fake checkpoint db with a thread_id column matching task ids.
    runtime_dir = Path(settings.report_dir) / "_runtime"
    ckpt_path = runtime_dir / "DeepResearch_GraphCheckpoints.sqlite"
    conn = sqlite3.connect(str(ckpt_path))
    try:
        conn.execute(
            "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_id TEXT, blob TEXT)"
        )
        for tid in ("fresh1", "stale1", "stale1", "stale2", "running1"):
            conn.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?)",
                (tid, f"ckpt-{tid}", "{}"),
            )
        conn.commit()
    finally:
        conn.close()

    return settings


@pytest.mark.asyncio
async def test_dry_run_lists_candidates_without_deleting(populated_runtime):
    from deep_research_runtime.maintenance import prune

    result = prune(populated_runtime, retention_days=30, dry_run=True)
    candidate_ids = {c["task_id"] for c in result["candidates"]}
    assert candidate_ids == {"stale1", "stale2"}
    assert result["dry_run"] is True
    assert result["checkpoint_deleted"] == {}

    # Registry rows still intact.
    runtime_dir = Path(populated_runtime.report_dir) / "_runtime"
    conn = sqlite3.connect(str(runtime_dir / "DeepResearch_TaskRegistry.sqlite"))
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE task_id = 'stale1'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 1


@pytest.mark.asyncio
async def test_actual_prune_removes_stale_tasks(populated_runtime):
    from deep_research_runtime.maintenance import prune

    result = prune(populated_runtime, retention_days=30, dry_run=False)
    candidate_ids = {c["task_id"] for c in result["candidates"]}
    assert candidate_ids == {"stale1", "stale2"}
    assert result["dry_run"] is False
    # Each stale task had at least 1 checkpoint row.
    assert result["checkpoint_deleted"]["checkpoints"] >= 3
    # Registry: tasks + task_events + report_versions should drop rows.
    assert result["registry_deleted"]["tasks"] == 2
    assert result["registry_deleted"]["task_events"] == 5
    assert result["registry_deleted"]["report_versions"] == 1


@pytest.mark.asyncio
async def test_running_tasks_are_never_pruned(populated_runtime):
    from deep_research_runtime.maintenance import prune

    result = prune(populated_runtime, retention_days=0, dry_run=False)
    # retention=0 means "anything terminal is fair game", but `running1`
    # has lifecycle="running" so it must survive.
    candidate_ids = {c["task_id"] for c in result["candidates"]}
    assert "running1" not in candidate_ids

    runtime_dir = Path(populated_runtime.report_dir) / "_runtime"
    conn = sqlite3.connect(str(runtime_dir / "DeepResearch_TaskRegistry.sqlite"))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE task_id = 'running1'"
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_prune_handles_missing_checkpoint_file(settings):
    """If the checkpoint DB doesn't exist yet, prune should still succeed."""
    from deep_research_runtime.maintenance import prune
    from deep_research_runtime.storage import TaskRegistryStore

    store = TaskRegistryStore(settings)
    await store.save_task_meta("old", {
        "lifecycle": "failed",
        "failed_at": _now_iso(offset_days=-100),
    })
    await store.close()
    # No checkpoint file is created — only the registry exists.
    result = prune(settings, retention_days=30, dry_run=False)
    assert result["candidates"][0]["task_id"] == "old"
    assert result["checkpoint_tables_discovered"] == []
