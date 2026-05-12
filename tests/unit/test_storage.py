"""Tests for the aiosqlite-backed TaskRegistryStore.

These guard the schema and concurrency contract: any future refactor must
keep the public async API and survive concurrent writes from many tasks
without losing events.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.asyncio


async def test_save_and_load_task_meta(store):
    await store.save_task_meta("t1", {"topic": "x", "status_message": "draft"})
    meta = await store.load_task_meta("t1")
    assert meta["topic"] == "x"
    assert meta["status_message"] == "draft"
    # created_at/updated_at are populated by save_task_meta itself
    assert meta["created_at"]
    assert meta["updated_at"]


async def test_status_update_preserves_created_at(store):
    await store.save_task_meta("t1", {"topic": "x"})
    created = (await store.load_task_meta("t1"))["created_at"]
    await asyncio.sleep(0.01)
    await store.set_status("t1", "running", lifecycle="running")
    meta = await store.load_task_meta("t1")
    assert meta["created_at"] == created
    assert meta["status_message"] == "running"
    assert meta["lifecycle"] == "running"


async def test_event_ordering(store):
    for i in range(5):
        await store.append_task_event("t1", "stage-a", f"msg-{i}", level="info")
    events = await store.recent_task_events("t1", limit=10)
    assert [e["message"] for e in events] == [f"msg-{i}" for i in range(5)]


async def test_concurrent_writes_dont_lose_events(store):
    """Hammer the store with many concurrent writers; nothing should be lost."""

    async def writer(task: str, n: int) -> None:
        for i in range(n):
            await store.append_task_event(task, "work", f"{task}-{i}")

    await asyncio.gather(*(writer(f"t{k}", 25) for k in range(8)))
    for k in range(8):
        events = await store.recent_task_events(f"t{k}", limit=100)
        assert len(events) == 25


async def test_report_version_increments(store):
    v1 = await store.save_report_version("t1", "# v1", change_summary="initial")
    v2 = await store.save_report_version("t1", "# v2", change_summary="follow-up")
    v3 = await store.save_report_version("t1", "# v3", change_summary="patch")
    assert (v1, v2, v3) == (1, 2, 3)
    versions = await store.list_report_versions("t1")
    assert [v["version"] for v in versions] == [1, 2, 3]
    latest = await store.load_report_version("t1", version=0)
    assert latest["version"] == 3
    assert latest["content"] == "# v3"


async def test_progress_events_filtered_from_normal_events(store):
    await store.append_task_event("t1", "stage", "regular event")
    await store.append_progress_event(
        "t1", "section_done", section_id="s1", status="ok", message="done"
    )
    # recent_task_events returns ALL events (regular + progress); progress
    # events are stored under stage='progress'.
    regular = await store.recent_task_events("t1", limit=10)
    assert len(regular) == 2

    progress = await store.get_progress_events("t1", limit=10)
    assert len(progress) == 1
    assert progress[0]["type"] == "section_done"
    assert progress[0]["section_id"] == "s1"


async def test_ping_returns_true_on_healthy_store(store):
    assert await store.ping() is True
