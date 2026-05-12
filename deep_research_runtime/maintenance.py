"""Maintenance utilities for the Deep Research engine.

This module is invoked by the ``deep-research-mcp prune`` CLI subcommand and
is responsible for keeping the on-disk state from growing unboundedly.

What grows over time:

* **TaskRegistry SQLite** — every task adds rows to ``tasks`` (1 row),
  ``task_events`` (often hundreds during a long run), and possibly
  ``report_versions``. Old completed tasks rarely need to be queried again.
* **LangGraph checkpoint SQLite** — every graph step writes a checkpoint
  row keyed by ``thread_id`` (== our ``task_id``). Long-running follow-ups
  on the same task accumulate many checkpoints.

The pruner takes a conservative approach: it only removes data tied to
**completed or failed tasks older than ``retention_days``**. Tasks still in
``running``/``draft``/``pending`` state are never touched, so an in-flight
research is safe.

Design choices:

* The pruner discovers checkpoint tables at runtime via ``sqlite_master``
  rather than hardcoding LangGraph's schema, so it keeps working across
  LangGraph minor versions.
* It deletes by ``thread_id`` only (the public concept) so we never depend
  on internal column names like ``parent_checkpoint_id``.
* Dry-run mode reports what *would* be deleted without touching anything.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .settings import Settings


# Lifecycle values that mark a task as terminal and therefore eligible for
# pruning once it's older than the retention window.
TERMINAL_LIFECYCLES = {"completed", "failed", "cancelled"}


def _parse_iso(value: str) -> Optional[datetime]:
    """Best-effort ISO-8601 parser that tolerates the slight variations the
    storage layer emits (``now_iso`` includes tz info; older rows might not).
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _list_prunable_task_ids(registry_path: Path, retention_days: int) -> List[Tuple[str, str, str]]:
    """Return ``(task_id, lifecycle, finished_at)`` for tasks past retention.

    We open the registry SQLite read-only via a separate sync connection so
    this can run as part of a CLI tool without coordinating with a live
    server's aiosqlite connection.
    """
    if not registry_path.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, retention_days))
    conn = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT task_id, meta_json, updated_at FROM tasks"
        ).fetchall()
    finally:
        conn.close()

    out: List[Tuple[str, str, str]] = []
    import json

    for row in rows:
        try:
            meta = json.loads(row["meta_json"] or "{}")
        except Exception:
            meta = {}
        lifecycle = str(meta.get("lifecycle") or "").lower()
        if lifecycle not in TERMINAL_LIFECYCLES:
            continue
        # Prefer the lifecycle-specific timestamp, else fall back to updated_at.
        finished_iso = (
            meta.get("completed_at")
            or meta.get("failed_at")
            or meta.get("cancelled_at")
            or row["updated_at"]
        )
        finished_dt = _parse_iso(str(finished_iso or ""))
        if finished_dt is None or finished_dt > cutoff:
            continue
        out.append((str(row["task_id"]), lifecycle, str(finished_iso)))
    return out


def _discover_checkpoint_tables(checkpoint_path: Path) -> List[Tuple[str, str]]:
    """Find tables in the checkpoint DB that have a ``thread_id`` column.

    Returns a list of ``(table_name, column_name)`` pairs. We accept either
    a ``thread_id`` column (used by current LangGraph) or a fallback name
    so a small schema rename doesn't break the pruner silently.
    """
    if not checkpoint_path.exists():
        return []
    conn = sqlite3.connect(f"file:{checkpoint_path}?mode=ro", uri=True)
    try:
        names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        result: List[Tuple[str, str]] = []
        for table in names:
            cols = [
                row[1]
                for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            ]
            for candidate in ("thread_id", "thread"):
                if candidate in cols:
                    result.append((table, candidate))
                    break
        return result
    finally:
        conn.close()


def _delete_checkpoint_rows(
    checkpoint_path: Path,
    tables: List[Tuple[str, str]],
    task_ids: List[str],
) -> Dict[str, int]:
    """Delete rows in each checkpoint table whose thread_id is in ``task_ids``.

    Returns a per-table count of deleted rows. We chunk the IN clause to keep
    SQLite happy with the variable-binding limit.
    """
    if not task_ids:
        return {}
    deleted: Dict[str, int] = {}
    conn = sqlite3.connect(str(checkpoint_path))
    try:
        with conn:
            for table, col in tables:
                total = 0
                # SQLite's default SQLITE_MAX_VARIABLE_NUMBER is 32766 on
                # modern builds, but stay well below that to be safe.
                chunk_size = 500
                for i in range(0, len(task_ids), chunk_size):
                    chunk = task_ids[i : i + chunk_size]
                    placeholders = ",".join("?" * len(chunk))
                    cursor = conn.execute(
                        f'DELETE FROM "{table}" WHERE "{col}" IN ({placeholders})',
                        chunk,
                    )
                    total += cursor.rowcount or 0
                deleted[table] = total
    finally:
        conn.close()
    return deleted


def _delete_registry_rows(
    registry_path: Path,
    task_ids: List[str],
) -> Dict[str, int]:
    """Delete the registry rows tied to the supplied task ids.

    We remove from ``tasks``, ``task_events`` and ``report_versions`` so a
    pruned task is fully gone — its metadata, timeline, and report history.
    """
    if not task_ids or not registry_path.exists():
        return {}
    deleted: Dict[str, int] = {}
    conn = sqlite3.connect(str(registry_path))
    try:
        with conn:
            for table in ("task_events", "report_versions", "tasks"):
                total = 0
                chunk_size = 500
                for i in range(0, len(task_ids), chunk_size):
                    chunk = task_ids[i : i + chunk_size]
                    placeholders = ",".join("?" * len(chunk))
                    cursor = conn.execute(
                        f"DELETE FROM {table} WHERE task_id IN ({placeholders})",
                        chunk,
                    )
                    total += cursor.rowcount or 0
                deleted[table] = total
        # VACUUM has to run outside any transaction; the `with conn` block
        # above commits and closes its implicit transaction by the time we
        # get here. Wrap in try because VACUUM occasionally fails on Windows
        # if another connection still has the file open — disk reclaim is
        # best-effort, not load-bearing.
        try:
            conn.execute("VACUUM")
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()
    return deleted


def prune(
    settings: Settings,
    *,
    retention_days: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Prune terminal tasks older than ``retention_days``.

    Parameters
    ----------
    settings:
        Used to locate the registry + checkpoint files (under
        ``<report_dir>/_runtime``).
    retention_days:
        Overrides ``settings.checkpoint_retention_days`` when given. The CLI
        passes the value the user typed on the command line.
    dry_run:
        When ``True``, identify what would be pruned but do not delete
        anything. Useful for the CLI's ``--dry-run`` flag.

    Returns a structured report suitable for printing or asserting on in
    tests::

        {
            "retention_days": int,
            "candidates": [{"task_id": ..., "lifecycle": ..., "finished_at": ...}, ...],
            "checkpoint_deleted": {table: rows, ...},
            "registry_deleted": {table: rows, ...},
            "dry_run": bool,
        }
    """
    retention = settings.checkpoint_retention_days if retention_days is None else retention_days

    runtime_dir = Path(settings.report_dir) / "_runtime"
    registry_path = runtime_dir / "DeepResearch_TaskRegistry.sqlite"
    checkpoint_path = runtime_dir / "DeepResearch_GraphCheckpoints.sqlite"

    candidates = _list_prunable_task_ids(registry_path, retention)
    candidate_ids = [task_id for task_id, _, _ in candidates]

    checkpoint_tables = _discover_checkpoint_tables(checkpoint_path)

    if dry_run or not candidate_ids:
        return {
            "retention_days": retention,
            "candidates": [
                {"task_id": tid, "lifecycle": lc, "finished_at": ft}
                for tid, lc, ft in candidates
            ],
            "checkpoint_tables_discovered": [name for name, _ in checkpoint_tables],
            "checkpoint_deleted": {},
            "registry_deleted": {},
            "dry_run": True if dry_run else False,
        }

    checkpoint_deleted = _delete_checkpoint_rows(
        checkpoint_path, checkpoint_tables, candidate_ids
    )
    registry_deleted = _delete_registry_rows(registry_path, candidate_ids)

    return {
        "retention_days": retention,
        "candidates": [
            {"task_id": tid, "lifecycle": lc, "finished_at": ft}
            for tid, lc, ft in candidates
        ],
        "checkpoint_tables_discovered": [name for name, _ in checkpoint_tables],
        "checkpoint_deleted": checkpoint_deleted,
        "registry_deleted": registry_deleted,
        "dry_run": False,
    }
