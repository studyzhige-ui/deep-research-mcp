"""SQLite-backed task registry for the Deep Research engine.

Originally this module wrapped a synchronous ``sqlite3.Connection`` behind an
``asyncio.Lock``.  That works, but every database call still blocked the event
loop while sqlite did its work — under concurrent tasks the latency stacked
up and writes against the same connection could time out.

The current implementation keeps the **exact same external async API** but:

* Uses :mod:`aiosqlite` so the actual SQL runs on a background thread and the
  event loop stays responsive.
* Enables WAL journal mode + ``synchronous=NORMAL`` to allow concurrent
  readers alongside the single writer, which is what the research workflow
  needs (many ``recent_task_events`` / ``get_progress_events`` polls during a
  long-running task).
* Performs schema/pragma initialization synchronously in ``__init__`` so the
  file is guaranteed ready before any async caller appears.  The sync
  connection is closed immediately after; runtime traffic goes through the
  aiosqlite connection.

Schema, column names, and behavior of every public method are unchanged —
this is a pure infrastructure swap.
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

from .settings import Settings


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


class TaskRegistryStore:
    """Persistent registry for tasks, events, and report versions.

    Thread-safety / concurrency model:

    * One shared ``aiosqlite`` connection is opened lazily on the first async
      call and reused for the lifetime of the store.
    * ``self._lock`` serializes write operations so we never interleave a
      multi-statement update (read-modify-write of ``meta_json``). Reads
      against WAL can run in parallel at the SQLite layer; the asyncio lock
      only adds extra ordering when convenient — it does not hurt
      correctness.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._runtime_dir = Path(self.settings.report_dir) / "_runtime"
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._runtime_dir / "DeepResearch_TaskRegistry.sqlite"
        self._lock = asyncio.Lock()
        self._aconn: Optional[aiosqlite.Connection] = None
        self._aconn_lock = asyncio.Lock()

        # Initialize schema + pragmas synchronously so the file is ready
        # before the first async caller. Close the sync handle immediately —
        # runtime traffic goes through aiosqlite.
        self._bootstrap_sync()

    # ──────────────────────────────────────────────────────────────────
    #  Bootstrap (sync, runs once)
    # ──────────────────────────────────────────────────────────────────

    # Column names that mirror frequently-filtered/displayed meta_json fields
    # into their own table columns. meta_json remains the source of truth;
    # the columns exist so list/status queries can run as native SQL without
    # parsing every row's JSON blob, and so a future index can be added on
    # ``lifecycle`` without changing application code.
    _MIRRORED_COLUMNS = ("lifecycle", "stage", "error_code", "topic")

    def _bootstrap_sync(self) -> None:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            # WAL gives us non-blocking reads alongside the single writer.
            # synchronous=NORMAL is safe with WAL and ~2x faster than FULL.
            # busy_timeout protects against transient lock contention if the
            # OS or an external tool briefly holds the file.
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA busy_timeout = 5000")
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        task_id TEXT PRIMARY KEY,
                        meta_json TEXT NOT NULL DEFAULT '{}',
                        draft_json TEXT NOT NULL DEFAULT '{}',
                        status_text TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL DEFAULT '',
                        updated_at TEXT NOT NULL DEFAULT ''
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS task_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        extra_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_task_events_task_id_id ON task_events(task_id, id)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS report_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        change_summary TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL DEFAULT '',
                        UNIQUE(task_id, version)
                    )
                    """
                )
                # Mirror high-frequency meta_json fields into their own
                # columns. Idempotent: ALTER TABLE ADD COLUMN is wrapped in
                # a PRAGMA check because SQLite < 3.35 has no
                # ``IF NOT EXISTS`` clause on ADD COLUMN. Existing databases
                # from older versions of this server therefore upgrade
                # transparently on next start.
                existing_cols = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
                }
                for col in self._MIRRORED_COLUMNS:
                    if col not in existing_cols:
                        conn.execute(
                            f"ALTER TABLE tasks ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                        )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tasks_lifecycle_updated ON tasks(lifecycle, updated_at)"
                )
                # Backfill mirrored columns for rows that existed before the
                # ALTER. Cheap: only touches rows where lifecycle column is
                # empty AND meta_json contains the field — old databases
                # only. Subsequent writes via ``save_task_meta`` keep the
                # columns in sync.
                rows = conn.execute(
                    "SELECT task_id, meta_json FROM tasks WHERE lifecycle = ''"
                ).fetchall()
                for row in rows:
                    try:
                        meta = json.loads(row["meta_json"] or "{}")
                    except Exception:
                        continue
                    if not isinstance(meta, dict):
                        continue
                    values = [str(meta.get(col) or "") for col in self._MIRRORED_COLUMNS]
                    if not any(values):
                        continue
                    assignments = ", ".join(f"{col} = ?" for col in self._MIRRORED_COLUMNS)
                    conn.execute(
                        f"UPDATE tasks SET {assignments} WHERE task_id = ?",
                        (*values, row["task_id"]),
                    )
        finally:
            conn.close()

    # ──────────────────────────────────────────────────────────────────
    #  Lazy aiosqlite connection
    # ──────────────────────────────────────────────────────────────────

    async def _conn(self) -> aiosqlite.Connection:
        """Return the shared aiosqlite connection, opening it on first call.

        We double-check the singleton under ``_aconn_lock`` so concurrent
        first-callers don't both try to open it.
        """
        if self._aconn is not None:
            return self._aconn
        async with self._aconn_lock:
            if self._aconn is None:
                conn = await aiosqlite.connect(str(self._db_path))
                conn.row_factory = aiosqlite.Row
                # Re-assert pragmas — WAL is persisted, but synchronous and
                # busy_timeout are per-connection.
                await conn.execute("PRAGMA synchronous = NORMAL")
                await conn.execute("PRAGMA busy_timeout = 5000")
                self._aconn = conn
        return self._aconn

    @property
    def registry_path(self) -> str:
        return str(self._db_path)

    async def ping(self) -> bool:
        try:
            conn = await self._conn()
            async with conn.execute("SELECT 1") as cur:
                await cur.fetchone()
            return True
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────
    #  Task metadata
    # ──────────────────────────────────────────────────────────────────

    async def _load_task_meta_inner(self, conn: aiosqlite.Connection, task_id: str) -> Dict[str, Any]:
        """Internal helper that assumes ``self._lock`` is already held."""
        async with conn.execute("SELECT meta_json FROM tasks WHERE task_id = ?", (task_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["meta_json"] or "{}")
        except Exception:
            return {}

    async def load_task_meta(self, task_id: str) -> Dict[str, Any]:
        conn = await self._conn()
        async with self._lock:
            return await self._load_task_meta_inner(conn, task_id)

    async def save_task_meta(
        self,
        task_id: str,
        updates: Dict[str, Any],
        *,
        status_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Merge ``updates`` into the persisted meta_json blob.

        ``status_text`` updates the dedicated ``status_text`` column in the
        same write, so a single call covers both ``meta_json`` and the
        free-text status display. Pass ``None`` (the default) to leave the
        column untouched.
        """
        conn = await self._conn()
        async with self._lock:
            current = await self._load_task_meta_inner(conn, task_id)
            current.update(updates)
            updated_at = now_iso()
            created_at = current.get("created_at") or updated_at
            current["updated_at"] = updated_at
            current["created_at"] = created_at
            async with conn.execute(
                "SELECT draft_json, status_text FROM tasks WHERE task_id = ?", (task_id,)
            ) as cur:
                row = await cur.fetchone()
            draft_json = row["draft_json"] if row else "{}"
            existing_status = row["status_text"] if row else ""
            new_status_text = (
                status_text if status_text is not None
                else (existing_status if row else str(current.get("status_message") or ""))
            )
            mirror_values = [str(current.get(col) or "") for col in self._MIRRORED_COLUMNS]
            mirror_cols = ", ".join(self._MIRRORED_COLUMNS)
            mirror_placeholders = ", ".join("?" for _ in self._MIRRORED_COLUMNS)
            mirror_assignments = ", ".join(
                f"{col}=excluded.{col}" for col in self._MIRRORED_COLUMNS
            )
            await conn.execute(
                f"""
                INSERT INTO tasks(task_id, meta_json, draft_json, status_text, created_at, updated_at, {mirror_cols})
                VALUES(?, ?, ?, ?, ?, ?, {mirror_placeholders})
                ON CONFLICT(task_id) DO UPDATE SET
                    meta_json=excluded.meta_json,
                    draft_json=excluded.draft_json,
                    status_text=excluded.status_text,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    {mirror_assignments}
                """,
                (
                    task_id,
                    json.dumps(current, ensure_ascii=False),
                    draft_json,
                    new_status_text,
                    created_at,
                    updated_at,
                    *mirror_values,
                ),
            )
            await conn.commit()
            return current

    # Timestamp keys used by ``finalize_task`` — one per terminal lifecycle.
    _FINALIZE_TS_KEY = {
        "completed": "completed_at",
        "failed": "failed_at",
        "cancelled": "cancelled_at",
    }
    _FINALIZE_LEVEL = {"completed": "info", "failed": "error", "cancelled": "warning"}

    async def finalize_task(
        self,
        task_id: str,
        lifecycle: str,
        *,
        status_msg: str,
        event_msg: Optional[str] = None,
        error: str = "",
        error_code: str = "",
        exception_type: str = "",
        traceback: str = "",
    ) -> None:
        """Atomically record a task's terminal transition.

        Replaces the historical ``save_task_meta`` + ``append_task_event`` +
        ``set_status`` triple-write. ``stage`` is intentionally NOT
        overwritten — the last in-progress stage is preserved so callers can
        see *where* the task terminated (e.g. ``stage=researcher`` plus
        ``lifecycle=failed`` is far more useful than the old
        ``stage=failed``).
        """
        if lifecycle not in self._FINALIZE_TS_KEY:
            raise ValueError(f"finalize_task: lifecycle must be one of {list(self._FINALIZE_TS_KEY)}, got {lifecycle!r}")
        ts_key = self._FINALIZE_TS_KEY[lifecycle]
        meta_updates: Dict[str, Any] = {
            "lifecycle": lifecycle,
            "status_message": status_msg,
            ts_key: now_iso(),
        }
        if error:
            meta_updates["error"] = error
        if error_code:
            meta_updates["error_code"] = error_code
        if exception_type:
            meta_updates["exception_type"] = exception_type
        await self.save_task_meta(task_id, meta_updates, status_text=status_msg)
        event_extra: Dict[str, Any] = {}
        if error:
            event_extra["error"] = error
        if error_code:
            event_extra["error_code"] = error_code
        if exception_type:
            event_extra["exception_type"] = exception_type
        if traceback:
            event_extra["traceback"] = traceback
        await self.append_task_event(
            task_id,
            lifecycle,
            event_msg or status_msg,
            level=self._FINALIZE_LEVEL[lifecycle],
            **event_extra,
        )

    # ──────────────────────────────────────────────────────────────────
    #  Events (timeline)
    # ──────────────────────────────────────────────────────────────────

    async def append_task_event(self, task_id: str, stage: str, message: str, level: str = "info", **extra: Any) -> None:
        conn = await self._conn()
        timestamp = now_iso()
        async with self._lock:
            await conn.execute(
                """
                INSERT INTO task_events(task_id, timestamp, stage, level, message, extra_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    timestamp,
                    stage,
                    level,
                    message,
                    json.dumps(extra, ensure_ascii=False),
                ),
            )
            await conn.commit()

    async def recent_task_events(self, task_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        conn = await self._conn()
        async with self._lock:
            async with conn.execute(
                """
                SELECT timestamp, stage, level, message, extra_json
                FROM task_events
                WHERE task_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (task_id, limit),
            ) as cur:
                rows = await cur.fetchall()
        items: List[Dict[str, Any]] = []
        for row in reversed(rows):
            try:
                extra = json.loads(row["extra_json"] or "{}")
            except Exception:
                extra = {}
            items.append(
                {
                    "timestamp": row["timestamp"],
                    "stage": row["stage"],
                    "level": row["level"],
                    "message": row["message"],
                    "extra": extra,
                }
            )
        return items

    # ──────────────────────────────────────────────────────────────────
    #  Status & draft
    # ──────────────────────────────────────────────────────────────────

    async def set_status(self, task_id: str, status: str, lifecycle: str = "", **extra: Any) -> None:
        """Update the free-text ``status_text`` column plus any meta fields.

        Thin shim over :meth:`save_task_meta`: previously this method had its
        own copy of the read-modify-write logic; collapsing onto
        save_task_meta eliminates the duplicate code path and ensures a
        single SQLite round-trip for a status update.
        """
        updates: Dict[str, Any] = {"status_message": status, **extra}
        if lifecycle:
            updates["lifecycle"] = lifecycle
        await self.save_task_meta(task_id, updates, status_text=status)

    async def get_status(self, task_id: str) -> str:
        conn = await self._conn()
        async with self._lock:
            async with conn.execute(
                "SELECT status_text FROM tasks WHERE task_id = ?", (task_id,)
            ) as cur:
                row = await cur.fetchone()
        return str(row["status_text"] or "") if row else ""

    async def save_draft(self, task_id: str, topic: str, plan: str) -> None:
        draft = {"topic": topic, "plan": plan}
        conn = await self._conn()
        async with self._lock:
            current = await self._load_task_meta_inner(conn, task_id)
            updated_at = now_iso()
            created_at = current.get("created_at") or updated_at
            await conn.execute(
                """
                INSERT INTO tasks(task_id, meta_json, draft_json, status_text, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    draft_json=excluded.draft_json,
                    updated_at=excluded.updated_at
                """,
                (
                    task_id,
                    json.dumps(current, ensure_ascii=False),
                    json.dumps(draft, ensure_ascii=False),
                    str(current.get("status_message") or ""),
                    created_at,
                    updated_at,
                ),
            )
            await conn.commit()

    async def load_draft(self, task_id: str) -> Dict[str, Any]:
        conn = await self._conn()
        async with self._lock:
            async with conn.execute(
                "SELECT draft_json FROM tasks WHERE task_id = ?", (task_id,)
            ) as cur:
                row = await cur.fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["draft_json"] or "{}")
        except Exception:
            return {}

    # ──────────────────────────────────────────────────────────────────
    #  Progress events (streaming-style)
    # ──────────────────────────────────────────────────────────────────

    async def append_progress_event(
        self,
        task_id: str,
        event_type: str,
        *,
        section_id: str = "",
        status: str = "",
        message: str = "",
        **extra: Any,
    ) -> None:
        """Store a structured progress event for real-time status reporting.

        Event types: plan_ready, section_start, section_done, search_query,
                     reflector_start, reflector_done, writer_start, writer_done
        """
        payload = {
            "type": event_type,
            "section_id": section_id,
            "status": status,
            "message": message,
            **extra,
        }
        conn = await self._conn()
        timestamp = now_iso()
        async with self._lock:
            await conn.execute(
                """
                INSERT INTO task_events(task_id, timestamp, stage, level, message, extra_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    timestamp,
                    "progress",
                    "info",
                    message,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            await conn.commit()

    async def get_progress_events(self, task_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent progress events for real-time status display."""
        conn = await self._conn()
        async with self._lock:
            async with conn.execute(
                """
                SELECT timestamp, extra_json
                FROM task_events
                WHERE task_id = ? AND stage = 'progress'
                ORDER BY id DESC
                LIMIT ?
                """,
                (task_id, limit),
            ) as cur:
                rows = await cur.fetchall()
        items: List[Dict[str, Any]] = []
        for row in reversed(rows):
            try:
                data = json.loads(row["extra_json"] or "{}")
            except Exception:
                data = {}
            data["timestamp"] = row["timestamp"]
            items.append(data)
        return items

    # ──────────────────────────────────────────────────────────────────
    #  Report version management
    # ──────────────────────────────────────────────────────────────────

    async def save_report_version(
        self, task_id: str, report_content: str, change_summary: str = ""
    ) -> int:
        """Save a new version of the report. Returns the version number."""
        conn = await self._conn()
        async with self._lock:
            async with conn.execute(
                "SELECT MAX(version) as max_v FROM report_versions WHERE task_id = ?",
                (task_id,),
            ) as cur:
                row = await cur.fetchone()
            next_version = (row["max_v"] or 0) + 1 if row else 1
            await conn.execute(
                """
                INSERT INTO report_versions(task_id, version, content, change_summary, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (task_id, next_version, report_content, change_summary, now_iso()),
            )
            await conn.commit()
            return next_version

    async def load_report_version(self, task_id: str, version: int = 0) -> Dict[str, Any]:
        """Load a specific report version. version=0 means latest."""
        conn = await self._conn()
        async with self._lock:
            if version <= 0:
                async with conn.execute(
                    "SELECT * FROM report_versions WHERE task_id = ? ORDER BY version DESC LIMIT 1",
                    (task_id,),
                ) as cur:
                    row = await cur.fetchone()
            else:
                async with conn.execute(
                    "SELECT * FROM report_versions WHERE task_id = ? AND version = ?",
                    (task_id, version),
                ) as cur:
                    row = await cur.fetchone()
        if not row:
            return {}
        return {
            "task_id": row["task_id"],
            "version": row["version"],
            "content": row["content"],
            "change_summary": row["change_summary"],
            "created_at": row["created_at"],
        }

    async def list_report_versions(self, task_id: str) -> List[Dict[str, Any]]:
        """List all versions of a report (without full content)."""
        conn = await self._conn()
        async with self._lock:
            async with conn.execute(
                "SELECT version, change_summary, created_at FROM report_versions WHERE task_id = ? ORDER BY version",
                (task_id,),
            ) as cur:
                rows = await cur.fetchall()
        return [
            {"version": r["version"], "change_summary": r["change_summary"], "created_at": r["created_at"]}
            for r in rows
        ]

    async def list_task_ids(self) -> List[str]:
        conn = await self._conn()
        async with self._lock:
            async with conn.execute(
                "SELECT task_id FROM tasks ORDER BY updated_at DESC, task_id DESC"
            ) as cur:
                rows = await cur.fetchall()
        return [str(row["task_id"]) for row in rows]

    async def list_tasks_with_meta(
        self, limit: int = 20, lifecycle: str = ""
    ) -> List[Dict[str, Any]]:
        """Return recent tasks with key summary fields.

        ``lifecycle`` filtering and the result columns are sourced from the
        mirrored ``tasks`` columns (lifecycle / stage / topic), so the query
        never has to parse ``meta_json`` for the common list/summary case.
        This is what makes the call cheap enough for an MCP tool that
        clients may poll.
        """
        conn = await self._conn()
        capped = max(1, int(limit))
        wanted = lifecycle.strip()
        params: tuple
        if wanted:
            sql = (
                "SELECT task_id, lifecycle, stage, topic, status_text, "
                "created_at, updated_at FROM tasks WHERE lifecycle = ? "
                "ORDER BY updated_at DESC, task_id DESC LIMIT ?"
            )
            params = (wanted, capped)
        else:
            sql = (
                "SELECT task_id, lifecycle, stage, topic, status_text, "
                "created_at, updated_at FROM tasks "
                "ORDER BY updated_at DESC, task_id DESC LIMIT ?"
            )
            params = (capped,)
        async with self._lock:
            async with conn.execute(sql, params) as cur:
                rows = await cur.fetchall()
        return [
            {
                "task_id": str(row["task_id"]),
                "topic": str(row["topic"] or ""),
                "lifecycle": str(row["lifecycle"] or ""),
                "stage": str(row["stage"] or ""),
                "status": str(row["status_text"] or ""),
                "created_at": str(row["created_at"] or ""),
                "updated_at": str(row["updated_at"] or ""),
            }
            for row in rows
        ]

    async def close(self) -> None:
        async with self._lock:
            if self._aconn is not None:
                try:
                    await self._aconn.close()
                except Exception:
                    pass
                self._aconn = None
