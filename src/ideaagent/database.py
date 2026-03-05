"""SQLite-backed database for IdeaAgent.

Replaces the previous MongoDB dependency with the Python standard-library
``sqlite3`` module.  No external service is required; the database file is
stored at ``~/.ideaagent/tasks.db`` by default (configurable via
``DB_PATH`` in ``.env``).
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import settings, ensure_directories
from .exceptions import DatabaseError
from .models import Task, TaskStatus

logger = logging.getLogger("IdeaAgent.database")

_CREATE_TASKS_TABLE = """
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    status      TEXT NOT NULL,
    research_type TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_research_type ON tasks (research_type)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks (created_at)",
]


class Database:
    """SQLite database manager for IdeaAgent.

    The class exposes the same public API as the old MongoDB-based
    ``Database`` so that the rest of the code requires no changes.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialise the database, creating the file and schema if needed.

        Args:
            db_path: Path to the SQLite database file.  Defaults to
                ``settings.db_path`` (``~/.ideaagent/tasks.db``).
        """
        ensure_directories()
        self._db_path: Path = Path(db_path) if db_path else settings.db_path
        self._connected: bool = False
        self._init_db()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Return a new SQLite connection with row-factory set."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the schema if it does not exist yet."""
        try:
            with self._get_conn() as conn:
                conn.execute(_CREATE_TASKS_TABLE)
                for idx_sql in _CREATE_INDEXES:
                    conn.execute(idx_sql)
                conn.commit()
            self._connected = True
            logger.debug("SQLite database initialised at %s", self._db_path)
        except Exception as exc:
            logger.warning("Database initialisation failed: %s", exc)
            self._connected = False

    # ── Public API ────────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        """Return ``True`` if the database is accessible."""
        if not self._connected:
            return False
        try:
            with self._get_conn() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def save_task(self, task: Task) -> bool:
        """Insert or update a task.

        Args:
            task: :class:`~ideaagent.models.Task` to persist.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        try:
            task.updated_at = datetime.now()
            data_json = json.dumps(task.to_dict())
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO tasks (id, data, status, research_type, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        data       = excluded.data,
                        status     = excluded.status,
                        updated_at = excluded.updated_at
                    """,
                    (
                        task.id,
                        data_json,
                        task.status.value,
                        task.research_type.value,
                        task.created_at.isoformat(),
                        task.updated_at.isoformat(),
                    ),
                )
                conn.commit()
            return True
        except Exception as exc:
            logger.error("Error saving task %s: %s", task.id, exc)
            return False

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by its ID.

        Args:
            task_id: Unique task identifier.

        Returns:
            :class:`~ideaagent.models.Task` if found, ``None`` otherwise.
        """
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT data FROM tasks WHERE id = ?", (task_id,)
                ).fetchone()
            if row:
                return Task.from_dict(json.loads(row["data"]))
            return None
        except Exception as exc:
            logger.error("Error getting task %s: %s", task_id, exc)
            return None

    def get_tasks_by_status(self, status: TaskStatus) -> list[Task]:
        """Return all tasks with a given status.

        Args:
            status: :class:`~ideaagent.models.TaskStatus` to filter by.

        Returns:
            List of matching :class:`~ideaagent.models.Task` objects.
        """
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT data FROM tasks WHERE status = ? ORDER BY created_at DESC",
                    (status.value,),
                ).fetchall()
            return [Task.from_dict(json.loads(r["data"])) for r in rows]
        except Exception as exc:
            logger.error("Error getting tasks by status %s: %s", status, exc)
            return []

    def get_all_tasks(self, limit: int = 100) -> list[Task]:
        """Return the most recent tasks.

        Args:
            limit: Maximum number of tasks to return.

        Returns:
            List of :class:`~ideaagent.models.Task` objects, newest first.
        """
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT data FROM tasks ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [Task.from_dict(json.loads(r["data"])) for r in rows]
        except Exception as exc:
            logger.error("Error getting all tasks: %s", exc)
            return []

    def delete_task(self, task_id: str) -> bool:
        """Delete a task by ID.

        Args:
            task_id: Unique task identifier.

        Returns:
            ``True`` if a row was deleted, ``False`` otherwise.
        """
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "DELETE FROM tasks WHERE id = ?", (task_id,)
                )
                conn.commit()
            return cursor.rowcount > 0
        except Exception as exc:
            logger.error("Error deleting task %s: %s", task_id, exc)
            return False

    def get_task_statistics(self) -> dict:
        """Return aggregate statistics about stored tasks.

        Returns:
            Dictionary with keys ``total``, ``by_status``, and
            ``by_research_type``.
        """
        try:
            with self._get_conn() as conn:
                total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

                by_status_rows = conn.execute(
                    "SELECT status, COUNT(*) as cnt FROM tasks GROUP BY status"
                ).fetchall()
                by_status = {r["status"]: r["cnt"] for r in by_status_rows}

                by_type_rows = conn.execute(
                    "SELECT research_type, COUNT(*) as cnt FROM tasks GROUP BY research_type"
                ).fetchall()
                by_type = {r["research_type"]: r["cnt"] for r in by_type_rows}

            return {"total": total, "by_status": by_status, "by_research_type": by_type}
        except Exception as exc:
            logger.error("Error getting statistics: %s", exc)
            return {"total": 0, "by_status": {}, "by_research_type": {}}

    def close(self) -> None:
        """No-op: SQLite connections are opened/closed per-operation."""
        pass
