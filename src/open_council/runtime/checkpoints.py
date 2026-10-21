"""SQLite LangGraph checkpointer helpers and thread history."""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from open_council.cli.constants import GLOBAL_CONFIG_DIR

DEFAULT_CHECKPOINT_PATH = GLOBAL_CONFIG_DIR / "checkpoints.sqlite"


@dataclass(slots=True)
class ThreadSummary:
    thread_id: str
    mode: str
    updated_at: float


def checkpoint_db_path() -> Path:
    path = DEFAULT_CHECKPOINT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def new_thread_id() -> str:
    return str(uuid.uuid4())


def ensure_thread_id(raw: str | None) -> str:
    cleaned = (raw or "").strip()
    return cleaned or new_thread_id()


def _metadata_connect(path: Path | None = None) -> sqlite3.Connection:
    db_path = path or checkpoint_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS open_council_threads (
            thread_id TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def touch_thread(*, thread_id: str, mode: str, path: Path | None = None) -> None:
    try:
        with _metadata_connect(path) as conn:
            conn.execute(
                """
                INSERT INTO open_council_threads (thread_id, mode, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    mode=excluded.mode,
                    updated_at=excluded.updated_at
                """,
                (thread_id, mode, time.time()),
            )
            conn.commit()
    except (OSError, sqlite3.Error):
        return


def list_threads(*, path: Path | None = None, limit: int = 50) -> list[ThreadSummary]:
    try:
        with _metadata_connect(path) as conn:
            rows = conn.execute(
                """
                SELECT thread_id, mode, updated_at
                FROM open_council_threads
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
    except (OSError, sqlite3.Error):
        return []
    return [
        ThreadSummary(thread_id=str(row[0]), mode=str(row[1]), updated_at=float(row[2]))
        for row in rows
    ]


@asynccontextmanager
async def open_async_sqlite_saver(path: Path | None = None) -> AsyncIterator[Any]:
    """
    Yield a LangGraph AsyncSqliteSaver bound to the global checkpoint DB.

    Graphs run via `astream`/`ainvoke`, which require the async saver. The saver is
    tied to the running event loop, so open it inside each `asyncio.run` turn.
    Yields `None` when the database cannot be opened (best-effort persistence).
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = path or DEFAULT_CHECKPOINT_PATH
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(db_path))
    except (OSError, sqlite3.Error):
        yield None
        return
    try:
        yield AsyncSqliteSaver(conn)
    finally:
        await conn.close()
