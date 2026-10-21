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
    title: str = ""


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
            updated_at REAL NOT NULL,
            title TEXT
        )
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(open_council_threads)")}
    if "title" not in columns:
        conn.execute("ALTER TABLE open_council_threads ADD COLUMN title TEXT")
    conn.commit()
    return conn


def touch_thread(
    *,
    thread_id: str,
    mode: str,
    title: str | None = None,
    path: Path | None = None,
) -> None:
    """Upsert a thread registry row; `title` is kept from the first turn that sets it."""
    cleaned_title = " ".join((title or "").split())[:120] or None
    try:
        with _metadata_connect(path) as conn:
            conn.execute(
                """
                INSERT INTO open_council_threads (thread_id, mode, updated_at, title)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    mode=excluded.mode,
                    updated_at=excluded.updated_at,
                    title=COALESCE(open_council_threads.title, excluded.title)
                """,
                (thread_id, mode, time.time(), cleaned_title),
            )
            conn.commit()
    except (OSError, sqlite3.Error):
        return


def get_thread_mode(thread_id: str, *, path: Path | None = None) -> str | None:
    """Return the mode a thread was created in, or None when the thread is unknown."""
    try:
        with _metadata_connect(path) as conn:
            row = conn.execute(
                "SELECT mode FROM open_council_threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
    except (OSError, sqlite3.Error):
        return None
    return str(row[0]) if row else None


def list_threads(*, path: Path | None = None, limit: int = 50) -> list[ThreadSummary]:
    try:
        with _metadata_connect(path) as conn:
            rows = conn.execute(
                """
                SELECT thread_id, mode, updated_at, title
                FROM open_council_threads
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
    except (OSError, sqlite3.Error):
        return []
    return [
        ThreadSummary(
            thread_id=str(row[0]),
            mode=str(row[1]),
            updated_at=float(row[2]),
            title=str(row[3] or ""),
        )
        for row in rows
    ]


def restore_chat_history(values: dict[str, Any]) -> list[dict[str, str]]:
    """
    Rebuild the conversation from a thread's latest checkpoint values.

    The REPL appends the final answer after the graph finishes, so a completed turn's
    checkpoint ends with the user's question and holds the answer in `final_synthesis`.
    A trailing question without an answer (cancelled turn) is dropped.
    """
    history = [dict(message) for message in values.get("chat_history", []) or []]
    if not history or history[-1].get("role") != "user":
        return history
    answer = str(values.get("final_synthesis", "") or "").strip()
    if answer:
        history.append({"role": "assistant", "content": answer})
    else:
        history.pop()
    return history


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
