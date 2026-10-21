"""SQLite cache for search listings and URL extraction content."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from open_council.cli.constants import GLOBAL_CONFIG_DIR
from open_council.tools.search_base import SearchResult

DEFAULT_CACHE_PATH = GLOBAL_CONFIG_DIR / "search_cache.sqlite"


class SearchCache:
    """
    Persistent SQLite cache for search query results and extracted page text.

    Keys:
      - search listings: sha256(provider + normalized query + max_results)
      - page content: normalized URL
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_CACHE_PATH
        self._enabled = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
        except (OSError, sqlite3.Error):
            # Cache is best-effort; never break graph execution if disk is unavailable.
            self._enabled = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS search_listings (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    query TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS page_content (
                    url TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def search_key(provider: str, query: str, *, max_results: int) -> str:
        raw = f"{provider.strip().lower()}|{query.strip().lower()}|{max_results}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_search_results(self, provider: str, query: str, *, max_results: int) -> list[SearchResult] | None:
        if not self._enabled:
            return None
        key = self.search_key(provider, query, max_results=max_results)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM search_listings WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        return [
            SearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("snippet", "")),
                source=str(item.get("source", provider)),
            )
            for item in payload
        ]

    def put_search_results(
        self,
        provider: str,
        query: str,
        *,
        max_results: int,
        results: list[SearchResult],
    ) -> None:
        if not self._enabled:
            return
        key = self.search_key(provider, query, max_results=max_results)
        payload = json.dumps(
            [
                {
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet,
                    "source": item.source,
                }
                for item in results
            ]
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO search_listings (cache_key, provider, query, payload, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (key, provider, query, payload, time.time()),
            )
            conn.commit()

    def get_page_content(self, url: str) -> str | None:
        if not self._enabled:
            return None
        normalized = url.strip()
        if not normalized:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM page_content WHERE url = ?",
                (normalized,),
            ).fetchone()
        if row is None:
            return None
        return str(row["content"])

    def put_page_content(self, url: str, content: str) -> None:
        if not self._enabled:
            return
        normalized = url.strip()
        if not normalized:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO page_content (url, content, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    content=excluded.content,
                    updated_at=excluded.updated_at
                """,
                (normalized, content, time.time()),
            )
            conn.commit()
