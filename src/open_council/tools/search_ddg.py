"""DuckDuckGo search provider implementation."""

from __future__ import annotations

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import io
import os
from typing import Any

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import (
    DuckDuckGoSearchException,
    RatelimitException,
    TimeoutException,
)

from open_council.tools.search_base import BaseSearchProvider, SearchResult


class DuckDuckGoSearchProvider(BaseSearchProvider):
    """
    Tier-1 search implementation using `duckduckgo-search`.

    This provider normalizes raw DDG results and converts exceptions into
    safe `SearchResult` error items rather than raising in graph flows.
    """

    provider_name = "duckduckgo"
    search_timeout_seconds = 8.0

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        """
        Perform a DuckDuckGo text search in a non-blocking way.

        Args:
            query: User search query.
            max_results: Maximum number of normalized results to return.

        Returns:
            List of normalized `SearchResult` items. On provider failures, a
            single error `SearchResult` is returned with details in `snippet`.
        """
        cleaned_query = self._validate_query(query)
        bounded_results = max(1, max_results)

        try:
            raw_results = await asyncio.to_thread(
                self._search_sync_with_timeout, cleaned_query, bounded_results
            )
        except FutureTimeoutError as exc:
            return [self._error_result(f"DuckDuckGo search timed out: {exc}")]
        except RatelimitException as exc:
            return [self._error_result(f"DuckDuckGo rate limited: {exc}")]
        except (TimeoutException, DuckDuckGoSearchException) as exc:
            return [self._error_result(f"DuckDuckGo search failed: {exc}")]
        except Exception as exc:  # noqa: BLE001
            return [self._error_result(f"Unexpected search error: {exc}")]

        return [self._normalize_result(item) for item in raw_results]

    def _search_sync_with_timeout(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """
        Run sync search in an isolated worker thread with hard timeout.

        This prevents indefinite blocking in DDG internals from freezing the
        event loop turn.
        """
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._search_sync, query, max_results)
            return future.result(timeout=self.search_timeout_seconds)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _search_sync(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """
        Execute synchronous DDG search call.

        Args:
            query: Validated query text.
            max_results: Result limit.

        Returns:
            Raw DDG result dictionaries.
        """
        backend = os.getenv("OPEN_COUNCIL_DDG_BACKEND", "lite").strip() or "lite"
        timeout_seconds = max(1, int(self.search_timeout_seconds))
        # Suppress noisy provider impersonation warnings from leaking into CLI UI.
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            with DDGS(timeout=timeout_seconds) as ddgs:
                return list(ddgs.text(query, backend=backend, max_results=max_results))

    def _normalize_result(self, result: dict[str, Any]) -> SearchResult:
        """
        Map one provider-native DDG item to `SearchResult`.

        Args:
            result: Raw DDG result payload.

        Returns:
            Normalized `SearchResult`.
        """
        title = str(result.get("title") or "").strip() or "Untitled"
        url = str(result.get("href") or result.get("url") or "").strip()
        snippet = str(result.get("body") or result.get("snippet") or "").strip()
        return SearchResult(
            title=title,
            url=url,
            snippet=snippet,
            source=self.provider_name,
        )

    def _error_result(self, message: str) -> SearchResult:
        """
        Build standardized error result item for graceful degradation.

        Args:
            message: Human-readable provider failure message.

        Returns:
            Error-flavored `SearchResult` suitable for graph state usage.
        """
        return SearchResult(
            title="Search unavailable",
            url="",
            snippet=message,
            source=self.provider_name,
        )
