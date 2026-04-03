"""DuckDuckGo search provider implementation."""

from __future__ import annotations

import asyncio
from typing import Any

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import (
    DuckDuckGoSearchException,
    RatelimitException,
    TimeoutException,
)

from open_council.tools.search_base import BaseSearchProvider, SearchResult


class DuckDuckGoSearchProvider(BaseSearchProvider):
    """Tier-1 search implementation using `duckduckgo-search`."""

    provider_name = "duckduckgo"

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        cleaned_query = self._validate_query(query)
        bounded_results = max(1, max_results)

        try:
            raw_results = await asyncio.to_thread(
                self._search_sync, cleaned_query, bounded_results
            )
        except RatelimitException as exc:
            return [self._error_result(f"DuckDuckGo rate limited: {exc}")]
        except (TimeoutException, DuckDuckGoSearchException) as exc:
            return [self._error_result(f"DuckDuckGo search failed: {exc}")]
        except Exception as exc:  # noqa: BLE001
            return [self._error_result(f"Unexpected search error: {exc}")]

        return [self._normalize_result(item) for item in raw_results]

    def _search_sync(self, query: str, max_results: int) -> list[dict[str, Any]]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    def _normalize_result(self, result: dict[str, Any]) -> SearchResult:
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
        return SearchResult(
            title="Search unavailable",
            url="",
            snippet=message,
            source=self.provider_name,
        )
