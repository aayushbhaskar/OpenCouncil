"""Tavily Tier-2 search provider."""

from __future__ import annotations

import os

import aiohttp

from open_council.tools.search_base import BaseSearchProvider, SearchResult


class TavilySearchProvider(BaseSearchProvider):
    """Async Tavily search conforming to `BaseSearchProvider`."""

    provider_name = "tavily"
    api_url = "https://api.tavily.com/search"

    def __init__(self, *, api_key: str | None = None, timeout_seconds: float = 15.0) -> None:
        self.api_key = (api_key or os.getenv("TAVILY_API_KEY", "")).strip()
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        cleaned = self._validate_query(query)
        if not self.api_key:
            return [self._error_result("Tavily API key missing.")]

        payload = {
            "api_key": self.api_key,
            "query": cleaned,
            "max_results": max(1, max_results),
            "include_answer": False,
        }
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.api_url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
        except Exception as exc:  # noqa: BLE001
            return [self._error_result(f"Tavily search failed: {exc}")]

        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return [self._error_result("Tavily returned unexpected payload.")]

        normalized: list[SearchResult] = []
        for item in results[: max(1, max_results)]:
            if not isinstance(item, dict):
                continue
            normalized.append(
                SearchResult(
                    title=str(item.get("title") or "Untitled").strip() or "Untitled",
                    url=str(item.get("url") or "").strip(),
                    snippet=str(item.get("content") or item.get("snippet") or "").strip(),
                    source=self.provider_name,
                )
            )
        return normalized or [self._error_result("Tavily returned no results.")]

    def _error_result(self, message: str) -> SearchResult:
        return SearchResult(
            title="Search unavailable",
            url="",
            snippet=message,
            source=self.provider_name,
        )
