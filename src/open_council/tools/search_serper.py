"""Serper Tier-2 search provider."""

from __future__ import annotations

import os

import aiohttp

from open_council.tools.search_base import BaseSearchProvider, SearchResult


class SerperSearchProvider(BaseSearchProvider):
    """Async Serper.dev Google search conforming to `BaseSearchProvider`."""

    provider_name = "serper"
    api_url = "https://google.serper.dev/search"

    def __init__(self, *, api_key: str | None = None, timeout_seconds: float = 15.0) -> None:
        self.api_key = (api_key or os.getenv("SERPER_API_KEY", "")).strip()
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        cleaned = self._validate_query(query)
        if not self.api_key:
            return [self._error_result("Serper API key missing.")]

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": cleaned, "num": max(1, max_results)}
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
        except Exception as exc:  # noqa: BLE001
            return [self._error_result(f"Serper search failed: {exc}")]

        organic = data.get("organic") if isinstance(data, dict) else None
        if not isinstance(organic, list):
            return [self._error_result("Serper returned unexpected payload.")]

        normalized: list[SearchResult] = []
        for item in organic[: max(1, max_results)]:
            if not isinstance(item, dict):
                continue
            normalized.append(
                SearchResult(
                    title=str(item.get("title") or "Untitled").strip() or "Untitled",
                    url=str(item.get("link") or item.get("url") or "").strip(),
                    snippet=str(item.get("snippet") or "").strip(),
                    source=self.provider_name,
                )
            )
        return normalized or [self._error_result("Serper returned no results.")]

    def _error_result(self, message: str) -> SearchResult:
        return SearchResult(
            title="Search unavailable",
            url="",
            snippet=message,
            source=self.provider_name,
        )
