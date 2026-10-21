"""Resolve active search provider and optional caching wrapper."""

from __future__ import annotations

import os
from pathlib import Path

from open_council.core.fake import fake_mode_enabled, fake_search_results
from open_council.runtime.provider_readiness import has_real_api_key
from open_council.tools.search_base import BaseSearchProvider, SearchResult
from open_council.tools.search_cache import SearchCache
from open_council.tools.search_ddg import DuckDuckGoSearchProvider
from open_council.tools.search_serper import SerperSearchProvider
from open_council.tools.search_tavily import TavilySearchProvider


class CachedSearchProvider(BaseSearchProvider):
    """Decorator that caches search listings by provider+query."""

    def __init__(self, inner: BaseSearchProvider, cache: SearchCache | None = None) -> None:
        self.inner = inner
        self.cache = cache or SearchCache()
        self.provider_name = inner.provider_name

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        cleaned = self._validate_query(query)
        cached = self.cache.get_search_results(
            self.provider_name,
            cleaned,
            max_results=max_results,
        )
        if cached is not None:
            return cached
        results = await self.inner.search(cleaned, max_results=max_results)
        # Do not cache pure error payloads with empty URLs.
        if results and any(item.url for item in results):
            self.cache.put_search_results(
                self.provider_name,
                cleaned,
                max_results=max_results,
                results=results,
            )
        return results


class FakeSearchProvider(BaseSearchProvider):
    """Deterministic offline provider used when fake mode is enabled."""

    provider_name = "fake"

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        cleaned = self._validate_query(query)
        return [
            SearchResult(title=title, url=url, snippet=snippet, source=self.provider_name)
            for title, url, snippet in fake_search_results(cleaned, max_results)
        ]


def resolve_search_provider(*, use_cache: bool = True, cache_path: Path | None = None) -> BaseSearchProvider:
    """
    Pick Tier-2 provider when keys exist, otherwise DuckDuckGo.

    Order: Tavily → Serper → DuckDuckGo. Fake mode bypasses all providers and the cache.
    """
    if fake_mode_enabled():
        return FakeSearchProvider()
    if has_real_api_key(os.getenv("TAVILY_API_KEY", "")):
        inner: BaseSearchProvider = TavilySearchProvider()
    elif has_real_api_key(os.getenv("SERPER_API_KEY", "")):
        inner = SerperSearchProvider()
    else:
        inner = DuckDuckGoSearchProvider()

    if not use_cache:
        return inner
    cache = SearchCache(path=cache_path) if cache_path is not None else SearchCache()
    return CachedSearchProvider(inner=inner, cache=cache)
