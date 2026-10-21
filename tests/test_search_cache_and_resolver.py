from __future__ import annotations

import pytest

from open_council.tools.search_base import SearchResult
from open_council.tools.search_cache import SearchCache
from open_council.tools.search_resolver import CachedSearchProvider, resolve_search_provider


def test_search_cache_round_trips_listings_and_pages(tmp_path) -> None:
    cache = SearchCache(path=tmp_path / "cache.sqlite")
    results = [
        SearchResult(title="A", url="https://a.example", snippet="sa", source="duckduckgo"),
        SearchResult(title="B", url="https://b.example", snippet="sb", source="duckduckgo"),
    ]
    cache.put_search_results("duckduckgo", "q", max_results=5, results=results)
    loaded = cache.get_search_results("duckduckgo", "q", max_results=5)
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].url == "https://a.example"

    cache.put_page_content("https://a.example", "page body")
    assert cache.get_page_content("https://a.example") == "page body"
    assert cache.get_page_content("https://missing.example") is None


@pytest.mark.asyncio
async def test_cached_search_provider_skips_inner_on_hit(tmp_path) -> None:
    cache = SearchCache(path=tmp_path / "cache.sqlite")
    calls = {"n": 0}

    class _Inner:
        provider_name = "fake"

        async def search(self, query: str, *, max_results: int = 5):
            calls["n"] += 1
            return [
                SearchResult(
                    title="T",
                    url="https://example.com",
                    snippet=query,
                    source="fake",
                )
            ]

    provider = CachedSearchProvider(inner=_Inner(), cache=cache)
    first = await provider.search("hello", max_results=3)
    second = await provider.search("hello", max_results=3)
    assert first[0].url == "https://example.com"
    assert second[0].url == "https://example.com"
    assert calls["n"] == 1


def test_resolve_search_provider_prefers_tavily(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-real-key")
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    provider = resolve_search_provider(use_cache=True, cache_path=tmp_path / "c.sqlite")
    assert provider.provider_name == "tavily"


def test_resolve_search_provider_prefers_serper_when_no_tavily(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SERPER_API_KEY", "serper-real-key")
    provider = resolve_search_provider(use_cache=False)
    assert provider.provider_name == "serper"


def test_resolve_search_provider_falls_back_to_ddg(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    provider = resolve_search_provider(use_cache=False)
    assert provider.provider_name == "duckduckgo"
