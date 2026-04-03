from __future__ import annotations

import asyncio

import pytest
from duckduckgo_search.exceptions import RatelimitException

from open_council.tools.jina_reader import JinaReader
from open_council.tools.search_ddg import DuckDuckGoSearchProvider


@pytest.mark.asyncio
async def test_ddg_provider_normalizes_results(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = DuckDuckGoSearchProvider()

    def _stub_sync_search(query: str, max_results: int) -> list[dict[str, str]]:
        assert query == "open council"
        assert max_results == 3
        return [
            {
                "title": "Open Council",
                "href": "https://example.com",
                "body": "A CLI-first multi-agent tool.",
            }
        ]

    monkeypatch.setattr(provider, "_search_sync", _stub_sync_search)

    results = await provider.search("open council", max_results=3)

    assert len(results) == 1
    assert results[0].title == "Open Council"
    assert results[0].url == "https://example.com"
    assert results[0].source == "duckduckgo"


@pytest.mark.asyncio
async def test_ddg_provider_returns_safe_error_on_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = DuckDuckGoSearchProvider()

    def _stub_sync_search(_: str, __: int) -> list[dict[str, str]]:
        raise RatelimitException("rate limit reached")

    monkeypatch.setattr(provider, "_search_sync", _stub_sync_search)

    results = await provider.search("open council")

    assert len(results) == 1
    assert results[0].title == "Search unavailable"
    assert "rate limited" in results[0].snippet


@pytest.mark.asyncio
async def test_jina_reader_returns_jina_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = JinaReader(timeout_seconds=10.0)

    async def _stub_fetch_text(url: str, *, timeout: float) -> str:
        assert timeout == 10.0
        assert url.startswith("https://r.jina.ai/")
        return "# markdown content"

    monkeypatch.setattr(reader, "_fetch_text", _stub_fetch_text)

    output = await reader.fetch_markdown("https://example.com")

    assert output == "# markdown content"


@pytest.mark.asyncio
async def test_jina_reader_falls_back_to_html_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = JinaReader(timeout_seconds=10.0)

    async def _stub_fetch_text(url: str, *, timeout: float) -> str:
        _ = timeout
        if url.startswith("https://r.jina.ai/"):
            raise asyncio.TimeoutError()
        return "<html><body><h1>Title</h1><p>Hello world</p></body></html>"

    monkeypatch.setattr(reader, "_fetch_text", _stub_fetch_text)

    output = await reader.fetch_markdown("https://example.com")

    assert "Title Hello world" in output


@pytest.mark.asyncio
async def test_jina_reader_returns_error_text_when_both_paths_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = JinaReader(timeout_seconds=10.0)

    async def _stub_fetch_text(_: str, *, timeout: float) -> str:
        _ = timeout
        raise asyncio.TimeoutError("timed out")

    monkeypatch.setattr(reader, "_fetch_text", _stub_fetch_text)

    output = await reader.fetch_markdown("https://example.com")

    assert output.startswith("[reader_error]")
