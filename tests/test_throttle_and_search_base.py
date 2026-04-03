from __future__ import annotations

import asyncio

import pytest

from open_council.core.throttle import AsyncThrottle
from open_council.tools.search_base import BaseSearchProvider, SearchResult


class DummySearchProvider(BaseSearchProvider):
    provider_name = "dummy"

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        cleaned = self._validate_query(query)
        return [
            SearchResult(
                title=f"Result for {cleaned}",
                url="https://example.com",
                snippet="demo",
                source=self.provider_name,
            )
        ][:max_results]


@pytest.mark.asyncio
async def test_async_throttle_limits_parallelism() -> None:
    throttle = AsyncThrottle(max_concurrent=1)
    running = 0
    max_running = 0
    lock = asyncio.Lock()

    async def _operation(value: int) -> int:
        nonlocal running, max_running
        async with lock:
            running += 1
            max_running = max(max_running, running)
        await asyncio.sleep(0.02)
        async with lock:
            running -= 1
        return value

    tasks = [throttle.run(lambda v=v: _operation(v)) for v in range(5)]
    results = await asyncio.gather(*tasks)

    assert sorted(results) == [0, 1, 2, 3, 4]
    assert max_running == 1


@pytest.mark.asyncio
async def test_search_base_normalized_result_shape() -> None:
    provider = DummySearchProvider()
    results = await provider.search("langgraph")

    assert len(results) == 1
    assert results[0].source == "dummy"
    assert results[0].url == "https://example.com"


def test_search_base_rejects_empty_query() -> None:
    provider = DummySearchProvider()
    with pytest.raises(ValueError):
        provider._validate_query("   ")
