"""Shared asyncio throttling utilities for outbound network calls."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class AsyncThrottle:
    """Limits concurrent async operations via an asyncio.Semaphore."""

    def __init__(self, max_concurrent: int = 2) -> None:
        self.max_concurrent = max(1, max_concurrent)
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def run(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Execute an async callable while honoring the shared semaphore."""
        async with self._semaphore:
            return await operation()


def _load_max_concurrent() -> int:
    raw_value = os.getenv("MAX_CONCURRENT_REQUESTS", "2").strip()
    try:
        parsed = int(raw_value)
    except ValueError:
        return 2
    return max(1, parsed)


network_throttle = AsyncThrottle(max_concurrent=_load_max_concurrent())
