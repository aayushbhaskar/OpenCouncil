"""Shared asyncio throttling utilities for outbound network calls."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class AsyncThrottle:
    """
    Gatekeeper for concurrent async operations.

    This class wraps an `asyncio.Semaphore` and provides a single execution
    method that runs a coroutine-producing callable inside the semaphore
    boundary. It is used to enforce global outbound request limits and reduce
    API rate-limit spikes when multiple graph nodes run in parallel.
    """

    def __init__(self, max_concurrent: int = 2) -> None:
        """
        Initialize the throttle with a bounded concurrency level.

        Args:
            max_concurrent: Desired parallel operation ceiling.
                Values lower than 1 are clamped to 1.
        """
        self.max_concurrent = max(1, max_concurrent)
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def run(self, operation: Callable[[], Awaitable[T]]) -> T:
        """
        Execute an async operation while honoring the throttle limit.

        Args:
            operation: Zero-argument callable that returns an awaitable.

        Returns:
            The awaited result from `operation`.
        """
        async with self._semaphore:
            return await operation()


def _load_max_concurrent() -> int:
    """
    Read and sanitize concurrency limit from environment.

    Returns:
        An integer >= 1 loaded from `MAX_CONCURRENT_REQUESTS`, or `2` when the
        environment value is missing/invalid.
    """
    raw_value = os.getenv("MAX_CONCURRENT_REQUESTS", "2").strip()
    try:
        parsed = int(raw_value)
    except ValueError:
        return 2
    return max(1, parsed)


network_throttle = AsyncThrottle(max_concurrent=_load_max_concurrent())
