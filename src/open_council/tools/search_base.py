"""Async base interfaces for search providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class SearchResult:
    """
    Normalized search result returned by all providers.

    Attributes:
        title: Human-readable result title.
        url: Result URL.
        snippet: Short descriptive excerpt.
        source: Provider label that produced this item.
    """

    title: str
    url: str
    snippet: str = ""
    source: str = ""


class BaseSearchProvider(ABC):
    """
    Abstract async search provider contract.

    Concrete implementations must normalize provider-native responses into
    `SearchResult` objects and keep network behavior asynchronous.
    """

    provider_name: str

    @abstractmethod
    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        """
        Execute provider search asynchronously.

        Args:
            query: Search query string.
            max_results: Max number of results to return.

        Returns:
            Provider-normalized list of `SearchResult` items.
        """

    def _validate_query(self, query: str) -> str:
        """
        Validate and normalize raw query input.

        Args:
            query: Raw search text provided by caller.

        Returns:
            Trimmed query text.

        Raises:
            ValueError: If the query is empty/whitespace.
        """
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Search query cannot be empty.")
        return cleaned
