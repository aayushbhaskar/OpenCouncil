"""Async base interfaces for search providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class SearchResult:
    """Normalized search result returned by all providers."""

    title: str
    url: str
    snippet: str = ""
    source: str = ""


class BaseSearchProvider(ABC):
    """Abstract async search provider contract."""

    provider_name: str

    @abstractmethod
    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        """Run async search and return normalized result objects."""

    def _validate_query(self, query: str) -> str:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Search query cannot be empty.")
        return cleaned
