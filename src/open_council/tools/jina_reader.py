"""Async Jina reader with strict timeout and HTML fallback parsing."""

from __future__ import annotations

import asyncio
import html
import re

import aiohttp


class JinaReader:
    """
    Reader that fetches URL content through r.jina.ai with fallback extraction.

    It first requests markdown from Jina Reader. If that fails or returns empty,
    it falls back to direct HTML fetch and lightweight text extraction.
    """

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        """
        Configure reader timeout budget.

        Args:
            timeout_seconds: Hard request timeout used for Jina and fallback
                HTML fetch calls.
        """
        self.timeout_seconds = timeout_seconds

    async def fetch_markdown(self, url: str) -> str:
        """
        Fetch markdown from Jina Reader with robust fallback behavior.

        Args:
            url: Target HTTP(S) URL to read.

        Returns:
            Jina markdown when available, otherwise extracted plain text from
            raw HTML, or a safe `[reader_error]`/`[reader_warning]` message.
        """
        target_url = self._validate_url(url)
        jina_url = self._build_jina_url(target_url)

        try:
            markdown = await self._fetch_text(jina_url, timeout=self.timeout_seconds)
            if markdown.strip():
                return markdown
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return await self._fallback_from_html(target_url, str(exc))

        return await self._fallback_from_html(target_url, "Jina returned empty content.")

    async def _fallback_from_html(self, url: str, reason: str) -> str:
        """
        Attempt fallback extraction by requesting and stripping raw HTML.

        Args:
            url: Original source URL.
            reason: Description of why Jina path failed.

        Returns:
            Extracted text when possible, otherwise structured warning/error
            strings suitable for graph-state propagation.
        """
        try:
            html_content = await self._fetch_text(url, timeout=self.timeout_seconds)
            extracted = self._strip_html(html_content)
            if extracted:
                return extracted
            return f"[reader_warning] Empty HTML fallback for {url}. Initial failure: {reason}"
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return (
                f"[reader_error] Could not read {url}. "
                f"Jina failed: {reason}. HTML fallback failed: {exc}"
            )

    async def _fetch_text(self, url: str, *, timeout: float) -> str:
        """
        Execute one HTTP GET and return response body as text.

        Args:
            url: Absolute URL to fetch.
            timeout: Request timeout in seconds.

        Returns:
            Response text body.

        Raises:
            aiohttp.ClientError: On transport/HTTP failures.
            asyncio.TimeoutError: On timeout.
        """
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()

    @staticmethod
    def _build_jina_url(url: str) -> str:
        """
        Compose Jina Reader URL for a target source URL.

        Args:
            url: Original source URL.

        Returns:
            Jina Reader request URL.
        """
        return f"https://r.jina.ai/{url}"

    @staticmethod
    def _validate_url(url: str) -> str:
        """
        Validate supported URL schemes.

        Args:
            url: Candidate URL string.

        Returns:
            Trimmed URL.

        Raises:
            ValueError: If URL does not start with http:// or https://.
        """
        cleaned = url.strip()
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return cleaned

    @staticmethod
    def _strip_html(content: str) -> str:
        """
        Convert HTML content into compact plain text.

        Args:
            content: Raw HTML string.

        Returns:
            Normalized plain text with scripts/styles/tags removed.
        """
        no_script = re.sub(r"<script.*?>.*?</script>", " ", content, flags=re.S | re.I)
        no_style = re.sub(r"<style.*?>.*?</style>", " ", no_script, flags=re.S | re.I)
        no_tags = re.sub(r"<[^>]+>", " ", no_style)
        normalized = re.sub(r"\s+", " ", html.unescape(no_tags)).strip()
        return normalized
