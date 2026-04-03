"""Async Jina reader with strict timeout and HTML fallback parsing."""

from __future__ import annotations

import asyncio
import html
import re

import aiohttp


class JinaReader:
    """Fetches markdown via r.jina.ai, with resilient fallback behavior."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def fetch_markdown(self, url: str) -> str:
        """Return markdown for a URL via Jina, falling back to HTML text extraction."""
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
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()

    @staticmethod
    def _build_jina_url(url: str) -> str:
        return f"https://r.jina.ai/{url}"

    @staticmethod
    def _validate_url(url: str) -> str:
        cleaned = url.strip()
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return cleaned

    @staticmethod
    def _strip_html(content: str) -> str:
        no_script = re.sub(r"<script.*?>.*?</script>", " ", content, flags=re.S | re.I)
        no_style = re.sub(r"<style.*?>.*?</style>", " ", no_script, flags=re.S | re.I)
        no_tags = re.sub(r"<[^>]+>", " ", no_style)
        normalized = re.sub(r"\s+", " ", html.unescape(no_tags)).strip()
        return normalized
