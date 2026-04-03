"""Async LiteLLM wrapper with ordered provider fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from litellm import acompletion


@dataclass(slots=True)
class LLMAttempt:
    """One provider attempt record for observability/debugging."""

    provider: str
    model: str
    error: str | None = None


@dataclass(slots=True)
class LLMResult:
    """Normalized response consumed by graph nodes."""

    ok: bool
    content: str
    provider: str | None
    model: str | None
    attempts: list[LLMAttempt]
    error: str | None = None
    raw_response: Any | None = None


class LiteLLMClient:
    """Defensive async wrapper around `litellm.acompletion`."""

    def __init__(self) -> None:
        self.timeout_seconds = float(os.getenv("LITELLM_TIMEOUT_SECONDS", "30"))
        self.provider_models = [
            ("groq", os.getenv("GROQ_MODEL", "groq/llama-3.1-70b-versatile")),
            ("gemini", os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")),
            ("ollama", os.getenv("OLLAMA_MODEL", "ollama/llama3.1")),
        ]

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Call model completion with strict Groq -> Gemini -> Ollama fallback."""
        attempts: list[LLMAttempt] = []

        for provider, model in self.provider_models:
            try:
                response = await acompletion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout_seconds,
                    **self._provider_kwargs(provider),
                )
                content = self._extract_content(response)
                attempts.append(LLMAttempt(provider=provider, model=model))
                return LLMResult(
                    ok=True,
                    content=content,
                    provider=provider,
                    model=model,
                    attempts=attempts,
                    raw_response=response,
                )
            except Exception as exc:  # noqa: BLE001
                attempts.append(LLMAttempt(provider=provider, model=model, error=str(exc)))

        return LLMResult(
            ok=False,
            content="",
            provider=None,
            model=None,
            attempts=attempts,
            error="All fallback providers failed (Groq -> Gemini -> Ollama).",
        )

    def _provider_kwargs(self, provider: str) -> dict[str, Any]:
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            return {"api_key": api_key} if api_key else {}
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            return {"api_key": api_key} if api_key else {}
        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
            return {"api_base": base_url}
        return {}

    @staticmethod
    def _extract_content(response: Any) -> str:
        try:
            return response.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            return ""
