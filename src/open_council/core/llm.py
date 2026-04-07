"""Async LiteLLM wrapper with ordered provider fallback."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import acompletion

from open_council.core.throttle import network_throttle


@dataclass(slots=True)
class LLMAttempt:
    """
    Metadata for one provider attempt in a fallback sequence.

    Attributes:
        provider: Logical provider name (e.g., groq, gemini, ollama).
        model: Model identifier used for this attempt.
        error: Optional error message when the attempt failed.
    """

    provider: str
    model: str
    error: str | None = None


@dataclass(slots=True)
class LLMResult:
    """
    Normalized LLM response object consumed by graph nodes.

    Attributes:
        ok: Whether at least one provider call succeeded.
        content: Extracted assistant text content (empty on failure).
        provider: Provider that produced the successful answer.
        model: Model that produced the successful answer.
        attempts: Ordered provider-attempt records for observability.
        error: Aggregate failure reason when all providers fail.
        raw_response: Raw LiteLLM payload for optional advanced inspection.
    """

    ok: bool
    content: str
    provider: str | None
    model: str | None
    attempts: list[LLMAttempt]
    error: str | None = None
    raw_response: Any | None = None


class LiteLLMClient:
    """
    Defensive async wrapper around `litellm.acompletion`.

    This client applies:
    - strict provider fallback ordering
    - shared semaphore throttling
    - normalized response shape
    - graceful non-raising failure semantics
    """

    def __init__(self) -> None:
        """
        Initialize timeout and provider model routing defaults from env vars.

        Environment:
            LITELLM_TIMEOUT_SECONDS
            GROQ_MODEL
            GEMINI_MODEL
            OLLAMA_MODEL
        """
        self.timeout_seconds = float(os.getenv("LITELLM_TIMEOUT_SECONDS", "30"))
        self.provider_models = [
            ("groq", os.getenv("GROQ_MODEL", "groq/llama-3.3-70b-versatile")),
            ("gemini", os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")),
            ("ollama", os.getenv("OLLAMA_MODEL", "ollama/llama3.1")),
        ]
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        provider_models: list[tuple[str, str]] | None = None,
    ) -> LLMResult:
        """
        Execute chat completion with Groq -> Gemini -> Ollama fallback.

        Args:
            messages: OpenAI-style message array for LiteLLM completion.
            temperature: Sampling temperature sent to provider.
            max_tokens: Optional max output token cap.
            provider_models: Optional provider/model chain override for this
                call. When omitted, uses the default Groq -> Gemini -> Ollama
                sequence.

        Returns:
            `LLMResult` containing content on success, or a safe failure object
            when all fallback providers fail.
        """
        attempts: list[LLMAttempt] = []

        active_provider_models = provider_models or self.provider_models

        for index, (provider, model) in enumerate(active_provider_models):
            try:
                response = await asyncio.wait_for(
                    network_throttle.run(
                        lambda: acompletion(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout=self.timeout_seconds,
                            **self._provider_kwargs(provider),
                        )
                    ),
                    timeout=self.timeout_seconds + 5.0,
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
                has_next_provider = index < len(active_provider_models) - 1
                if has_next_provider:
                    next_provider, _ = active_provider_models[index + 1]
                    print(
                        f"Provider retry: {provider} unavailable, trying {next_provider}..."
                    )

        return LLMResult(
            ok=False,
            content="",
            provider=None,
            model=None,
            attempts=attempts,
            error="All fallback providers failed (Groq -> Gemini -> Ollama).",
        )

    def _provider_kwargs(self, provider: str) -> dict[str, Any]:
        """
        Build provider-specific LiteLLM kwargs from environment config.

        Args:
            provider: Logical provider key for the active attempt.

        Returns:
            Keyword arguments passed into `litellm.acompletion`.
        """
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
        """
        Extract assistant text content from a LiteLLM response payload.

        Args:
            response: Raw response object returned by LiteLLM.

        Returns:
            Extracted content string, or empty string when extraction fails.
        """
        try:
            return response.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            return ""


def configure_litellm_logging(*, debug: bool) -> None:
    """
    Configure LiteLLM diagnostics verbosity for CLI runtime.

    Args:
        debug: When True, allow verbose LiteLLM diagnostics.
    """
    litellm.suppress_debug_info = not debug
    if debug:
        os.environ["LITELLM_LOG"] = "DEBUG"
    else:
        os.environ.pop("LITELLM_LOG", None)
