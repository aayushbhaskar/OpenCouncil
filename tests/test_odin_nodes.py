from __future__ import annotations

import pytest

from open_council.core.llm import LLMResult
from open_council.graphs.odin_nodes import (
    judge_node,
    pragmatic_worker_node,
    skeptical_worker_node,
)
from open_council.state.executive import initialize_odin_state


@pytest.mark.asyncio
async def test_pragmatic_worker_node_returns_typed_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _stub_complete(
        self: object,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResult:
        _ = self, max_tokens
        assert temperature == 0.2
        assert messages[1]["content"] == "design a resilient cache"
        return LLMResult(
            ok=True,
            content="Prioritize simplicity and observability.",
            provider="groq",
            model="groq/llama-3.1-70b-versatile",
            attempts=[],
        )

    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)

    state = initialize_odin_state("design a resilient cache")
    update = await pragmatic_worker_node(state)

    assert set(update.keys()) == {"parallel_drafts"}
    assert len(update["parallel_drafts"]) == 1
    assert update["parallel_drafts"][0]["worker_id"] == "muninn"
    assert update["parallel_drafts"][0]["model"] == "groq/llama-3.1-70b-versatile"


@pytest.mark.asyncio
async def test_skeptical_worker_node_returns_safe_fallback_on_result_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _stub_complete(
        self: object,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResult:
        _ = self, messages, temperature, max_tokens
        return LLMResult(
            ok=False,
            content="",
            provider=None,
            model=None,
            attempts=[],
            error="all providers failed",
        )

    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)

    state = initialize_odin_state("design a resilient cache")
    update = await skeptical_worker_node(state)

    assert set(update.keys()) == {"parallel_drafts"}
    assert update["parallel_drafts"][0]["worker_id"] == "huginn"
    assert update["parallel_drafts"][0]["model"] == "unavailable"
    assert "Fallback reason: all providers failed." in update["parallel_drafts"][0]["draft"]


@pytest.mark.asyncio
async def test_judge_node_produces_final_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _stub_complete(
        self: object,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResult:
        _ = self, max_tokens
        assert temperature == 0.1
        assert "Council drafts:" in messages[1]["content"]
        return LLMResult(
            ok=True,
            content="Final Odin verdict.",
            provider="gemini",
            model="gemini/gemini-2.5-flash",
            attempts=[],
        )

    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)

    state = initialize_odin_state("design a resilient cache")
    state["parallel_drafts"] = [
        {"worker_id": "muninn", "model": "x", "draft": "do X first"},
        {"worker_id": "huginn", "model": "y", "draft": "watch Y risk"},
    ]
    update = await judge_node(state)

    assert update == {"final_synthesis": "Final Odin verdict."}


@pytest.mark.asyncio
async def test_judge_node_returns_safe_fallback_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _stub_complete(
        self: object,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResult:
        _ = self, messages, temperature, max_tokens
        raise RuntimeError("network down")

    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)

    state = initialize_odin_state("design a resilient cache")
    update = await judge_node(state)

    assert set(update.keys()) == {"final_synthesis"}
    assert "Unexpected error: network down." in update["final_synthesis"]
