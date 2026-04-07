from __future__ import annotations

import pytest

from open_council.core.llm import LLMResult
from open_council.graphs.odin_nodes import (
    huginn_draft_node,
    huginn_extract_node,
    huginn_query_gen_node,
    huginn_reason_gate_node,
    huginn_reason_refine_node,
    huginn_search_node,
    judge_node,
    muninn_draft_node,
    muninn_extract_node,
    muninn_query_gen_node,
    muninn_reason_gate_node,
    muninn_reason_refine_node,
    muninn_search_node,
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
        provider_models: list[tuple[str, str]] | None = None,
    ) -> LLMResult:
        _ = self, max_tokens
        assert temperature == 0.2
        assert "Current user query:\ndesign a resilient cache" in messages[1]["content"]
        assert provider_models is not None
        return LLMResult(
            ok=True,
            content="Prioritize simplicity and observability.",
            provider="groq",
            model="groq/llama-3.3-70b-versatile",
            attempts=[],
        )

    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)

    state = initialize_odin_state("design a resilient cache")
    update = await pragmatic_worker_node(state)

    assert set(update.keys()) == {"parallel_drafts"}
    assert len(update["parallel_drafts"]) == 1
    assert update["parallel_drafts"][0]["worker_id"] == "muninn"
    assert update["parallel_drafts"][0]["model"] == "groq/llama-3.3-70b-versatile"


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
        provider_models: list[tuple[str, str]] | None = None,
    ) -> LLMResult:
        _ = self, messages, temperature, max_tokens, provider_models
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
        provider_models: list[tuple[str, str]] | None = None,
    ) -> LLMResult:
        _ = self, max_tokens
        assert temperature == 0.1
        assert "Council drafts:" in messages[1]["content"]
        assert provider_models is not None
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
        provider_models: list[tuple[str, str]] | None = None,
    ) -> LLMResult:
        _ = self, messages, temperature, max_tokens, provider_models
        raise RuntimeError("network down")

    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)

    state = initialize_odin_state("design a resilient cache")
    update = await judge_node(state)

    assert set(update.keys()) == {"final_synthesis"}
    assert "Unexpected error: network down." in update["final_synthesis"]


@pytest.mark.asyncio
async def test_worker_and_judge_use_node_specific_models(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_provider_models: list[list[tuple[str, str]]] = []

    async def _stub_complete(
        self: object,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        provider_models: list[tuple[str, str]] | None = None,
    ) -> LLMResult:
        _ = self, messages, temperature, max_tokens
        captured_provider_models.append(list(provider_models or []))
        return LLMResult(
            ok=True,
            content="ok",
            provider="groq",
            model="groq/custom",
            attempts=[],
        )

    monkeypatch.setenv("MUNINN_MODEL", "gemini/custom-muninn")
    monkeypatch.setenv("HUGINN_MODEL", "ollama/custom-huginn")
    monkeypatch.setenv("ODIN_MODEL", "groq/custom-odin")
    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)

    state = initialize_odin_state("design a resilient cache")
    await pragmatic_worker_node(state)
    await skeptical_worker_node(state)
    state["parallel_drafts"] = [
        {"worker_id": "muninn", "model": "x", "draft": "d1"},
        {"worker_id": "huginn", "model": "y", "draft": "d2"},
    ]
    await judge_node(state)

    assert captured_provider_models[0][0] == ("gemini", "gemini/custom-muninn")
    assert captured_provider_models[1][0] == ("ollama", "ollama/custom-huginn")
    assert captured_provider_models[2][0] == ("groq", "groq/custom-odin")


@pytest.mark.asyncio
async def test_reason_gate_sets_needs_search_and_note(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub_complete(
        self: object,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        provider_models: list[tuple[str, str]] | None = None,
    ) -> LLMResult:
        _ = self, messages, temperature, max_tokens, provider_models
        return LLMResult(
            ok=True,
            content="SEARCH_NEEDED: yes\nREASON: Need current policy updates.",
            provider="groq",
            model="groq/x",
            attempts=[],
        )

    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)
    state = initialize_odin_state("What changed in AI regulation this week?")
    update = await muninn_reason_gate_node(state)

    assert "muninn_retrieval" in update
    assert update["muninn_retrieval"]["needs_search"] is True
    assert "policy updates" in update["muninn_retrieval"]["reasoning_note"]


@pytest.mark.asyncio
async def test_query_search_extract_pipeline_respects_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub_complete(
        self: object,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        provider_models: list[tuple[str, str]] | None = None,
    ) -> LLMResult:
        _ = self, messages, temperature, max_tokens, provider_models
        return LLMResult(
            ok=True,
            content="q1\nq2\nq3",
            provider="groq",
            model="groq/x",
            attempts=[],
        )

    class _FakeProvider:
        async def search(self, query: str, *, max_results: int = 5):
            _ = query
            return [
                type(
                    "R",
                    (),
                    {
                        "title": f"t{i}",
                        "url": f"https://example.com/{i}",
                        "snippet": "s",
                        "source": "duckduckgo",
                    },
                )()
                for i in range(max_results + 2)
            ]

    class _FakeReader:
        def __init__(self, timeout_seconds: float = 10.0) -> None:
            _ = timeout_seconds

        async def fetch_markdown(self, url: str) -> str:
            return f"content for {url}"

    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)
    monkeypatch.setattr("open_council.graphs.odin_nodes.DuckDuckGoSearchProvider", _FakeProvider)
    monkeypatch.setattr("open_council.graphs.odin_nodes.JinaReader", _FakeReader)

    state = initialize_odin_state("Assess latest CVE impact")
    state["muninn_retrieval"] = {
        "needs_search": True,
        "reasoning_note": "Need external evidence",
        "search_queries": [],
        "search_hits": [],
        "extracted_pages": [],
        "refined_reasoning": "",
    }
    query_update = await muninn_query_gen_node(state)
    state.update(query_update)
    assert len(state["muninn_retrieval"]["search_queries"]) == 2

    search_update = await muninn_search_node(state)
    state.update(search_update)
    assert len(state["muninn_retrieval"]["search_hits"]) == 5

    extract_update = await muninn_extract_node(state)
    state.update(extract_update)
    assert len(state["muninn_retrieval"]["extracted_pages"]) == 5


@pytest.mark.asyncio
async def test_huginn_no_search_path_skips_retrieval_and_still_drafts(monkeypatch: pytest.MonkeyPatch) -> None:
    sequence = iter(
        [
            LLMResult(ok=True, content="SEARCH_NEEDED: no\nREASON: Stable topic.", provider="groq", model="g", attempts=[]),
            LLMResult(ok=True, content="refined reasoning", provider="groq", model="g", attempts=[]),
            LLMResult(ok=True, content="final draft", provider="groq", model="g", attempts=[]),
        ]
    )

    async def _stub_complete(
        self: object,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        provider_models: list[tuple[str, str]] | None = None,
    ) -> LLMResult:
        _ = self, messages, temperature, max_tokens, provider_models
        return next(sequence)

    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)

    state = initialize_odin_state("Foundational concept question")
    state.update(await huginn_reason_gate_node(state))
    state.update(await huginn_query_gen_node(state))
    state.update(await huginn_search_node(state))
    state.update(await huginn_extract_node(state))
    state.update(await huginn_reason_refine_node(state))
    draft_update = await huginn_draft_node(state)

    assert state["huginn_retrieval"]["needs_search"] is False
    assert state["huginn_retrieval"]["search_queries"] == []
    assert state["huginn_retrieval"]["search_hits"] == []
    assert draft_update["parallel_drafts"][0]["worker_id"] == "huginn"
