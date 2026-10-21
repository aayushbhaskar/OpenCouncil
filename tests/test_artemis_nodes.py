from __future__ import annotations

import pytest

from open_council.core.llm import LLMResult
from open_council.graphs.artemis_nodes import (
    artemis_critic_node,
    artemis_planner_node,
    artemis_specialist_draft_node,
    artemis_synthesize_node,
    route_after_critic,
)
from open_council.state.artemis import initialize_artemis_state


@pytest.mark.asyncio
async def test_artemis_planner_parses_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub_complete(self, messages, **kwargs):  # noqa: ANN001
        _ = self, messages, kwargs
        return LLMResult(
            ok=True,
            content="PLAN:\n- Map sources\nQUERIES:\nq1\nq2\nq3\nq4",
            provider="groq",
            model="g",
            attempts=[],
        )

    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)
    state = initialize_artemis_state("Research topic X")
    update = await artemis_planner_node(state)
    assert "Map sources" in update["plan"]
    assert update["search_queries"] == ["q1", "q2", "q3"]


@pytest.mark.asyncio
async def test_critic_reject_increments_revision_and_routes_revise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _stub_complete(self, messages, **kwargs):  # noqa: ANN001
        _ = self, messages, kwargs
        return LLMResult(
            ok=True,
            content="VERDICT: REJECT\nFEEDBACK: Missing citations.",
            provider="groq",
            model="g",
            attempts=[],
        )

    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)
    state = initialize_artemis_state("q")
    state["specialist_draft"] = "draft"
    state["revision_count"] = 0
    update = await artemis_critic_node(state)
    merged = {**state, **update}
    assert update["critic_accepted"] is False
    assert update["revision_count"] == 1
    assert route_after_critic(merged) == "revise"


@pytest.mark.asyncio
async def test_critic_max_revisions_routes_to_synthesize(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub_complete(self, messages, **kwargs):  # noqa: ANN001
        _ = self, messages, kwargs
        return LLMResult(
            ok=True,
            content="VERDICT: REJECT\nFEEDBACK: Still weak.",
            provider="groq",
            model="g",
            attempts=[],
        )

    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)
    state = initialize_artemis_state("q")
    state["specialist_draft"] = "draft"
    state["revision_count"] = 2
    update = await artemis_critic_node(state)
    merged = {**state, **update}
    assert update["revision_count"] == 3
    assert route_after_critic(merged) == "synthesize"


@pytest.mark.asyncio
async def test_synthesize_uses_specialist_on_llm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub_complete(self, messages, **kwargs):  # noqa: ANN001
        _ = self, messages, kwargs
        return LLMResult(ok=False, content="", provider=None, model=None, attempts=[], error="down")

    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)
    state = initialize_artemis_state("q")
    state["specialist_draft"] = "fallback draft body"
    update = await artemis_synthesize_node(state)
    assert update["final_synthesis"] == "fallback draft body"


@pytest.mark.asyncio
async def test_specialist_draft_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub_complete(self, messages, **kwargs):  # noqa: ANN001
        _ = self, messages, kwargs
        return LLMResult(ok=True, content="researched draft", provider="groq", model="g", attempts=[])

    monkeypatch.setattr("open_council.core.llm.LiteLLMClient.complete", _stub_complete)
    state = initialize_artemis_state("q")
    state["plan"] = "do research"
    update = await artemis_specialist_draft_node(state)
    assert update["specialist_draft"] == "researched draft"
