from __future__ import annotations

import pytest

from open_council.graphs import executive_graph
from open_council.state.executive import initialize_odin_state


@pytest.mark.asyncio
async def test_build_odin_graph_executes_workers_then_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _passthrough(state: dict[str, object]) -> dict[str, object]:
        return {}

    async def _muninn_draft(state: dict[str, object]) -> dict[str, list[dict[str, str]]]:
        _ = state
        return {
            "parallel_drafts": [
                {"worker_id": "muninn", "model": "mock-model", "draft": "pragmatic draft"}
            ]
        }

    async def _huginn_draft(state: dict[str, object]) -> dict[str, list[dict[str, str]]]:
        _ = state
        return {
            "parallel_drafts": [
                {"worker_id": "huginn", "model": "mock-model", "draft": "skeptical draft"}
            ]
        }

    async def _judge_node(state: dict[str, object]) -> dict[str, str]:
        drafts = state.get("parallel_drafts", [])
        assert isinstance(drafts, list)
        assert len(drafts) == 2
        return {"final_synthesis": "final verdict"}

    for name in [
        "muninn_reason_gate_node",
        "muninn_query_gen_node",
        "muninn_search_node",
        "muninn_extract_node",
        "muninn_reason_refine_node",
        "huginn_reason_gate_node",
        "huginn_query_gen_node",
        "huginn_search_node",
        "huginn_extract_node",
        "huginn_reason_refine_node",
    ]:
        monkeypatch.setattr(executive_graph, name, _passthrough)
    monkeypatch.setattr(executive_graph, "muninn_draft_node", _muninn_draft)
    monkeypatch.setattr(executive_graph, "huginn_draft_node", _huginn_draft)
    monkeypatch.setattr(executive_graph, "judge_node", _judge_node)

    graph = executive_graph.build_odin_graph()
    result = await graph.ainvoke(initialize_odin_state("design a resilient cache"))

    assert result["query"] == "design a resilient cache"
    assert len(result["parallel_drafts"]) == 2
    assert result["final_synthesis"] == "final verdict"


@pytest.mark.asyncio
async def test_compile_executive_graph_alias_builds_runnable_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _passthrough(state: dict[str, object]) -> dict[str, object]:
        return {}

    async def _draft(state: dict[str, object]) -> dict[str, list[dict[str, str]]]:
        _ = state
        return {"parallel_drafts": [{"worker_id": "x", "model": "m", "draft": "d"}]}

    async def _judge(state: dict[str, object]) -> dict[str, str]:
        _ = state
        return {"final_synthesis": "ok"}

    for name in [
        "muninn_reason_gate_node",
        "muninn_query_gen_node",
        "muninn_search_node",
        "muninn_extract_node",
        "muninn_reason_refine_node",
        "huginn_reason_gate_node",
        "huginn_query_gen_node",
        "huginn_search_node",
        "huginn_extract_node",
        "huginn_reason_refine_node",
    ]:
        monkeypatch.setattr(executive_graph, name, _passthrough)
    monkeypatch.setattr(executive_graph, "muninn_draft_node", _draft)
    monkeypatch.setattr(executive_graph, "huginn_draft_node", _draft)
    monkeypatch.setattr(executive_graph, "judge_node", _judge)

    graph = executive_graph.compile_executive_graph()
    result = await graph.ainvoke(initialize_odin_state("q"))

    assert result["final_synthesis"] == "ok"
