from __future__ import annotations

import pytest

from open_council.graphs import executive_graph
from open_council.state.executive import initialize_odin_state


@pytest.mark.asyncio
async def test_build_odin_graph_executes_workers_then_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _muninn_node(state: dict[str, object]) -> dict[str, list[dict[str, str]]]:
        _ = state
        return {
            "parallel_drafts": [
                {"worker_id": "muninn", "model": "mock-model", "draft": "pragmatic draft"}
            ]
        }

    async def _huginn_node(state: dict[str, object]) -> dict[str, list[dict[str, str]]]:
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

    monkeypatch.setattr(executive_graph, "pragmatic_worker_node", _muninn_node)
    monkeypatch.setattr(executive_graph, "skeptical_worker_node", _huginn_node)
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
    async def _worker(state: dict[str, object]) -> dict[str, list[dict[str, str]]]:
        _ = state
        return {"parallel_drafts": [{"worker_id": "x", "model": "m", "draft": "d"}]}

    async def _judge(state: dict[str, object]) -> dict[str, str]:
        _ = state
        return {"final_synthesis": "ok"}

    monkeypatch.setattr(executive_graph, "pragmatic_worker_node", _worker)
    monkeypatch.setattr(executive_graph, "skeptical_worker_node", _worker)
    monkeypatch.setattr(executive_graph, "judge_node", _judge)

    graph = executive_graph.compile_executive_graph()
    result = await graph.ainvoke(initialize_odin_state("q"))

    assert result["final_synthesis"] == "ok"
