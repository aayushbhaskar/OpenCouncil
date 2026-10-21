from __future__ import annotations

import asyncio

from open_council.graphs import artemis_graph
from open_council.runtime.checkpoints import (
    ensure_thread_id,
    list_threads,
    open_async_sqlite_saver,
    touch_thread,
)
from open_council.state.artemis import initialize_artemis_state


def test_touch_and_list_threads(tmp_path) -> None:
    db = tmp_path / "checkpoints.sqlite"
    touch_thread(thread_id="t1", mode="odin", path=db)
    touch_thread(thread_id="t2", mode="artemis", path=db)
    rows = list_threads(path=db, limit=10)
    assert [row.thread_id for row in rows] == ["t2", "t1"]
    assert rows[0].mode == "artemis"


def test_ensure_thread_id_generates_when_blank() -> None:
    generated = ensure_thread_id("")
    assert len(generated) > 8
    assert ensure_thread_id("abc") == "abc"


def test_async_saver_checkpoints_streamed_graph_across_event_loops(tmp_path, monkeypatch) -> None:
    async def _planner(state):
        return {"plan": "p", "search_queries": ["q"]}

    async def _noop(state):
        return {}

    async def _critic(state):
        return {"critic_accepted": True, "critic_verdict": "ACCEPT", "revision_count": 0}

    async def _synth(state):
        return {"final_synthesis": f"answer: {state['query']}"}

    monkeypatch.setattr(artemis_graph, "artemis_planner_node", _planner)
    monkeypatch.setattr(artemis_graph, "artemis_search_node", _noop)
    monkeypatch.setattr(artemis_graph, "artemis_extract_node", _noop)
    monkeypatch.setattr(artemis_graph, "artemis_specialist_draft_node", _noop)
    monkeypatch.setattr(artemis_graph, "artemis_critic_node", _critic)
    monkeypatch.setattr(artemis_graph, "artemis_synthesize_node", _synth)

    db = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "t-async"}}

    async def _turn(query: str) -> str:
        async with open_async_sqlite_saver(db) as saver:
            assert saver is not None
            graph = artemis_graph.build_artemis_graph(checkpointer=saver)
            state = initialize_artemis_state(query=query)
            async for _ in graph.astream(state, config=config, stream_mode="updates"):
                pass
            snapshot = await graph.aget_state(config)
            return snapshot.values["final_synthesis"]

    assert asyncio.run(_turn("first")) == "answer: first"
    assert asyncio.run(_turn("second")) == "answer: second"
