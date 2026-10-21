"""Compiled cyclic graph for Artemis (Academic) mode."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from open_council.graphs.artemis_nodes import (
    artemis_critic_node,
    artemis_extract_node,
    artemis_planner_node,
    artemis_search_node,
    artemis_specialist_draft_node,
    artemis_synthesize_node,
    route_after_critic,
)
from open_council.state.artemis import ArtemisState


def build_artemis_graph(*, checkpointer: Any | None = None):
    """
    Build Artemis cyclic research graph.

    Flow:
      START -> planner -> search -> extract -> specialist_draft -> critic
        -> (revise) planner ... OR (synthesize) -> END
    """
    graph = StateGraph(ArtemisState)
    graph.add_node("artemis_planner", artemis_planner_node)
    graph.add_node("artemis_search", artemis_search_node)
    graph.add_node("artemis_extract", artemis_extract_node)
    graph.add_node("artemis_specialist_draft", artemis_specialist_draft_node)
    graph.add_node("artemis_critic", artemis_critic_node)
    graph.add_node("artemis_synthesize", artemis_synthesize_node)

    graph.add_edge(START, "artemis_planner")
    graph.add_edge("artemis_planner", "artemis_search")
    graph.add_edge("artemis_search", "artemis_extract")
    graph.add_edge("artemis_extract", "artemis_specialist_draft")
    graph.add_edge("artemis_specialist_draft", "artemis_critic")
    graph.add_conditional_edges(
        "artemis_critic",
        route_after_critic,
        {
            "revise": "artemis_planner",
            "synthesize": "artemis_synthesize",
        },
    )
    graph.add_edge("artemis_synthesize", END)

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
