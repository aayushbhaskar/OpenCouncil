"""Compiled DAG for Odin (Executive) mode."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from open_council.graphs.odin_nodes import (
    judge_node,
    pragmatic_worker_node,
    skeptical_worker_node,
)
from open_council.state.executive import OdinState


def build_odin_graph():
    """Build and compile Odin DAG: START -> workers -> judge -> END."""
    graph = StateGraph(OdinState)
    graph.add_node("muninn_worker", pragmatic_worker_node)
    graph.add_node("huginn_worker", skeptical_worker_node)
    graph.add_node("odin_judge", judge_node)

    graph.add_edge(START, "muninn_worker")
    graph.add_edge(START, "huginn_worker")
    graph.add_edge("muninn_worker", "odin_judge")
    graph.add_edge("huginn_worker", "odin_judge")
    graph.add_edge("odin_judge", END)

    return graph.compile()


def compile_executive_graph():
    """Compatibility name for PRD task references."""
    return build_odin_graph()
