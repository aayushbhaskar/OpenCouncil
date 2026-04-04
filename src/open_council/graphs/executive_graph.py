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
    """
    Build and compile the Odin DAG topology.

    Graph:
        START -> (muninn_worker + huginn_worker in parallel) -> odin_judge -> END

    Returns:
        A compiled LangGraph runnable supporting `ainvoke` and `astream`.
    """
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
    """
    Compatibility wrapper for legacy/PRD naming.

    Returns:
        The same compiled graph returned by `build_odin_graph()`.
    """
    return build_odin_graph()
