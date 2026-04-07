"""Compiled DAG for Odin (Executive) mode."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

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
)
from open_council.state.executive import OdinState


def build_odin_graph():
    """
    Build and compile the Odin DAG topology.

    Graph:
        START -> parallel ReAct worker chains -> odin_judge -> END

    Returns:
        A compiled LangGraph runnable supporting `ainvoke` and `astream`.
    """
    graph = StateGraph(OdinState)
    graph.add_node("muninn_reason_gate", muninn_reason_gate_node)
    graph.add_node("muninn_query_gen", muninn_query_gen_node)
    graph.add_node("muninn_search", muninn_search_node)
    graph.add_node("muninn_extract", muninn_extract_node)
    graph.add_node("muninn_reason_refine", muninn_reason_refine_node)
    graph.add_node("muninn_draft", muninn_draft_node)
    graph.add_node("huginn_reason_gate", huginn_reason_gate_node)
    graph.add_node("huginn_query_gen", huginn_query_gen_node)
    graph.add_node("huginn_search", huginn_search_node)
    graph.add_node("huginn_extract", huginn_extract_node)
    graph.add_node("huginn_reason_refine", huginn_reason_refine_node)
    graph.add_node("huginn_draft", huginn_draft_node)
    graph.add_node("odin_judge", judge_node)

    graph.add_edge(START, "muninn_reason_gate")
    graph.add_edge(START, "huginn_reason_gate")

    graph.add_edge("muninn_reason_gate", "muninn_query_gen")
    graph.add_edge("muninn_query_gen", "muninn_search")
    graph.add_edge("muninn_search", "muninn_extract")
    graph.add_edge("muninn_extract", "muninn_reason_refine")
    graph.add_edge("muninn_reason_refine", "muninn_draft")

    graph.add_edge("huginn_reason_gate", "huginn_query_gen")
    graph.add_edge("huginn_query_gen", "huginn_search")
    graph.add_edge("huginn_search", "huginn_extract")
    graph.add_edge("huginn_extract", "huginn_reason_refine")
    graph.add_edge("huginn_reason_refine", "huginn_draft")

    graph.add_edge("muninn_draft", "odin_judge")
    graph.add_edge("huginn_draft", "odin_judge")
    graph.add_edge("odin_judge", END)

    return graph.compile()


def compile_executive_graph():
    """
    Compatibility wrapper for legacy/PRD naming.

    Returns:
        The same compiled graph returned by `build_odin_graph()`.
    """
    return build_odin_graph()
