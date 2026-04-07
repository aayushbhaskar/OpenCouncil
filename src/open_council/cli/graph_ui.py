"""Graph execution helpers with Rich streaming UI."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from open_council.state.executive import OdinState


async def invoke_odin_graph_with_ui(*, graph: Any, state: OdinState, console: Console) -> OdinState:
    """
    Execute one Odin graph turn while rendering transient Rich spinners.

    Args:
        graph: Compiled graph runnable exposing `astream`/`ainvoke`.
        state: Input graph state for this turn.
        console: Rich console for transient progress rendering.

    Returns:
        Merged resulting state after processing streamed node updates.
    """
    if not hasattr(graph, "astream"):
        return await graph.ainvoke(state)

    merged_state: OdinState = dict(state)
    node_labels = {
        "muninn_reason_gate": "Muninn reasoning",
        "muninn_query_gen": "Muninn generating search queries",
        "muninn_search": "Muninn searching web",
        "muninn_extract": "Muninn extracting pages",
        "muninn_reason_refine": "Muninn refining with evidence",
        "muninn_draft": "Muninn drafting thesis",
        "huginn_reason_gate": "Huginn reasoning",
        "huginn_query_gen": "Huginn generating search queries",
        "huginn_search": "Huginn searching web",
        "huginn_extract": "Huginn extracting pages",
        "huginn_reason_refine": "Huginn refining with evidence",
        "huginn_draft": "Huginn drafting antithesis",
        "odin_judge": "Odin judging",
    }
    worker_chains = {
        "muninn": [
            "muninn_reason_gate",
            "muninn_query_gen",
            "muninn_search",
            "muninn_extract",
            "muninn_reason_refine",
            "muninn_draft",
        ],
        "huginn": [
            "huginn_reason_gate",
            "huginn_query_gen",
            "huginn_search",
            "huginn_extract",
            "huginn_reason_refine",
            "huginn_draft",
        ],
    }
    completed_nodes: set[str] = set()
    started_nodes: set[str] = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task_ids = {
            node: progress.add_task(description=node_labels[node], total=1, start=False)
            for node in node_labels
        }
        progress.start_task(task_ids["muninn_reason_gate"])
        progress.start_task(task_ids["huginn_reason_gate"])
        started_nodes.add("muninn_reason_gate")
        started_nodes.add("huginn_reason_gate")

        async for chunk in graph.astream(state, stream_mode="updates"):
            node_name = extract_node_name(chunk)
            node_update = extract_node_update(chunk, node_name)
            merged_state = merge_state_update(merged_state, node_update)

            if node_name in task_ids and node_name not in completed_nodes:
                completed_nodes.add(node_name)
                if node_name not in started_nodes:
                    progress.start_task(task_ids[node_name])
                    started_nodes.add(node_name)
                progress.update(task_ids[node_name], completed=1)

                for chain in worker_chains.values():
                    if node_name in chain:
                        index = chain.index(node_name)
                        if index + 1 < len(chain):
                            next_node = chain[index + 1]
                            if next_node not in started_nodes:
                                progress.start_task(task_ids[next_node])
                                started_nodes.add(next_node)

                if {"muninn_draft", "huginn_draft"}.issubset(completed_nodes) and "odin_judge" not in started_nodes:
                    progress.start_task(task_ids["odin_judge"])
                    started_nodes.add("odin_judge")

    return merged_state


def extract_node_name(chunk: Any) -> str | None:
    """
    Extract node name from one streamed update chunk.

    Args:
        chunk: Streamed graph update payload.

    Returns:
        Node key when available, otherwise `None`.
    """
    if isinstance(chunk, dict) and chunk:
        return next(iter(chunk))
    return None


def extract_node_update(chunk: Any, node_name: str | None) -> dict[str, Any]:
    """
    Extract node-specific state update payload from stream chunk.

    Args:
        chunk: Streamed update payload.
        node_name: Node key previously extracted from the chunk.

    Returns:
        Node update dictionary, or empty dict when unavailable.
    """
    if node_name is None:
        return {}
    node_payload = chunk.get(node_name)
    if isinstance(node_payload, dict):
        return node_payload
    return {}


def merge_state_update(state: OdinState, update: dict[str, Any]) -> OdinState:
    """
    Merge streamed state delta into the current state snapshot.

    Args:
        state: Current merged state.
        update: State delta emitted by one graph node.

    Returns:
        Updated merged state honoring list-append semantics for
        `parallel_drafts`.
    """
    merged: OdinState = dict(state)
    for key, value in update.items():
        if key == "parallel_drafts" and isinstance(value, list):
            existing = list(merged.get("parallel_drafts", []))
            existing.extend(value)
            merged["parallel_drafts"] = existing
            continue
        merged[key] = value
    return merged
