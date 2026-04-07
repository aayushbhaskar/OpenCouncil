"""Graph execution helpers with Rich streaming UI."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from open_council.state.executive import OdinState

WORKER_CHAINS = {
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

NODE_LABELS = {
    "muninn_reason_gate": "Muninn: reasoning",
    "muninn_query_gen": "Muninn: generating search queries",
    "muninn_search": "Muninn: searching web",
    "muninn_extract": "Muninn: extracting pages",
    "muninn_reason_refine": "Muninn: refining with evidence",
    "muninn_draft": "Muninn: drafting thesis",
    "huginn_reason_gate": "Huginn: reasoning",
    "huginn_query_gen": "Huginn: generating search queries",
    "huginn_search": "Huginn: searching web",
    "huginn_extract": "Huginn: extracting pages",
    "huginn_reason_refine": "Huginn: refining with evidence",
    "huginn_draft": "Huginn: drafting antithesis",
    "odin_judge": "Odin: judging",
}


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
    phase_state = initialize_phase_state(now=time.monotonic())
    stop_heartbeat = asyncio.Event()
    with Live(
        render_phase_lines(phase_state, now=time.monotonic()),
        transient=True,
        console=console,
        refresh_per_second=10,
    ) as live:
        heartbeat_task = asyncio.create_task(
            _heartbeat_refresh(live=live, phase_state=phase_state, stop_event=stop_heartbeat)
        )
        try:
            async for chunk in graph.astream(state, stream_mode="updates"):
                node_name = extract_node_name(chunk)
                node_update = extract_node_update(chunk, node_name)
                merged_state = merge_state_update(merged_state, node_update)
                if node_name:
                    advance_phase_state(phase_state=phase_state, node_name=node_name, now=time.monotonic())
                    live.update(render_phase_lines(phase_state, now=time.monotonic()), refresh=True)
        finally:
            stop_heartbeat.set()
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task

    return merged_state


def initialize_phase_state(*, now: float) -> dict[str, Any]:
    """Initialize transient phase-state for current-step UI rendering."""
    return {
        "active_worker_nodes": {
            "muninn": WORKER_CHAINS["muninn"][0],
            "huginn": WORKER_CHAINS["huginn"][0],
        },
        "worker_started_at": {
            "muninn": now,
            "huginn": now,
        },
        "odin_active": False,
        "odin_started_at": None,
        "completed_nodes": set(),
    }


def advance_phase_state(*, phase_state: dict[str, Any], node_name: str, now: float) -> None:
    """Advance current-step tracker when one graph node completes."""
    completed_nodes: set[str] = phase_state["completed_nodes"]
    if node_name in completed_nodes:
        return
    completed_nodes.add(node_name)

    for worker, chain in WORKER_CHAINS.items():
        if node_name not in chain:
            continue
        index = chain.index(node_name)
        if index + 1 < len(chain):
            phase_state["active_worker_nodes"][worker] = chain[index + 1]
            phase_state["worker_started_at"][worker] = now
        else:
            phase_state["active_worker_nodes"][worker] = None
            phase_state["worker_started_at"][worker] = None

    if (
        {"muninn_draft", "huginn_draft"}.issubset(completed_nodes)
        and not phase_state["odin_active"]
        and node_name != "odin_judge"
    ):
        phase_state["odin_active"] = True
        phase_state["odin_started_at"] = now
    if node_name == "odin_judge":
        phase_state["odin_active"] = False
        phase_state["odin_started_at"] = None


def render_phase_lines(phase_state: dict[str, Any], *, now: float) -> Group:
    """Render only current active phase lines with live elapsed suffixes."""
    lines: list[Any] = []
    for worker in ("muninn", "huginn"):
        node_name = phase_state["active_worker_nodes"].get(worker)
        if not node_name:
            continue
        started_at = phase_state["worker_started_at"].get(worker) or now
        elapsed = int(max(0, now - started_at))
        label = NODE_LABELS.get(node_name, node_name)
        lines.append(Spinner("dots", f"{label} ({elapsed}s)"))

    if phase_state.get("odin_active"):
        started_at = phase_state.get("odin_started_at") or now
        elapsed = int(max(0, now - started_at))
        lines.append(Spinner("dots", f"{NODE_LABELS['odin_judge']} ({elapsed}s)"))

    if not lines:
        lines.append(Text(""))
    return Group(*lines)


async def _heartbeat_refresh(*, live: Live, phase_state: dict[str, Any], stop_event: asyncio.Event) -> None:
    """Keep UI visibly alive between streamed node updates."""
    while not stop_event.is_set():
        live.update(render_phase_lines(phase_state, now=time.monotonic()), refresh=True)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=0.2)
        except asyncio.TimeoutError:
            continue


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
