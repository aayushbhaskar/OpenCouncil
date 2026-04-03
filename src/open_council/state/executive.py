"""Typed state schema for Executive Mode DAG."""

from __future__ import annotations

from operator import add
from typing import Annotated, NotRequired, TypedDict


class WorkerDraft(TypedDict):
    """One worker output used by the synthesizer/judge node."""

    worker_id: str
    model: str
    draft: str


class ExecutiveState(TypedDict):
    """
    Shared LangGraph state for the Executive mode.

    - `query`: original user prompt
    - `parallel_drafts`: worker responses gathered in parallel
    - `final_synthesis`: final judge output
    """

    query: str
    parallel_drafts: Annotated[list[WorkerDraft], add]
    final_synthesis: NotRequired[str]


def initialize_executive_state(query: str) -> ExecutiveState:
    """Create a valid initial Executive state payload."""
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query cannot be empty.")
    return {
        "query": cleaned_query,
        "parallel_drafts": [],
    }
