"""Typed state schema for Odin (Executive) mode DAG."""

from __future__ import annotations

from operator import add
from typing import Annotated, NotRequired, TypedDict


class WorkerDraft(TypedDict):
    """One worker output used by the synthesizer/judge node."""

    worker_id: str
    model: str
    draft: str


class ChatMessage(TypedDict):
    """One persisted chat turn in the CLI session."""

    role: str
    content: str


class OdinState(TypedDict):
    """
    Shared LangGraph state for Odin mode.

    - `query`: original user prompt
    - `parallel_drafts`: worker responses gathered in parallel
    - `final_synthesis`: final judge output
    - `chat_history`: persisted turn-by-turn conversation context
    """

    query: str
    parallel_drafts: Annotated[list[WorkerDraft], add]
    final_synthesis: NotRequired[str]
    chat_history: list[ChatMessage]


def initialize_odin_state(query: str) -> OdinState:
    """Create a valid initial Odin state payload."""
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query cannot be empty.")
    return {
        "query": cleaned_query,
        "parallel_drafts": [],
        "chat_history": [],
    }
