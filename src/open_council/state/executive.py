"""Typed state schema for Odin (Executive) mode DAG."""

from __future__ import annotations

from operator import add
from typing import Annotated, NotRequired, TypedDict


class WorkerDraft(TypedDict):
    """
    One worker output consumed by the Odin judge node.

    Attributes:
        worker_id: Stable worker identity label.
        model: Model identifier used by the worker call.
        draft: Worker-generated analysis text.
    """

    worker_id: str
    model: str
    draft: str


class ChatMessage(TypedDict):
    """
    One persisted chat turn in the CLI session.

    Attributes:
        role: Speaker role (`user` or `assistant`).
        content: Message content shown and reused as context.
    """

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
    """
    Create a valid initial Odin state payload.

    Args:
        query: First user query for the session.

    Returns:
        Initialized `OdinState` dictionary with empty drafts/history.

    Raises:
        ValueError: If query is empty or whitespace only.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query cannot be empty.")
    return {
        "query": cleaned_query,
        "parallel_drafts": [],
        "chat_history": [],
    }
