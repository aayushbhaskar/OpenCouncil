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


class WorkerSearchHit(TypedDict):
    """
    One normalized web search hit for worker retrieval context.

    Attributes:
        title: Result title text.
        url: Result URL.
        snippet: Short result summary.
        source: Provider label (e.g., duckduckgo).
    """

    title: str
    url: str
    snippet: str
    source: str


class WorkerExtraction(TypedDict):
    """
    One extracted page payload retained for worker reasoning.

    Attributes:
        url: Source URL of extracted content.
        content: Extracted markdown/text content (bounded length).
    """

    url: str
    content: str


class WorkerRetrievalState(TypedDict):
    """
    Per-worker retrieval context used across ReAct-style subnodes.

    Attributes:
        needs_search: Whether worker determined web search is needed.
        reasoning_note: Short rationale for the search decision.
        search_queries: Generated search queries (bounded list).
        search_hits: Aggregated normalized DDG search hits.
        extracted_pages: Extracted page contents from Jina reader.
        refined_reasoning: Post-retrieval reasoning summary.
    """

    needs_search: bool
    reasoning_note: str
    search_queries: list[str]
    search_hits: list[WorkerSearchHit]
    extracted_pages: list[WorkerExtraction]
    refined_reasoning: str


class OdinState(TypedDict):
    """
    Shared LangGraph state for Odin mode.

    - `query`: original user prompt
    - `parallel_drafts`: worker responses gathered in parallel
    - `final_synthesis`: final judge output
    - `chat_history`: persisted turn-by-turn conversation context
    - `show_drafts`: runtime UI toggle for printing worker drafts
    - `muninn_retrieval`: Muninn retrieval pipeline state
    - `huginn_retrieval`: Huginn retrieval pipeline state
    """

    query: str
    parallel_drafts: Annotated[list[WorkerDraft], add]
    final_synthesis: NotRequired[str]
    chat_history: list[ChatMessage]
    show_drafts: NotRequired[bool]
    muninn_retrieval: NotRequired[WorkerRetrievalState]
    huginn_retrieval: NotRequired[WorkerRetrievalState]


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
        "show_drafts": False,
        "muninn_retrieval": _initialize_worker_retrieval_state(),
        "huginn_retrieval": _initialize_worker_retrieval_state(),
    }


def _initialize_worker_retrieval_state() -> WorkerRetrievalState:
    """
    Build an empty per-worker retrieval state payload.

    Returns:
        Zeroed retrieval context for one worker branch.
    """
    return {
        "needs_search": False,
        "reasoning_note": "",
        "search_queries": [],
        "search_hits": [],
        "extracted_pages": [],
        "refined_reasoning": "",
    }
