"""Typed state schema for Artemis (Academic) cyclic research graph."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from open_council.state.executive import ChatMessage, WorkerExtraction, WorkerSearchHit


class ArtemisState(TypedDict):
    """Shared state for Artemis Planner → Specialists → Critic loops."""

    query: str
    plan: NotRequired[str]
    search_queries: NotRequired[list[str]]
    search_hits: NotRequired[list[WorkerSearchHit]]
    extracted_pages: NotRequired[list[WorkerExtraction]]
    specialist_draft: NotRequired[str]
    critic_verdict: NotRequired[str]
    critic_feedback: NotRequired[str]
    critic_accepted: NotRequired[bool]
    revision_count: NotRequired[int]
    final_synthesis: NotRequired[str]
    chat_history: NotRequired[list[ChatMessage]]
    show_drafts: NotRequired[bool]


def initialize_artemis_state(query: str) -> ArtemisState:
    return {
        "query": query,
        "plan": "",
        "search_queries": [],
        "search_hits": [],
        "extracted_pages": [],
        "specialist_draft": "",
        "critic_verdict": "",
        "critic_feedback": "",
        "critic_accepted": False,
        "revision_count": 0,
        "final_synthesis": "",
        "chat_history": [],
        "show_drafts": False,
    }
