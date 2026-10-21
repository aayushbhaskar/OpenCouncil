"""Offline fake mode: canned LLM, search and page responses for zero-cost testing."""

from __future__ import annotations

import os
import re

_FIRST_PASS_MARKER = "[fake first pass]"


def fake_mode_enabled() -> bool:
    """Return True when `OPEN_COUNCIL_FAKE_LLM=1` (or true/yes/on) is set."""
    return os.getenv("OPEN_COUNCIL_FAKE_LLM", "").strip().lower() in {"1", "true", "yes", "on"}


def fake_completion(messages: list[dict[str, str]]) -> str:
    """
    Return a canned response shaped for whichever node prompt is being answered.

    Responses follow each node's parser format (SEARCH_NEEDED, PLAN/QUERIES,
    VERDICT/FEEDBACK). The critic rejects the first Artemis draft and accepts the
    revision, so one fake turn exercises the revision loop.
    """
    system = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
    user = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
    combined = f"{system}\n{user}"
    query = _extract_query(user)
    history_count = _count_history(user)

    if "SEARCH_NEEDED" in combined:
        return "SEARCH_NEEDED: yes\nREASON: Fake mode always searches to exercise retrieval."
    if "high-signal web queries" in combined:
        return f"fake query about {query}\nfake counterpoint for {query}"
    if "Artemis Planner" in system:
        return (
            "PLAN:\n- Fake objective one for the query\n- Fake objective two for the query\n"
            f"QUERIES:\nfake research {query}\nfake evidence {query}"
        )
    if "Artemis Research Specialist" in system:
        revising = "Critic feedback to address:\n(none)" not in user
        marker = "[fake revision]" if revising else _FIRST_PASS_MARKER
        return (
            f"{marker} Fake specialist draft for: {query}. "
            "See ([source](https://example.com/fake-1))."
        )
    if "Artemis Critic" in system:
        if _FIRST_PASS_MARKER in user:
            return "VERDICT: REJECT\nFEEDBACK: Fake critic rejects the first pass to exercise revision."
        return "VERDICT: ACCEPT\nFEEDBACK: Fake critic accepts the revised draft."
    return (
        "## Fake response\n"
        f"Fake answer for: {query}\n\n"
        f"History messages seen: {history_count}"
    )


def fake_search_results(query: str, max_results: int) -> list[tuple[str, str, str]]:
    """Return deterministic (title, url, snippet) tuples for a query."""
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40] or "query"
    return [
        (
            f"Fake result {index} for {query}",
            f"https://example.com/{slug}/{index}",
            f"Fake snippet {index} about {query}.",
        )
        for index in range(1, max(1, min(max_results, 3)) + 1)
    ]


def fake_page_markdown(url: str) -> str:
    """Return canned page content for a URL."""
    return f"# Fake page\nFake extracted content for {url}."


def _extract_query(user: str) -> str:
    match = re.search(r"(?:Current user query|User query):\s*\n(.+)", user)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in user.splitlines() if line.strip()]
    return lines[-1] if lines else "unknown query"


def _count_history(user: str) -> int:
    match = re.search(
        r"Conversation history:\s*\n(.*?)\n\n(?:Current user query|User query):",
        user,
        flags=re.S,
    )
    if not match:
        return 0
    return len(re.findall(r"^- (?:user|assistant):", match.group(1), flags=re.M))
