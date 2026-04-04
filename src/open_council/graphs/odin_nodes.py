"""Odin mode worker and judge node functions."""

from __future__ import annotations

from typing import Any

from open_council.core.llm import LiteLLMClient
from open_council.state.executive import ChatMessage, OdinState, WorkerDraft

_MUNINN_PROMPT = """You are The Pragmatist, an expert consultant focused on execution, utility, and actionable solutions.

Your objective is to provide the most direct, effective, and optimistic path forward for the user's query, regardless of the domain (technology, business, personal strategy, etc.).

Core Directives:
1. Assume the user's goal is achievable. Focus on how to do it best.
2. Outline the most efficient steps, best practices, and immediate wins.
3. Be highly structured. Use clear headings and bullet points.
4. Do NOT dwell on edge cases, extreme risks, or philosophical debates. Leave the skepticism to others; your job is to build the roadmap.
5. Tone: Confident, actionable, and concise. Strip out all robotic filler (e.g., do not start with "Here is a pragmatic approach..."). Start immediately with your analysis.
"""
_HUGINN_PROMPT = """You are The Skeptic, an expert risk analyst and contrarian thinker.

Your objective is to ruthlessly but fairly analyze the user's query to expose hidden risks, unstated assumptions, and potential points of failure. You must look at the problem from the outside in, regardless of the domain.

Core Directives:
1. Question the premise. Is this even the right problem to be solving?
2. Highlight the hidden costs (time, money, technical debt, emotional toll).
3. Identify edge cases and catastrophic failure modes that a pure optimist would miss.
4. Do not simply be negative for the sake of it; your skepticism must be grounded in reality and aimed at protecting the user from making a mistake.
5. Tone: Analytical, cautious, and sharp. Strip out all robotic filler (e.g., do not start with "As a skeptic..."). Start immediately with your critique.
"""
_JUDGE_PROMPT = """You are The Judge (Odin), the final synthesizer in a multi-agent council.

You will be provided with the user's original query, an analysis from Muninn (the Pragmatist who focuses on execution), and an analysis from Huginn (the Skeptic who focuses on risk). Your objective is to reconcile these two opposing viewpoints into a single, authoritative verdict.

Core Directives:
1. Do not simply summarize what Muninn and Huginn said. You must weigh them against each other and make a definitive judgment call.
2. If Muninn's plan is fundamentally flawed based on Huginn's risks, say so. If Huginn is being overly cautious about a minor issue, override them.
3. Provide the user with a final, actionable resolution that balances speed/utility with safety/risk.
4. Keep verbosity to an absolute minimum. Every sentence must carry weight.

Required Output Format:
* The Verdict: A one-paragraph definitive answer to the user's query.
* The Critical Trade-off: The single most important tension between execution and risk that the user must manage.
* The Path Forward: 2-3 highly specific, synthesized steps the user should take immediately.

Tone: Authoritative, objective, and decisively brief. Do not use concluding filler phrases like "Ultimately, it is up to you." Make the call.
"""


async def pragmatic_worker_node(state: OdinState) -> dict[str, list[WorkerDraft]]:
    """
    Generate Muninn (pragmatist) worker draft.

    Args:
        state: Current Odin graph state.

    Returns:
        State delta containing a single appended `parallel_drafts` item.
    """
    return await _run_worker_node(
        state=state,
        worker_id="muninn",
        system_prompt=_MUNINN_PROMPT,
    )


async def skeptical_worker_node(state: OdinState) -> dict[str, list[WorkerDraft]]:
    """
    Generate Huginn (skeptic) worker draft.

    Args:
        state: Current Odin graph state.

    Returns:
        State delta containing a single appended `parallel_drafts` item.
    """
    return await _run_worker_node(
        state=state,
        worker_id="huginn",
        system_prompt=_HUGINN_PROMPT,
    )


async def judge_node(state: OdinState) -> dict[str, str]:
    """
    Synthesize worker drafts into the final Odin verdict.

    Args:
        state: Current Odin graph state, including worker drafts.

    Returns:
        State delta with `final_synthesis`. On failures, returns a safe
        explanatory message instead of raising.
    """
    client = LiteLLMClient()
    messages = _build_judge_messages(state)
    try:
        result = await client.complete(messages, temperature=0.1)
    except Exception as exc:  # noqa: BLE001
        return {
            "final_synthesis": (
                "Odin judge could not complete synthesis. "
                f"Unexpected error: {exc}."
            )
        }

    if result.ok:
        return {"final_synthesis": result.content}

    failure_message = (
        "Odin judge could not complete synthesis. "
        f"Fallback reason: {result.error or 'unknown error'}."
    )
    return {"final_synthesis": failure_message}


async def _run_worker_node(
    *,
    state: OdinState,
    worker_id: str,
    system_prompt: str,
) -> dict[str, list[WorkerDraft]]:
    """
    Shared worker-node execution logic for Muninn and Huginn.

    Args:
        state: Current Odin state.
        worker_id: Stable worker identity (`muninn` or `huginn`).
        system_prompt: Worker persona/system prompt.

    Returns:
        State delta with one worker draft entry. Returns fallback draft content
        when model calls fail or raise.
    """
    client = LiteLLMClient()
    messages = _build_worker_messages(state=state, system_prompt=system_prompt)
    try:
        result = await client.complete(messages, temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        draft: WorkerDraft = {
            "worker_id": worker_id,
            "model": "unavailable",
            "draft": (
                f"{worker_id.title()} worker could not generate a draft. "
                f"Unexpected error: {exc}."
            ),
        }
        return {"parallel_drafts": [draft]}

    if result.ok:
        draft: WorkerDraft = {
            "worker_id": worker_id,
            "model": result.model or "unknown",
            "draft": result.content,
        }
        return {"parallel_drafts": [draft]}

    draft = {
        "worker_id": worker_id,
        "model": "unavailable",
        "draft": (
            f"{worker_id.title()} worker could not generate a draft. "
            f"Fallback reason: {result.error or 'unknown error'}."
        ),
    }
    return {"parallel_drafts": [draft]}


def _build_worker_messages(*, state: OdinState, system_prompt: str) -> list[dict[str, str]]:
    """
    Build worker input messages using history and current query.

    Args:
        state: Current Odin state.
        system_prompt: Persona/system prompt for the worker.

    Returns:
        OpenAI-style message list for LiteLLM completion.
    """
    history_block = _format_chat_history(state.get("chat_history", []))
    user_content = (
        f"Conversation history:\n{history_block}\n\n"
        f"Current user query:\n{state['query']}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_judge_messages(state: OdinState) -> list[dict[str, str]]:
    """
    Build judge input message set from history plus worker drafts.

    Args:
        state: Current Odin state.

    Returns:
        OpenAI-style message list with synthesis instructions and context.
    """
    history_block = _format_chat_history(state.get("chat_history", []))
    drafts = state.get("parallel_drafts", [])
    formatted_drafts = "\n\n".join(_format_draft_for_judge(draft) for draft in drafts)
    if not formatted_drafts:
        formatted_drafts = "No worker drafts were produced."

    judge_input = (
        f"Conversation history:\n{history_block}\n\n"
        f"User query:\n{state['query']}\n\n"
        "Council drafts:\n"
        "- Muninn (Pragmatist): execution-focused view\n"
        "- Huginn (Skeptic): risk-focused view\n\n"
        f"{formatted_drafts}\n\n"
        "Produce a final answer that resolves disagreements and makes a clear call."
    )
    return [
        {"role": "system", "content": _JUDGE_PROMPT},
        {"role": "user", "content": judge_input},
    ]


def _format_draft_for_judge(draft: dict[str, Any]) -> str:
    """
    Format one worker draft for judge prompt ingestion.

    Args:
        draft: Worker draft payload.

    Returns:
        Readable line describing worker, model, and draft text.
    """
    worker_id = str(draft.get("worker_id", "unknown"))
    model = str(draft.get("model", "unknown"))
    content = str(draft.get("draft", "")).strip()
    return f"- Worker `{worker_id}` ({model}): {content}"


def _format_chat_history(history: list[ChatMessage]) -> str:
    """
    Render recent conversation turns into compact prompt text.

    Args:
        history: Persisted chat messages from Odin state.

    Returns:
        Multi-line role-prefixed history string limited to recent entries.
    """
    if not history:
        return "No prior conversation."
    formatted = []
    for message in history[-8:]:
        role = str(message.get("role", "unknown"))
        content = str(message.get("content", "")).strip()
        formatted.append(f"- {role}: {content}")
    return "\n".join(formatted)
