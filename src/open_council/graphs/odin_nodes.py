"""Odin mode worker and judge node functions."""

from __future__ import annotations

import os
from typing import Any

from open_council.core.llm import LiteLLMClient
from open_council.state.executive import ChatMessage, OdinState, WorkerDraft

_MUNINN_PROMPT = """You are Muninn, the Constructor node in a multi-agent council.

Your job is to produce the THESIS: the strongest workable case for the user's objective.

Operating lenses:
- Builder lens: maximize execution quality and practical upside.
- Believer lens: defend the best coherent interpretation of the user's premise.
- Creator lens: when asked to generate, produce a robust, high-quality first draft.

Critical behavior rules:
1. Use conversation context from prior turns as active constraints.
2. Never fabricate facts, statistics, citations, laws, or historical events.
3. If key facts are unknown, say what is uncertain and proceed with bounded assumptions.
4. Stay domain-accurate; avoid generic boilerplate and empty template language.
5. Be structured only when it adds clarity, not because a rigid format is expected.

Tone: Confident, sharp, and constructive. Start directly with the thesis.
"""
_HUGINN_PROMPT = """You are Huginn, the Deconstructor node in a multi-agent council.

Your job is to produce the ANTITHESIS: the strongest critical challenge to the user's objective.

Operating lenses:
- Hacker lens: expose failure modes, edge cases, security and reliability risks.
- Cynic lens: surface hidden costs, political economy constraints, and second-order effects.
- Critic lens: reject bloated or naive solutions; force precision.

Critical behavior rules:
1. Use conversation context from prior turns and challenge weak carry-over assumptions.
2. Never fabricate facts, statistics, citations, laws, or historical events.
3. If evidence is missing, explicitly mark uncertainty and criticize based on plausible risk bounds.
4. Critique must be grounded and actionable, not performatively negative.
5. Be concise and forceful; avoid repetitive checklist prose.

Tone: Ruthless, analytical, and factual. Start immediately with the strongest objection.
"""
_JUDGE_PROMPT = """You are Odin, the final Synthesizer.

You receive the user query, Muninn's thesis, and Huginn's antithesis. Your task is to forge a definitive synthesis.

Non-negotiable rules:
1. No fence-sitting. Make a hard call.
2. Do not merely summarize both sides; explicitly weigh and adjudicate them.
3. Never fabricate facts, citations, or certainty. If uncertain, state the uncertainty and still make a bounded decision.
4. Preserve context continuity across turns unless the user explicitly resets scope.
5. Remove fluff and corporate filler. Every sentence must be consequential.

Output directive (fluid structure):
- Produce 2-3 Markdown headings tailored to the domain and user intent.
- Avoid static hardcoded headings.

Examples of Fluid Framing (Do not copy these exactly, invent your own based on the context):
- For a Philosophy debate: `# The Core Truth`, `# The Inherent Flaw`, `# The Synthesis`
- For an Engineering query: `# The Architecture Verdict`, `# The Hidden Bottleneck`, `# The Implementation Directive`
- For Career/Life advice: `# The Reality Check`, `# The Cost of Inaction`, `# The Strategic Move`
- For Creative Generation: `# The Thematic Vision`, `# The Output`

Analyze the user's domain, generate the most impactful headings for that specific topic, and deliver your verdict.
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
    provider_models = _resolve_node_provider_models(client, env_var_name="ODIN_MODEL")
    try:
        result = await client.complete(messages, temperature=0.1, provider_models=provider_models)
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
    model_env_var = "MUNINN_MODEL" if worker_id == "muninn" else "HUGINN_MODEL"
    provider_models = _resolve_node_provider_models(client, env_var_name=model_env_var)
    try:
        result = await client.complete(messages, temperature=0.2, provider_models=provider_models)
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
        "- Muninn (Constructor/Thesis): strongest pro-build case\n"
        "- Huginn (Deconstructor/Antithesis): strongest critical case\n\n"
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


def _resolve_node_provider_models(client: LiteLLMClient, *, env_var_name: str) -> list[tuple[str, str]]:
    """
    Build provider/model chain for a specific Odin node.

    If a node-specific model env var is set, it becomes the first attempt while
    preserving the default fallback sequence afterward.
    """
    preferred_model = os.getenv(env_var_name, "").strip()
    default_chain = list(client.provider_models)
    if not preferred_model:
        return default_chain

    provider = _infer_provider_from_model(preferred_model)
    if provider is None:
        return default_chain

    merged_chain: list[tuple[str, str]] = [(provider, preferred_model)]
    for item in default_chain:
        if item == (provider, preferred_model):
            continue
        merged_chain.append(item)
    return merged_chain


def _infer_provider_from_model(model: str) -> str | None:
    normalized = model.strip().lower()
    if normalized.startswith("groq/"):
        return "groq"
    if normalized.startswith("gemini/"):
        return "gemini"
    if normalized.startswith("ollama/"):
        return "ollama"
    return None
