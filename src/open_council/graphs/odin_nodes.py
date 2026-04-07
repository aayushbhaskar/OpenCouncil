"""Odin mode worker and judge node functions."""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any

from open_council.core.llm import LLMResult, LiteLLMClient
from open_council.state.executive import (
    ChatMessage,
    OdinState,
    WorkerDraft,
    WorkerExtraction,
    WorkerRetrievalState,
    WorkerSearchHit,
)
from open_council.tools.jina_reader import JinaReader
from open_council.tools.search_ddg import DuckDuckGoSearchProvider

_MAX_QUERIES_PER_WORKER = 2
_MAX_SEARCH_RESULTS_PER_QUERY = 5
_MAX_EXTRACTED_PAGES_PER_WORKER = 5
_MAX_EXTRACT_CHARS_PER_PAGE = 2000
_WORKER_LLM_WALL_TIMEOUT_SECONDS = 45.0
_JUDGE_LLM_WALL_TIMEOUT_SECONDS = 45.0

_MUNINN_REASON_PROMPT = """You are Muninn (Thesis/Constructor).
Decide if fresh web search is required to answer the current user query well.
Return exactly:
SEARCH_NEEDED: yes|no
REASON: <one concise sentence>
Use search when recency, factual precision, or external specifics are needed."""

_HUGINN_REASON_PROMPT = """You are Huginn (Antithesis/Deconstructor).
Decide if fresh web search is required to challenge assumptions and surface risks.
Return exactly:
SEARCH_NEEDED: yes|no
REASON: <one concise sentence>
Use search when factual verification or external evidence strengthens critique."""

_QUERY_PROMPT = """Generate up to 2 high-signal web queries for this task.
Output one query per line only. No numbering. No commentary."""

_MUNINN_DRAFT_PROMPT = """You are Muninn, the thesis/constructor.
Build the strongest actionable pro-build case.
Ground claims in provided evidence. If evidence is weak, state uncertainty clearly.
No fabricated citations or data."""

_HUGINN_DRAFT_PROMPT = """You are Huginn, the antithesis/deconstructor.
Deliver the strongest grounded critique and risk analysis.
Ground claims in provided evidence. If evidence is weak, state uncertainty clearly.
No fabricated citations or data."""

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
"""


async def muninn_reason_gate_node(state: OdinState) -> dict[str, WorkerRetrievalState]:
    return await _reason_gate_node(state=state, worker_id="muninn", system_prompt=_MUNINN_REASON_PROMPT)


async def muninn_query_gen_node(state: OdinState) -> dict[str, WorkerRetrievalState]:
    return await _query_gen_node(state=state, worker_id="muninn")


async def muninn_search_node(state: OdinState) -> dict[str, WorkerRetrievalState]:
    return await _search_node(state=state, worker_id="muninn")


async def muninn_extract_node(state: OdinState) -> dict[str, WorkerRetrievalState]:
    return await _extract_node(state=state, worker_id="muninn")


async def muninn_reason_refine_node(state: OdinState) -> dict[str, WorkerRetrievalState]:
    return await _reason_refine_node(state=state, worker_id="muninn")


async def muninn_draft_node(state: OdinState) -> dict[str, list[WorkerDraft]]:
    return await _draft_node(state=state, worker_id="muninn", system_prompt=_MUNINN_DRAFT_PROMPT)


async def huginn_reason_gate_node(state: OdinState) -> dict[str, WorkerRetrievalState]:
    return await _reason_gate_node(state=state, worker_id="huginn", system_prompt=_HUGINN_REASON_PROMPT)


async def huginn_query_gen_node(state: OdinState) -> dict[str, WorkerRetrievalState]:
    return await _query_gen_node(state=state, worker_id="huginn")


async def huginn_search_node(state: OdinState) -> dict[str, WorkerRetrievalState]:
    return await _search_node(state=state, worker_id="huginn")


async def huginn_extract_node(state: OdinState) -> dict[str, WorkerRetrievalState]:
    return await _extract_node(state=state, worker_id="huginn")


async def huginn_reason_refine_node(state: OdinState) -> dict[str, WorkerRetrievalState]:
    return await _reason_refine_node(state=state, worker_id="huginn")


async def huginn_draft_node(state: OdinState) -> dict[str, list[WorkerDraft]]:
    return await _draft_node(state=state, worker_id="huginn", system_prompt=_HUGINN_DRAFT_PROMPT)


async def pragmatic_worker_node(state: OdinState) -> dict[str, list[WorkerDraft]]:
    """
    Compatibility wrapper for legacy test paths.
    """
    return await muninn_draft_node(state)


async def skeptical_worker_node(state: OdinState) -> dict[str, list[WorkerDraft]]:
    """
    Compatibility wrapper for legacy test paths.
    """
    return await huginn_draft_node(state)


async def _reason_gate_node(
    *,
    state: OdinState,
    worker_id: str,
    system_prompt: str,
) -> dict[str, WorkerRetrievalState]:
    started = time.monotonic()
    _trace(f"{worker_id}.reason_gate.start")
    context = _get_worker_context(state, worker_id=worker_id)
    history_block = _format_chat_history(state.get("chat_history", []))
    user_prompt = (
        f"Conversation history:\n{history_block}\n\n"
        f"Current user query:\n{state['query']}"
    )
    result = await _complete_worker_llm(
        worker_id=worker_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    if not result.ok:
        context["needs_search"] = False
        context["reasoning_note"] = f"Search decision fallback: {result.error or 'unknown error'}."
        _trace(
            f"{worker_id}.reason_gate.end",
            elapsed_s=round(time.monotonic() - started, 2),
            needs_search=False,
            fallback=True,
        )
        return _worker_context_update(worker_id=worker_id, context=context)

    context["needs_search"], context["reasoning_note"] = _parse_search_decision(result.content)
    _trace(
        f"{worker_id}.reason_gate.end",
        elapsed_s=round(time.monotonic() - started, 2),
        needs_search=context["needs_search"],
        fallback=False,
    )
    return _worker_context_update(worker_id=worker_id, context=context)


async def _query_gen_node(*, state: OdinState, worker_id: str) -> dict[str, WorkerRetrievalState]:
    started = time.monotonic()
    _trace(f"{worker_id}.query_gen.start")
    context = _get_worker_context(state, worker_id=worker_id)
    if not context.get("needs_search", False):
        context["search_queries"] = []
        _trace(
            f"{worker_id}.query_gen.end",
            elapsed_s=round(time.monotonic() - started, 2),
            skipped=True,
            query_count=0,
        )
        return _worker_context_update(worker_id=worker_id, context=context)

    history_block = _format_chat_history(state.get("chat_history", []))
    user_prompt = (
        f"Conversation history:\n{history_block}\n\n"
        f"Current user query:\n{state['query']}\n\n"
        f"Search rationale:\n{context.get('reasoning_note', '')}\n\n"
        f"{_QUERY_PROMPT}"
    )
    result = await _complete_worker_llm(
        worker_id=worker_id,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.1,
        max_tokens=120,
    )
    if not result.ok:
        context["search_queries"] = [state["query"]][:1]
        _trace(
            f"{worker_id}.query_gen.end",
            elapsed_s=round(time.monotonic() - started, 2),
            skipped=False,
            fallback=True,
            query_count=len(context["search_queries"]),
        )
        return _worker_context_update(worker_id=worker_id, context=context)

    parsed_queries = _parse_queries(result.content, limit=_MAX_QUERIES_PER_WORKER)
    context["search_queries"] = parsed_queries or [state["query"]][:1]
    _trace(
        f"{worker_id}.query_gen.end",
        elapsed_s=round(time.monotonic() - started, 2),
        skipped=False,
        fallback=False,
        query_count=len(context["search_queries"]),
    )
    return _worker_context_update(worker_id=worker_id, context=context)


async def _search_node(*, state: OdinState, worker_id: str) -> dict[str, WorkerRetrievalState]:
    started = time.monotonic()
    _trace(f"{worker_id}.search.start")
    context = _get_worker_context(state, worker_id=worker_id)
    if not context.get("needs_search", False):
        context["search_hits"] = []
        _trace(
            f"{worker_id}.search.end",
            elapsed_s=round(time.monotonic() - started, 2),
            skipped=True,
            hit_count=0,
        )
        return _worker_context_update(worker_id=worker_id, context=context)

    provider = DuckDuckGoSearchProvider()
    seen_urls: set[str] = set()
    hits: list[WorkerSearchHit] = []
    for query in context.get("search_queries", [])[:_MAX_QUERIES_PER_WORKER]:
        results = await provider.search(query, max_results=_MAX_SEARCH_RESULTS_PER_QUERY)
        for item in results:
            url = item.url.strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            hits.append(
                WorkerSearchHit(
                    title=item.title.strip(),
                    url=url,
                    snippet=item.snippet.strip(),
                    source=item.source.strip(),
                )
            )
            if len(hits) >= _MAX_EXTRACTED_PAGES_PER_WORKER:
                break
        if len(hits) >= _MAX_EXTRACTED_PAGES_PER_WORKER:
            break
    context["search_hits"] = hits
    _trace(
        f"{worker_id}.search.end",
        elapsed_s=round(time.monotonic() - started, 2),
        skipped=False,
        query_count=len(context.get("search_queries", [])),
        hit_count=len(hits),
    )
    return _worker_context_update(worker_id=worker_id, context=context)


async def _extract_node(*, state: OdinState, worker_id: str) -> dict[str, WorkerRetrievalState]:
    started = time.monotonic()
    _trace(f"{worker_id}.extract.start")
    context = _get_worker_context(state, worker_id=worker_id)
    hits = context.get("search_hits", [])
    if not context.get("needs_search", False) or not hits:
        context["extracted_pages"] = []
        _trace(
            f"{worker_id}.extract.end",
            elapsed_s=round(time.monotonic() - started, 2),
            skipped=True,
            extracted_count=0,
        )
        return _worker_context_update(worker_id=worker_id, context=context)

    reader = JinaReader(timeout_seconds=10.0)
    extracted: list[WorkerExtraction] = []
    for hit in hits[:_MAX_EXTRACTED_PAGES_PER_WORKER]:
        url = hit.get("url", "").strip()
        if not url:
            continue
        try:
            content = await reader.fetch_markdown(url)
        except Exception as exc:  # noqa: BLE001
            content = f"[reader_error] Could not read {url}: {exc}"
        compact = _compact_text(content, limit=_MAX_EXTRACT_CHARS_PER_PAGE)
        extracted.append(WorkerExtraction(url=url, content=compact))
    context["extracted_pages"] = extracted
    _trace(
        f"{worker_id}.extract.end",
        elapsed_s=round(time.monotonic() - started, 2),
        skipped=False,
        extracted_count=len(extracted),
    )
    return _worker_context_update(worker_id=worker_id, context=context)


async def _reason_refine_node(*, state: OdinState, worker_id: str) -> dict[str, WorkerRetrievalState]:
    started = time.monotonic()
    _trace(f"{worker_id}.reason_refine.start")
    context = _get_worker_context(state, worker_id=worker_id)
    if not context.get("needs_search", False):
        context["refined_reasoning"] = context.get("reasoning_note", "")
        _trace(
            f"{worker_id}.reason_refine.end",
            elapsed_s=round(time.monotonic() - started, 2),
            skipped=True,
        )
        return _worker_context_update(worker_id=worker_id, context=context)

    evidence_block = _build_evidence_block(context)
    result = await _complete_worker_llm(
        worker_id=worker_id,
        messages=[
            {
                "role": "user",
                "content": (
                    "Refine the worker's reasoning using the gathered web evidence.\n"
                    "Keep it concise and factual. Mark uncertainty when evidence is weak.\n\n"
                    f"User query:\n{state['query']}\n\n"
                    f"Evidence:\n{evidence_block}"
                ),
            }
        ],
        temperature=0.1,
        max_tokens=220,
    )
    if result.ok:
        context["refined_reasoning"] = result.content.strip()
    else:
        context["refined_reasoning"] = (
            f"Evidence refinement unavailable: {result.error or 'unknown error'}."
        )
    _trace(
        f"{worker_id}.reason_refine.end",
        elapsed_s=round(time.monotonic() - started, 2),
        fallback=not result.ok,
    )
    return _worker_context_update(worker_id=worker_id, context=context)


async def _draft_node(
    *,
    state: OdinState,
    worker_id: str,
    system_prompt: str,
) -> dict[str, list[WorkerDraft]]:
    started = time.monotonic()
    _trace(f"{worker_id}.draft.start")
    context = _get_worker_context(state, worker_id=worker_id)
    history_block = _format_chat_history(state.get("chat_history", []))
    evidence_block = _build_evidence_block(context)
    refined_reasoning = context.get("refined_reasoning", "").strip()
    notes = refined_reasoning or context.get("reasoning_note", "")

    result = await _complete_worker_llm(
        worker_id=worker_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_block}\n\n"
                    f"Current user query:\n{state['query']}\n\n"
                    f"Search required: {context.get('needs_search', False)}\n"
                    f"Search reasoning note: {context.get('reasoning_note', '')}\n"
                    f"Post-retrieval reasoning: {notes}\n\n"
                    f"Evidence block:\n{evidence_block}\n\n"
                    "Produce the worker draft for Odin."
                ),
            },
        ],
        temperature=0.2,
        max_tokens=500,
    )

    if result.ok:
        _trace(
            f"{worker_id}.draft.end",
            elapsed_s=round(time.monotonic() - started, 2),
            fallback=False,
        )
        return {
            "parallel_drafts": [
                WorkerDraft(
                    worker_id=worker_id,
                    model=result.model or "unknown",
                    draft=result.content,
                )
            ]
        }
    _trace(
        f"{worker_id}.draft.end",
        elapsed_s=round(time.monotonic() - started, 2),
        fallback=True,
    )
    return {
        "parallel_drafts": [
            WorkerDraft(
                worker_id=worker_id,
                model="unavailable",
                draft=(
                    f"{worker_id.title()} worker could not generate a draft. "
                    f"Fallback reason: {result.error or 'unknown error'}."
                ),
            )
        ]
    }


async def judge_node(state: OdinState) -> dict[str, str]:
    """
    Synthesize worker drafts into the final Odin verdict.
    """
    started = time.monotonic()
    _trace("odin.judge.start")
    client = LiteLLMClient()
    messages = _build_judge_messages(state)
    provider_models = _resolve_node_provider_models(client, env_var_name="ODIN_MODEL")
    try:
        result = await asyncio.wait_for(
            client.complete(
                messages,
                temperature=0.1,
                max_tokens=650,
                provider_models=provider_models,
            ),
            timeout=_JUDGE_LLM_WALL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return {
            "final_synthesis": (
                "Odin judge could not complete synthesis. "
                "Fallback reason: judge step timed out."
            )
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "final_synthesis": (
                "Odin judge could not complete synthesis. "
                f"Unexpected error: {exc}."
            )
        }

    if result.ok:
        _trace("odin.judge.end", elapsed_s=round(time.monotonic() - started, 2), fallback=False)
        return {"final_synthesis": result.content}
    _trace("odin.judge.end", elapsed_s=round(time.monotonic() - started, 2), fallback=True)
    return {
        "final_synthesis": (
            "Odin judge could not complete synthesis. "
            f"Fallback reason: {result.error or 'unknown error'}."
        )
    }


async def _complete_worker_llm(
    *,
    worker_id: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> LLMResult:
    client = LiteLLMClient()
    model_env_var = "MUNINN_MODEL" if worker_id == "muninn" else "HUGINN_MODEL"
    provider_models = _resolve_node_provider_models(client, env_var_name=model_env_var)
    try:
        return await asyncio.wait_for(
            client.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                provider_models=provider_models,
            ),
            timeout=_WORKER_LLM_WALL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return LLMResult(
            ok=False,
            content="",
            provider=None,
            model=None,
            attempts=[],
            error="worker step timed out",
        )
    except Exception as exc:  # noqa: BLE001
        return LLMResult(
            ok=False,
            content="",
            provider=None,
            model=None,
            attempts=[],
            error=str(exc),
        )


def _parse_search_decision(content: str) -> tuple[bool, str]:
    needed_match = re.search(r"SEARCH_NEEDED:\s*(yes|no)", content, flags=re.I)
    reason_match = re.search(r"REASON:\s*(.+)", content, flags=re.I)
    needs_search = bool(needed_match and needed_match.group(1).lower() == "yes")
    reason = (reason_match.group(1).strip() if reason_match else "").strip()
    if not reason:
        reason = "No explicit reasoning note provided."
    return needs_search, reason


def _parse_queries(content: str, *, limit: int) -> list[str]:
    candidates = []
    for line in content.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip()
        if cleaned:
            candidates.append(cleaned)
    unique: list[str] = []
    seen: set[str] = set()
    for query in candidates:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(query)
        if len(unique) >= limit:
            break
    return unique


def _build_evidence_block(context: WorkerRetrievalState) -> str:
    hits = context.get("search_hits", [])
    extracted = context.get("extracted_pages", [])
    if not hits and not extracted:
        return "No web evidence gathered."

    lines = []
    if hits:
        lines.append("Search hits:")
        for hit in hits:
            lines.append(
                f"- {hit.get('title', 'Untitled')} | {hit.get('url', '')} | {hit.get('snippet', '')}"
            )
    if extracted:
        lines.append("\nExtracted pages:")
        for page in extracted:
            lines.append(f"- {page.get('url', '')}: {page.get('content', '')}")
    return "\n".join(lines)


def _compact_text(content: str, *, limit: int) -> str:
    collapsed = " ".join(content.split())
    return collapsed[:limit].strip()


def _get_worker_context(state: OdinState, *, worker_id: str) -> WorkerRetrievalState:
    key = _worker_context_key(worker_id)
    raw = state.get(key)
    if isinstance(raw, dict):
        return {
            "needs_search": bool(raw.get("needs_search", False)),
            "reasoning_note": str(raw.get("reasoning_note", "")),
            "search_queries": list(raw.get("search_queries", [])),
            "search_hits": list(raw.get("search_hits", [])),
            "extracted_pages": list(raw.get("extracted_pages", [])),
            "refined_reasoning": str(raw.get("refined_reasoning", "")),
        }
    return {
        "needs_search": False,
        "reasoning_note": "",
        "search_queries": [],
        "search_hits": [],
        "extracted_pages": [],
        "refined_reasoning": "",
    }


def _worker_context_update(
    *,
    worker_id: str,
    context: WorkerRetrievalState,
) -> dict[str, WorkerRetrievalState]:
    return {str(_worker_context_key(worker_id)): context}


def _worker_context_key(worker_id: str) -> str:
    if worker_id == "muninn":
        return "muninn_retrieval"
    return "huginn_retrieval"


def _build_judge_messages(state: OdinState) -> list[dict[str, str]]:
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
    worker_id = str(draft.get("worker_id", "unknown"))
    model = str(draft.get("model", "unknown"))
    content = str(draft.get("draft", "")).strip()
    return f"- Worker `{worker_id}` ({model}): {content}"


def _format_chat_history(history: list[ChatMessage]) -> str:
    if not history:
        return "No prior conversation."
    formatted = []
    for message in history[-8:]:
        role = str(message.get("role", "unknown"))
        content = str(message.get("content", "")).strip()
        formatted.append(f"- {role}: {content}")
    return "\n".join(formatted)


def _resolve_node_provider_models(client: LiteLLMClient, *, env_var_name: str) -> list[tuple[str, str]]:
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


def _trace(event: str, **fields: Any) -> None:
    """
    Reserved hook for temporary node-level diagnostics.
    """
    _ = event, fields
