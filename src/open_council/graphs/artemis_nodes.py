"""Artemis Academic mode nodes: Planner, Specialists, Critic, Synthesis."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Literal

from open_council.core.llm import LLMResult, LiteLLMClient
from open_council.state.artemis import ArtemisState
from open_council.state.executive import WorkerExtraction, WorkerSearchHit
from open_council.tools.jina_reader import JinaReader
from open_council.tools.search_resolver import resolve_search_provider

_MAX_QUERIES = 3
_MAX_RESULTS = 5
_MAX_PAGES = 5
_MAX_CHARS = 2000
_MAX_REVISIONS = 3
_LLM_TIMEOUT = 45.0

_PLANNER_PROMPT = """You are the Artemis Planner, the first stage of a deep-research council.

Your job is to turn the user's question into a research plan that a specialist can execute
with web search, and that a critic can later check the draft against.

Planning rules:
1. Decompose the question into 2-4 concrete research objectives. Each objective must be
   checkable: a critic should be able to say whether a draft answered it or not.
2. Prefer objectives that pin down definitions, primary evidence, competing positions,
   and the current state of knowledge (dates, versions, figures) over vague themes.
3. Write up to 3 search queries that target sources able to settle those objectives:
   primary sources, official documentation, peer-reviewed work, reputable reporting.
   Queries must be specific (names, terms, years) rather than restating the question.
4. If prior critic feedback is provided, this is a revision round. Re-plan around the
   critic's gaps: change or add objectives and queries that close them. Do not repeat
   queries that already failed to produce evidence.
5. Respect conversation history: if the question is a follow-up, plan for the delta,
   not the whole topic again.
6. Do not answer the question here, and do not invent sources, authors, or titles.

Output format (exactly these two section labels, nothing before PLAN:):
PLAN:
- <objective 1>
- <objective 2>
QUERIES:
<query 1>
<query 2>
<query 3>
"""

_SPECIALIST_PROMPT = """You are the Artemis Research Specialist.

You receive the user's question, the Planner's objectives, gathered web evidence (search hits
and extracted page text with URLs), and, on revision rounds, the Critic's feedback.
Your job is to write the research draft that answers the question rigorously.

Evidence rules (non-negotiable):
1. Every factual claim that is not common knowledge must be traceable to the provided
   evidence. Cite it inline as a markdown link using the exact URL from the evidence,
   e.g. ([source](https://example.com/page)).
2. Never invent citations, URLs, authors, statistics, dates, or quotes. If the evidence
   does not support a point, either omit it or state plainly that it is unverified.
3. When sources disagree, present the disagreement and say which source is stronger and why
   (primary vs secondary, recency, methodology).
4. If evidence is thin or missing for an objective, say so explicitly under that objective
   instead of filling the gap with general knowledge presented as fact.

Draft rules:
5. Address every Planner objective. Use the objectives as your structure.
6. On revision rounds, fix every point the Critic raised. Do not just rephrase the old draft.
7. Preserve conversation context: follow-ups build on earlier answers unless the user resets.
8. Be precise and dense. No filler, no preamble like "Here is the research".
"""

_CRITIC_PROMPT = """You are the Artemis Critic, the quality gate of the research loop.

You receive the user's question, the Planner's objectives, the gathered evidence, and the
Specialist's draft. Decide whether the draft is good enough to ship, or must be revised.

Reject the draft if any of these hold:
1. A Planner objective is unanswered or only answered vaguely.
2. A material factual claim has no supporting citation, or cites a URL not present in
   the evidence.
3. A claim contradicts or overstates what the cited evidence actually says.
4. Conflicting sources were ignored, or uncertainty was hidden.
5. The draft drifts away from the user's actual question.

Accept the draft if it answers all objectives with grounded, honestly qualified claims,
even when some uncertainty remains and is clearly stated. Do not reject for style alone,
and do not demand evidence that the search budget plainly cannot find.

Your feedback is the Planner's and Specialist's only guidance for the next round, so make it
actionable: name the specific gaps, the unsupported claims, and what evidence or search
direction would fix them.

Output format (exactly):
VERDICT: ACCEPT|REJECT
FEEDBACK: <specific, actionable paragraph or short bullet list>
"""

_SYNTHESIS_PROMPT = """You are Artemis, the final synthesizer of the deep-research council.

You receive the user's question, the Planner's objectives, the Specialist's final draft,
and the Critic's last verdict and feedback. Produce the answer the user will read.

Non-negotiable rules:
1. Keep only claims supported by the draft's citations. Preserve those markdown links
   exactly; never add new citations or URLs that are not in the draft.
2. If the Critic's last verdict was REJECT (revision cap reached), state clearly which
   points remain unverified or contested instead of presenting them as settled.
3. Lead with the direct answer to the question, then the supporting evidence.
4. State uncertainty explicitly and briefly where it exists. Do not hedge everything.
5. Preserve conversation continuity for follow-up questions.
6. No filler, no "In conclusion", no restating the question.

Output directive (fluid structure):
- Use 2-3 Markdown headings tailored to the question and domain (avoid static templates).
- End with a short "Sources" list of the URLs actually cited.
"""


async def artemis_planner_node(state: ArtemisState) -> dict[str, Any]:
    history = _format_history(state.get("chat_history", []))
    feedback = state.get("critic_feedback", "").strip()
    revision = int(state.get("revision_count", 0) or 0)
    user = (
        f"Conversation history:\n{history}\n\n"
        f"User query:\n{state['query']}\n\n"
        f"Revision count: {revision}\n"
        f"Prior critic feedback:\n{feedback or '(none)'}\n"
    )
    result = await _complete(
        [
            {"role": "system", "content": _PLANNER_PROMPT},
            {"role": "user", "content": user},
        ],
        env_var="ARTEMIS_MODEL",
        max_tokens=350,
    )
    if not result.ok:
        return {
            "plan": f"Fallback plan for: {state['query']}",
            "search_queries": [state["query"]],
        }
    plan, queries = _parse_plan(result.content)
    return {
        "plan": plan or result.content.strip(),
        "search_queries": queries or [state["query"]],
    }


async def artemis_search_node(state: ArtemisState) -> dict[str, Any]:
    provider = resolve_search_provider(use_cache=True)
    queries = list(state.get("search_queries", []) or [state["query"]])[:_MAX_QUERIES]
    seen: set[str] = set()
    hits: list[WorkerSearchHit] = []
    for query in queries:
        results = await provider.search(query, max_results=_MAX_RESULTS)
        for item in results:
            url = item.url.strip()
            if not url or url in seen:
                continue
            seen.add(url)
            hits.append(
                WorkerSearchHit(
                    title=item.title.strip(),
                    url=url,
                    snippet=item.snippet.strip(),
                    source=item.source.strip(),
                )
            )
            if len(hits) >= _MAX_PAGES:
                break
        if len(hits) >= _MAX_PAGES:
            break
    return {"search_hits": hits}


async def artemis_extract_node(state: ArtemisState) -> dict[str, Any]:
    hits = state.get("search_hits", []) or []
    if not hits:
        return {"extracted_pages": []}
    reader = JinaReader(timeout_seconds=10.0, use_cache=True)
    pages: list[WorkerExtraction] = []
    for hit in hits[:_MAX_PAGES]:
        url = hit.get("url", "").strip()
        if not url:
            continue
        try:
            content = await asyncio.wait_for(reader.fetch_markdown(url), timeout=8.0)
        except Exception as exc:  # noqa: BLE001
            content = f"[reader_error] {exc}"
        compact = " ".join(content.split())[:_MAX_CHARS]
        pages.append(WorkerExtraction(url=url, content=compact))
    return {"extracted_pages": pages}


async def artemis_specialist_draft_node(state: ArtemisState) -> dict[str, Any]:
    evidence = _evidence_block(state)
    history = _format_history(state.get("chat_history", []))
    user = (
        f"Conversation history:\n{history}\n\n"
        f"User query:\n{state['query']}\n\n"
        f"Plan:\n{state.get('plan', '')}\n\n"
        f"Critic feedback to address:\n{state.get('critic_feedback', '') or '(none)'}\n\n"
        f"Evidence:\n{evidence}\n"
    )
    result = await _complete(
        [
            {"role": "system", "content": _SPECIALIST_PROMPT},
            {"role": "user", "content": user},
        ],
        env_var="ARTEMIS_MODEL",
        max_tokens=700,
    )
    if result.ok:
        return {"specialist_draft": result.content.strip()}
    return {
        "specialist_draft": (
            "Specialist draft unavailable. "
            f"Fallback reason: {result.error or 'unknown error'}."
        )
    }


async def artemis_critic_node(state: ArtemisState) -> dict[str, Any]:
    evidence = _evidence_block(state)
    user = (
        f"User query:\n{state['query']}\n\n"
        f"Plan:\n{state.get('plan', '')}\n\n"
        f"Specialist draft:\n{state.get('specialist_draft', '')}\n\n"
        f"Evidence:\n{evidence}\n"
    )
    result = await _complete(
        [
            {"role": "system", "content": _CRITIC_PROMPT},
            {"role": "user", "content": user},
        ],
        env_var="ARTEMIS_MODEL",
        max_tokens=250,
        temperature=0.0,
    )
    revision = int(state.get("revision_count", 0) or 0)
    if not result.ok:
        return {
            "critic_verdict": "ACCEPT",
            "critic_feedback": f"Critic unavailable; accepting draft. ({result.error})",
            "critic_accepted": True,
            "revision_count": revision,
        }
    accepted, feedback = _parse_critic(result.content)
    next_revision = revision if accepted else revision + 1
    return {
        "critic_verdict": "ACCEPT" if accepted else "REJECT",
        "critic_feedback": feedback,
        "critic_accepted": accepted,
        "revision_count": next_revision,
    }


async def artemis_synthesize_node(state: ArtemisState) -> dict[str, str]:
    user = (
        f"User query:\n{state['query']}\n\n"
        f"Plan:\n{state.get('plan', '')}\n\n"
        f"Specialist draft:\n{state.get('specialist_draft', '')}\n\n"
        f"Critic verdict: {state.get('critic_verdict', '')}\n"
        f"Critic feedback:\n{state.get('critic_feedback', '')}\n"
        f"Revisions used: {state.get('revision_count', 0)}\n"
    )
    result = await _complete(
        [
            {"role": "system", "content": _SYNTHESIS_PROMPT},
            {"role": "user", "content": user},
        ],
        env_var="ARTEMIS_MODEL",
        max_tokens=800,
        temperature=0.1,
    )
    if result.ok:
        return {"final_synthesis": result.content.strip()}
    draft = state.get("specialist_draft", "").strip()
    return {
        "final_synthesis": draft
        or f"Artemis could not synthesize. Fallback reason: {result.error or 'unknown error'}."
    }


def route_after_critic(state: ArtemisState) -> Literal["revise", "synthesize"]:
    accepted = bool(state.get("critic_accepted", False))
    revision = int(state.get("revision_count", 0) or 0)
    if accepted or revision >= _MAX_REVISIONS:
        return "synthesize"
    return "revise"


async def _complete(
    messages: list[dict[str, str]],
    *,
    env_var: str,
    max_tokens: int,
    temperature: float = 0.2,
) -> LLMResult:
    client = LiteLLMClient()
    preferred = os.getenv(env_var, "").strip()
    provider_models = list(client.provider_models)
    if preferred:
        provider = _infer_provider(preferred)
        if provider:
            provider_models = [(provider, preferred)] + [
                item for item in provider_models if item != (provider, preferred)
            ]
    try:
        return await asyncio.wait_for(
            client.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                provider_models=provider_models,
            ),
            timeout=_LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return LLMResult(
            ok=False,
            content="",
            provider=None,
            model=None,
            attempts=[],
            error="artemis step timed out",
        )


def _infer_provider(model: str) -> str | None:
    normalized = model.strip().lower()
    if normalized.startswith("groq/"):
        return "groq"
    if normalized.startswith("openrouter/"):
        return "openrouter"
    if normalized.startswith("gemini/"):
        return "gemini"
    if normalized.startswith("ollama/"):
        return "ollama"
    return None


def _parse_plan(content: str) -> tuple[str, list[str]]:
    plan_match = re.search(r"PLAN:\s*(.*?)(?:QUERIES:|$)", content, flags=re.S | re.I)
    queries_match = re.search(r"QUERIES:\s*(.*)$", content, flags=re.S | re.I)
    plan = (plan_match.group(1).strip() if plan_match else content.strip())
    queries: list[str] = []
    if queries_match:
        for line in queries_match.group(1).splitlines():
            cleaned = line.strip().lstrip("-*0123456789. ").strip()
            if cleaned:
                queries.append(cleaned)
            if len(queries) >= _MAX_QUERIES:
                break
    return plan, queries


def _parse_critic(content: str) -> tuple[bool, str]:
    verdict_match = re.search(r"VERDICT:\s*(ACCEPT|REJECT)", content, flags=re.I)
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", content, flags=re.S | re.I)
    accepted = bool(verdict_match and verdict_match.group(1).upper() == "ACCEPT")
    feedback = (feedback_match.group(1).strip() if feedback_match else content.strip())
    return accepted, feedback or "No feedback provided."


def _evidence_block(state: ArtemisState) -> str:
    hits = state.get("search_hits", []) or []
    pages = state.get("extracted_pages", []) or []
    if not hits and not pages:
        return "No web evidence gathered."
    lines: list[str] = []
    if hits:
        lines.append("Search hits:")
        for hit in hits:
            lines.append(
                f"- {hit.get('title', '')} | {hit.get('url', '')} | {hit.get('snippet', '')}"
            )
    if pages:
        lines.append("Extracted pages:")
        for page in pages:
            lines.append(f"- {page.get('url', '')}: {page.get('content', '')}")
    return "\n".join(lines)


def _format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "No prior conversation."
    lines = []
    for message in history[-8:]:
        lines.append(f"- {message.get('role', 'unknown')}: {str(message.get('content', '')).strip()}")
    return "\n".join(lines)
