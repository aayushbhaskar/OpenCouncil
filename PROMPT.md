# Open Council: Architecture & Coding Standards

## Project Overview
Open Council is a CLI-first, open-source multi-agent LLM orchestration tool designed to replicate Perplexity's "Council of Models." It utilizes a "Bring Your Own Key" (BYOK) hybrid model, mixing local offline models with cloud-based endpoints.

## Core Tech Stack
* **Language:** Python 3.11+
* **Orchestration:** LangGraph (State management, cyclic/acyclic routing)
* **LLM Routing:** LiteLLM (Universal API translation)
* **Concurrency:** `asyncio`
* **CLI UI:** `Rich` (Live displays, progress spinners, markdown rendering)

## Strict Coding Standards
1. **Async Everything:** All network calls, file I/O, and LLM generations MUST be asynchronous (`async def`, `await`). Never use blocking standard library calls (like `requests`). Use `aiohttp` or `httpx` for standard web requests.
2. **The Semaphore Rule:** Never make raw API calls directly from a LangGraph node. All LLM calls and Search calls MUST be routed through our `core/throttle.py` which wraps them in an `asyncio.Semaphore` to prevent 429 Rate Limits.
3. **Graceful Fallbacks:** The system must never crash on an LLM API failure. Use LiteLLM's built-in fallback array (Primary -> Secondary -> Local Ollama).
4. **No Raw Prints:** Standard `print()` statements will corrupt the async terminal UI. All CLI output must be routed through the `Rich` console components defined in `src/open_council/ui/`.
5. **Type Hinting:** Enforce strict Python typing (`typing.Dict`, `typing.List`, `typing.Annotated`). LangGraph state relies entirely on accurate `TypedDict` and `Pydantic` schemas.

## Graph Design Philosophy
* Nodes should be pure functions that take a `State` dict, perform an async action, and return a dictionary containing ONLY the keys they intend to update.
* Do not mutate the state inside the node directly; let LangGraph handle the state merging.