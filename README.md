# Open Council 🏛️

> **Open Council - Multi-Agent LLM Orchestrator (LangGraph + LiteLLM).**

Run a council of AI models in your terminal to debate, analyze, and stress-test ideas.

Open Council is a CLI-first orchestrator where multiple agents collaborate: some build the plan, others attack the risks, and a final judge synthesizes a decisive answer.

Think:
- Research assistant that cross-checks itself
- Devil's advocate for your architecture
- AI council that debates before answering

All from a single command:

```bash
council --mode odin
```

## ⚡ What happens when you run it?

You ask:

> "Design a scalable RAG system for real-time updates."

Open Council:
- Worker models propose practical architectures (indexing, retrieval, caching)
- Critic models surface failure modes (latency, consistency, cost)
- Judge model resolves conflict into one actionable verdict

You get:
- A structured recommendation
- The key trade-off to manage
- Immediate next steps

## 🧠 Why Open Council is different

Most LLM tools are single-model chats that fail hard when a provider fails.

Open Council is built for real-world reliability:
- Multi-agent debate before final answer
- Deterministic graph workflows using LangGraph (not prompt spaghetti)
- Automatic fallback routing: Groq -> Gemini -> local Ollama
- Resilient CLI UX with setup guidance and graceful interrupt handling

### Why this can beat "just ChatGPT"

`ChatGPT`: one answer from one model.

`Open Council`: multiple perspectives, explicit trade-offs, and a final synthesis designed for high-stakes decisions and system design reviews.

## 🌟 The Pantheon (Modes)

Open Council operates through specialized agentic graphs:

- **Odin (Executive Mode):** *[Available in MVP]* Parallel workers (Muninn and Huginn) + Odin judge synthesis.
- **Artemis (Academic Mode):** *[Coming Soon]* Iterative citation-heavy research loops.
- **Leviathan (Devil's Advocate):** *[Coming Soon]* Aggressive architecture and risk stress-testing.

## 🚀 Quick Start

### Instant path (Mac/Linux)

```bash
curl -fsSL https://aayushbhaskar.github.io/OpenCouncil/install.sh | bash
council --mode odin
```

This installs Open Council under `~/.open-council-app` and links `council` to `~/.local/bin`.

### One-minute wow test

Run:

```bash
council --mode odin
```

Then ask:

```text
Analyze microservices vs monolith for my startup.
```

Example response shape:

```text
The Verdict: Start with a modular monolith to ship faster and defer distributed complexity.
The Critical Trade-off: Initial velocity now vs migration cost later.
The Path Forward:
1) Define strict module boundaries and contracts today.
2) Instrument core performance paths and set scaling thresholds.
3) Extract the first service only when measured load exceeds those thresholds.
```

### Current MVP capabilities

- Odin mode LangGraph pipeline with Muninn + Huginn workers and Odin judge
- Async LiteLLM routing with Groq -> Gemini -> Ollama fallback (`open_council.core.llm`)
- Interactive CLI REPL with `/exit` and `/quit`
- Graceful Ctrl+C handling (first press warns, second exits cleanly)
- First-run setup wizard using `~/.open-council/.env` (temporary local `.env` fallback supported)
- Ollama readiness checks (binary, server, model) with actionable guidance

### Optional local fallback engine (Ollama)

Install Ollama from [Ollama Downloads](https://ollama.com/download), then:

```bash
# 1) Start local Ollama server
ollama serve

# 2) Pull the configured fallback model (default)
ollama pull llama3.1
```

If your configured model differs, pull that exact model name from `OLLAMA_MODEL`
(example: `OLLAMA_MODEL=ollama/llama3.1` maps to `ollama pull llama3.1`).

If `~/.open-council/.env` is missing, Open Council launches a first-run wizard to collect keys and run provider readiness checks before chat starts.

### Dev setup (Python 3.11+)

```bash
git clone https://github.com/aayushbhaskar/OpenCouncil.git
cd OpenCouncil
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q tests/test_llm_client.py
council --mode odin
```

## 🏗️ Architecture and resilience

Open Council is designed for graceful degradation:
- Network throttling via strict `asyncio.Semaphore`
- Tiered provider fallback using LiteLLM
- Deterministic orchestration with typed state in LangGraph

Local checkpointing is planned for a later phase (SQLite-backed, not wired in MVP yet).

## 🗺️ Roadmap

- [ ] Phase 1: Resilient MVP (Odin mode, LiteLLM routing, Rich CLI)
- [ ] Phase 2: Deep reasoning (Artemis mode, SQLite memory, web tools)
- [ ] Phase 3: Enterprise scale (Leviathan mode, local vector memory, cloud backends)
- [ ] Phase 4: Workstation layer (Ariadne mode, secure local file workflows)

## 🔎 Discoverability

Suggested GitHub topics:
`llm`, `multi-agent`, `langgraph`, `ai-orchestration`, `rag`, `llm-evaluation`, `cli-tool`, `ollama`