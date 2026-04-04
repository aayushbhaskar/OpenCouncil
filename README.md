# Open Council 🏛️

> **A private, multi-agent terminal workstation for deep research and system stress-testing.**

Open Council is a CLI-first, open-source AI orchestrator. It replicates enterprise-grade "Council of Models" workflows by utilizing a deterministic LangGraph state machine. It seamlessly mixes zero-cost local models (Ollama) with cloud endpoints (Groq, Gemini, Bedrock) to deliver deep reasoning, code analysis, and system architecture stress-testing right in your terminal.

## 🌟 The Pantheon (Modes)

Open Council operates through specialized agentic graphs, invoked via the CLI:

* **Odin (Executive Mode):** *[MVP In Progress]* Fast, overarching synthesis. Parallel worker models gather data, and a heavy-weight judge delivers the final truth.
* **Artemis (Academic Mode):** *[Coming Soon]* Deep, cyclic research. An agent that scours the web, loops through citations, and rigorously peer-reviews its own findings before outputting a thesis.
* **Leviathan (Devil's Advocate):** *[Coming Soon]* System architecture stress-testing. Feed it a design proposal, and aggressive Red Team agents will hunt for bottlenecks and security flaws.

## 🚀 Quick Start

Phase 1 MVP now includes:
- Odin mode LangGraph pipeline with Muninn + Huginn workers and Odin judge synthesis
- Async LiteLLM routing with Groq -> Gemini -> Ollama fallback (`open_council.core.llm`)
- Interactive CLI REPL (`council --mode odin`) with `/exit`/`/quit`, resilient Ctrl+C handling, and Rich output
- First-run setup wizard with global config at `~/.open-council/.env` (plus temporary local `.env` fallback)
- Ollama readiness checks (installed, server reachable, model available) with actionable startup guidance

**Mac/Linux one-command install (MVP):**
```bash
curl -fsSL https://aayushbhaskar.github.io/OpenCouncil/install.sh | bash
```

This installs Open Council under `~/.open-council-app` and links `council` to `~/.local/bin`.

⚡ Optional Power-Up: The Local Fallback Engine
For privacy and resilience, install Ollama from [Ollama Downloads](https://ollama.com/download). Open Council now checks Ollama readiness in detail (binary present, local server reachable, model pulled) and prints guidance before your first prompt.

For dev enthusiasts: Open Council requires **Python 3.11+**. 

```bash
# 1. Clone the repository
git clone https://github.com/aayushbhaskar/OpenCouncil.git
cd OpenCouncil

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Setup your environment variables
# Preferred (new): Open Council stores keys in ~/.open-council/.env
# Transitional fallback: if ~/.open-council/.env is missing but local .env exists,
# Open Council will still use local .env for compatibility.

# 5. Run tests for fallback behavior
pytest -q tests/test_llm_client.py

# 6. Run Odin mode
council --mode odin

# In chat:
# - Exit commands: /exit or /quit
# - If you press Ctrl+C once, Open Council shows guidance.
# - Press Ctrl+C again to exit immediately.
```

If `~/.open-council/.env` is missing, Open Council starts a first-run wizard to
collect keys and run Ollama readiness checks before entering chat.

### Ollama readiness checklist
```bash
# 1) Start the local Ollama server
ollama serve

# 2) Pull the configured fallback model (default)
ollama pull llama3.1
```

If your configured model differs, pull that exact model name from `OLLAMA_MODEL`
(for example, `OLLAMA_MODEL=ollama/llama3.1` maps to `ollama pull llama3.1`).

## 🏗️ Architecture & Resilience
Open Council is designed for "Graceful Degradation." It expects APIs to fail and handles them silently:

Network Throttling: Strict asyncio.Semaphore implementation prevents 429 Rate Limits from aggressive parallel generation.

Tiered Fallback Cascade: Uses LiteLLM to automatically route failed queries: Groq -> Gemini -> Local Ollama. If the cloud goes down, your local machine takes over without crashing the graph.

Local Checkpointing: Planned for the next phase via SQLite-backed checkpoints (not yet wired in the MVP iteration).

## 🗺️ Roadmap
[ ] Phase 1: The Resilient MVP (Odin Mode, LiteLLM Routing, Rich CLI)

[ ] Phase 2: Deep Reasoning (Artemis Mode, SQLite Memory, Web Scraping)

[ ] Phase 3: Enterprise Scale (Leviathan Mode, Local ChromaDB RAG, AWS Bedrock)

[ ] Phase 4: The Workstation (Ariadne Mode, Secure Local File Access)