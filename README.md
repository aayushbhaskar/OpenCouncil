# Open Council 🏛️

> **A private, multi-agent terminal workstation for deep research and system stress-testing.**

Open Council is a CLI-first, open-source AI orchestrator. It replicates enterprise-grade "Council of Models" workflows by utilizing a deterministic LangGraph state machine. It seamlessly mixes zero-cost local models (Ollama) with cloud endpoints (Groq, Gemini, Bedrock) to deliver deep reasoning, code analysis, and system architecture stress-testing right in your terminal.

## 🌟 The Pantheon (Modes)

Open Council operates through specialized agentic graphs, invoked via the CLI:

* **Odin (Executive Mode):** *[Coming Soon]* Fast, overarching synthesis. Parallel worker models gather data, and a heavy-weight judge delivers the final truth.
* **Artemis (Academic Mode):** *[Coming Soon]* Deep, cyclic research. An agent that scours the web, loops through citations, and rigorously peer-reviews its own findings before outputting a thesis.
* **Leviathan (Devil's Advocate):** *[Coming Soon]* System architecture stress-testing. Feed it a design proposal, and aggressive Red Team agents will hunt for bottlenecks and security flaws.

## 🚀 Quick Start

First Build Releasing Soon. Stay Tuned!

⚡ Optional Power-Up: The Local Fallback Engine
For the ultimate privacy and resilience experience, we highly recommend installing Ollama before your first run. If Open Council detects Ollama running on localhost, it will automatically use it as a zero-cost safety net if your cloud APIs hit rate limits. Once Ollama is installed, just run ollama pull llama3 in your terminal.

Open Council requires **Python 3.11+**. 

```bash
# 1. Clone the repository
git clone https://github.com/aayushbhaskar/open-council.git
cd open-council

# 2. Install dependencies
pip install -e .

# 3. Setup your environment variables
cp .env.example .env
# Open .env in your editor and add your free API keys

# 4. Invoke the Council
council run "Design a distributed caching architecture for a high-traffic e-commerce site" --mode leviathan

Note: On your first run, Open Council will launch an interactive terminal wizard to set up your .env API keys and auto-detect your local Ollama installation.
```

## 🏗️ Architecture & Resilience
Open Council is designed for "Graceful Degradation." It expects APIs to fail and handles them silently:

Network Throttling: Strict asyncio.Semaphore implementation prevents 429 Rate Limits from aggressive parallel generation.

Tiered Fallback Cascade: Uses LiteLLM to automatically route failed queries: Groq -> Gemini -> Local Ollama. If the cloud goes down, your local machine takes over without crashing the graph.

Local Checkpointing: LangGraph thread states are written to local SQLite, allowing you to pause a debate and resume it days later.

## 🗺️ Roadmap
[ ] Phase 1: The Resilient MVP (Odin Mode, LiteLLM Routing, Rich CLI)

[ ] Phase 2: Deep Reasoning (Artemis Mode, SQLite Memory, Web Scraping)

[ ] Phase 3: Enterprise Scale (Leviathan Mode, Local ChromaDB RAG, AWS Bedrock)

[ ] Phase 4: The Workstation (Ariadne Mode, Secure Local File Access)