# Open Council 🏛️

> **A private, multi-agent terminal workstation for deep research and system stress-testing.**

Open Council is a CLI-first, open-source AI orchestrator. It replicates enterprise-grade "Council of Models" workflows by utilizing a deterministic LangGraph state machine. It seamlessly mixes zero-cost local models (Ollama) with cloud endpoints (Groq, Gemini, Bedrock) to deliver deep reasoning, code analysis, and system architecture stress-testing right in your terminal.

## 🌟 The Pantheon (Modes)

Open Council operates through specialized agentic graphs, invoked via the CLI:

* **Odin (Executive Mode):** Fast, overarching synthesis. Parallel worker models gather data, and a heavy-weight judge delivers the final truth.
* **Artemis (Academic Mode):** Deep, cyclic research. An agent that scours the web, loops through citations, and rigorously peer-reviews its own findings before outputting a thesis.
* **Leviathan (Devil's Advocate):** *[Coming Soon]* System architecture stress-testing. Feed it a design proposal, and aggressive Red Team agents will hunt for bottlenecks and security flaws.
* **Vulcan (Code Interpreter):** *[Coming Soon]* An execution agent that writes, runs, and debugs scripts within a secure local Docker sandbox.
* **Ariadne (Local Repo Architect):** *[Coming Soon]* Grants the council read-access to your local codebase to map dependencies and troubleshoot structural bugs.

## 🚀 Quick Start

First Build Releasing Soon. Stay Tuned!

Open Council requires **Python 3.11+**. 

```bash
# 1. Clone the repository
git clone [https://github.com/yourusername/open-council.git](https://github.com/yourusername/open-council.git)
cd open-council

# 2. Install dependencies
pip install -e .

# 3. Invoke the Council
council run "Design a distributed caching architecture for a high-traffic e-commerce site" --mode leviathan
