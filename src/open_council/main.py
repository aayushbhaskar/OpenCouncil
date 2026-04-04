"""CLI entrypoint for Open Council."""

from __future__ import annotations

import argparse
import asyncio
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

from open_council.graphs.executive_graph import build_odin_graph
from open_council.state.executive import ChatMessage, OdinState, initialize_odin_state


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 1 CLI parser."""
    parser = argparse.ArgumentParser(prog="council", description="Open Council CLI")
    parser.add_argument(
        "--mode",
        choices=("odin", "artemis", "leviathan"),
        default="odin",
        help="Council mode to run (default: odin).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for verbose runtime diagnostics.",
    )
    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for council entry."""
    return build_parser().parse_args(argv)


def app(argv: Sequence[str] | None = None) -> None:
    """Phase 1 CLI entrypoint with mode/debug argument support."""
    args = parse_cli_args(argv)
    console = Console()
    console.print("Open Council starting...")
    console.print(f"Mode: {args.mode}")
    console.print(f"Debug: {'enabled' if args.debug else 'disabled'}")
    if args.mode != "odin":
        console.print(f"{args.mode} mode is planned and not yet wired in Phase 1.")
        return

    if not ensure_env_file_with_wizard(console=console):
        return
    run_odin_repl(console=console, debug=args.debug)


def run_odin_repl(console: Console, *, debug: bool = False) -> None:
    """Interactive Odin REPL with persistent state across turns."""
    graph = build_odin_graph()
    state: OdinState | None = None
    interrupt_state = {"armed": False}
    console.print("Odin ready. Type your query, or '/exit' / '/quit' to stop.")

    while True:
        user_input = _prompt_with_exit_controls(
            prompt="[bold cyan]you[/bold cyan]",
            console=console,
            interrupt_state=interrupt_state,
            default=None,
        )
        if user_input is None:
            return

        if not user_input:
            continue
        lowered = user_input.lower()
        if lowered in {"/exit", "/quit"}:
            console.print("\nExiting Open Council.")
            return
        if lowered == "exit":
            console.print("\nTo exit, use /exit.")
            continue
        if lowered == "quit":
            console.print("\nTo quit, use /quit.")
            continue

        state = _prepare_state_for_turn(previous_state=state, user_input=user_input)
        try:
            result = asyncio.run(_invoke_odin_graph_with_ui(graph=graph, state=state, console=console))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Graph execution failed:[/red] {exc}")
            if debug:
                raise
            continue

        state = _append_assistant_turn(
            state=result,
            final_synthesis=result.get("final_synthesis", "No synthesis produced."),
        )
        console.print(Markdown(state["chat_history"][-1]["content"]))


def _prepare_state_for_turn(*, previous_state: OdinState | None, user_input: str) -> OdinState:
    if previous_state is None:
        state = initialize_odin_state(user_input)
    else:
        state = {
            **previous_state,
            "query": user_input,
            "parallel_drafts": [],
        }
    return _append_chat_message(state=state, role="user", content=user_input)


def _append_assistant_turn(*, state: OdinState, final_synthesis: str) -> OdinState:
    return _append_chat_message(state=state, role="assistant", content=final_synthesis)


def _append_chat_message(*, state: OdinState, role: str, content: str) -> OdinState:
    history = list(state.get("chat_history", []))
    history.append(ChatMessage(role=role, content=content))
    return {
        **state,
        "chat_history": history,
    }


async def _invoke_odin_graph_with_ui(*, graph: Any, state: OdinState, console: Console) -> OdinState:
    """Run one Odin turn while rendering transient node spinners."""
    if not hasattr(graph, "astream"):
        return await graph.ainvoke(state)

    merged_state: OdinState = dict(state)
    node_labels = {
        "muninn_worker": "Muninn thinking",
        "huginn_worker": "Huginn analyzing risk",
        "odin_judge": "Odin judging",
    }
    completed_nodes: set[str] = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task_ids = {
            node: progress.add_task(description=node_labels[node], total=1, start=False)
            for node in node_labels
        }
        progress.start_task(task_ids["muninn_worker"])
        progress.start_task(task_ids["huginn_worker"])

        async for chunk in graph.astream(state, stream_mode="updates"):
            node_name = _extract_node_name(chunk)
            node_update = _extract_node_update(chunk, node_name)
            merged_state = _merge_state_update(merged_state, node_update)

            if node_name in task_ids and node_name not in completed_nodes:
                completed_nodes.add(node_name)
                progress.update(task_ids[node_name], completed=1)

                if (
                    {"muninn_worker", "huginn_worker"}.issubset(completed_nodes)
                    and "odin_judge" not in completed_nodes
                ):
                    progress.start_task(task_ids["odin_judge"])

    return merged_state


def _extract_node_name(chunk: Any) -> str | None:
    if isinstance(chunk, dict) and chunk:
        return next(iter(chunk))
    return None


def _extract_node_update(chunk: Any, node_name: str | None) -> dict[str, Any]:
    if node_name is None:
        return {}
    node_payload = chunk.get(node_name)
    if isinstance(node_payload, dict):
        return node_payload
    return {}


def _merge_state_update(state: OdinState, update: dict[str, Any]) -> OdinState:
    merged: OdinState = dict(state)
    for key, value in update.items():
        if key == "parallel_drafts" and isinstance(value, list):
            existing = list(merged.get("parallel_drafts", []))
            existing.extend(value)
            merged["parallel_drafts"] = existing
            continue
        merged[key] = value
    return merged


def ensure_env_file_with_wizard(
    *,
    console: Console,
    env_path: Path = Path(".env"),
    template_path: Path = Path(".env.example"),
) -> bool:
    """Create `.env` via first-run wizard when missing."""
    if env_path.exists():
        return True

    console.print("[yellow].env not found. Starting first-run setup.[/yellow]")
    interrupt_state = {"armed": False}

    groq_api_key = _prompt_with_exit_controls(
        prompt="Paste your GROQ_API_KEY (press Enter to skip)",
        console=console,
        interrupt_state=interrupt_state,
        default="",
    )
    if groq_api_key is None:
        return False
    gemini_api_key = _prompt_with_exit_controls(
        prompt="Paste your GEMINI_API_KEY (press Enter to skip)",
        console=console,
        interrupt_state=interrupt_state,
        default="",
    )
    if gemini_api_key is None:
        return False

    ollama_path = shutil.which("ollama")
    if ollama_path:
        console.print(f"[green]Detected Ollama:[/green] {ollama_path}")
    else:
        console.print(
            "[yellow]Ollama was not detected in PATH.[/yellow] "
            "You can still use Groq/Gemini and add Ollama later."
        )

    template = _read_env_template(template_path)
    rendered = _set_env_value(template, "GROQ_API_KEY", groq_api_key) if groq_api_key else template
    rendered = (
        _set_env_value(rendered, "GEMINI_API_KEY", gemini_api_key)
        if gemini_api_key
        else rendered
    )
    env_path.write_text(rendered, encoding="utf-8")
    console.print(f"[green]Created {env_path}[/green]. Update it anytime as needed.")
    return True


def _read_env_template(template_path: Path) -> str:
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return 'GROQ_API_KEY=""\nGEMINI_API_KEY=""\nOLLAMA_BASE_URL="http://localhost:11434"\n'


def _set_env_value(template: str, key: str, value: str) -> str:
    pattern = rf"(?m)^{re.escape(key)}=.*$"
    replacement = f'{key}="{value}"'
    if re.search(pattern, template):
        return re.sub(pattern, replacement, template)
    suffix = "" if template.endswith("\n") else "\n"
    return f"{template}{suffix}{replacement}\n"


def _prompt_with_exit_controls(
    *,
    prompt: str,
    console: Console,
    interrupt_state: dict[str, bool],
    default: str | None,
) -> str | None:
    while True:
        try:
            user_input = Prompt.ask(prompt, default=default).strip()
        except KeyboardInterrupt:
            if interrupt_state.get("armed"):
                console.print("\nExiting Open Council.")
                return None
            interrupt_state["armed"] = True
            console.print("[dim]Press Ctrl+C again to exit, or type /exit.[/dim]")
            continue
        except EOFError:
            console.print("\nExiting Open Council.")
            return None

        interrupt_state["armed"] = False
        if user_input.lower() in {"/exit", "/quit"}:
            console.print("\nExiting Open Council.")
            return None
        return user_input
