"""CLI entrypoint for Open Council."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from rich.console import Console
from rich.markdown import Markdown
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

    run_odin_repl(console=console, debug=args.debug)


def run_odin_repl(console: Console, *, debug: bool = False) -> None:
    """Interactive Odin REPL with persistent state across turns."""
    graph = build_odin_graph()
    state: OdinState | None = None
    console.print("Odin ready. Type your query, or '/exit' / '/quit' to stop.")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]you[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting Open Council.")
            return

        if not user_input:
            continue
        lowered = user_input.lower()
        if lowered in {"/exit", "/quit"}:
            console.print("Exiting Open Council.")
            return
        if lowered == "exit":
            console.print("To exit, use /exit.")
            continue
        if lowered == "quit":
            console.print("To quit, use /quit.")
            continue

        state = _prepare_state_for_turn(previous_state=state, user_input=user_input)
        try:
            result = asyncio.run(graph.ainvoke(state))
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
