"""Odin REPL loop and slash-command handlers."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from open_council.cli.constants import ALL_MODES, CONFIGURABLE_FLAGS, TEMPLATE_ENV_PATH, WIRED_MODES
from open_council.config.env_files import normalize_flag_value, read_env_template, set_env_value
from open_council.state.executive import ChatMessage, OdinState, initialize_odin_state


def run_odin_repl(
    console: Console,
    *,
    debug: bool = False,
    initial_show_drafts: bool = False,
    graph_builder,
    prompt_with_exit_controls_fn,
    invoke_odin_graph_with_ui_fn,
    resolve_env_path_fn,
) -> None:
    """
    Run the interactive Odin chat REPL loop.

    Args:
        console: Rich console used for all terminal output.
        debug: If True, re-raise graph execution exceptions after printing.
        initial_show_drafts: Initial worker draft visibility toggle.
        graph_builder: Graph constructor callback.
        prompt_with_exit_controls_fn: Input prompt callback.
        invoke_odin_graph_with_ui_fn: Graph invoke callback with UI streaming.
        resolve_env_path_fn: Env path resolver callback used by `/config`.

    Behavior:
        - Reads user turns via Rich prompt.
        - Enforces slash-based exit commands.
        - Executes one graph turn per user query.
        - Persists and reuses state across turns.
    """
    graph = graph_builder()
    state: OdinState | None = None
    current_mode = "odin"
    show_drafts = initial_show_drafts
    interrupt_state = {"armed": False}
    console.print(
        "Odin ready. Type your query, '/mode', '/config', '/show-drafts', "
        "or '/exit' / '/quit' to stop."
    )

    while True:
        user_input = prompt_with_exit_controls_fn(
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
        if lowered.startswith("/mode"):
            current_mode = handle_mode_command(
                command=user_input,
                current_mode=current_mode,
                console=console,
            )
            continue
        if lowered.startswith("/config"):
            handle_config_command(
                command=user_input,
                console=console,
                resolve_env_path_fn=resolve_env_path_fn,
            )
            continue
        if lowered.startswith("/show-drafts"):
            show_drafts = handle_show_drafts_command(
                command=user_input,
                show_drafts=show_drafts,
                console=console,
            )
            continue
        if current_mode != "odin":
            console.print(
                f"\n{current_mode} mode is selected but not wired yet. "
                "Use /mode odin to continue in Phase 1."
            )
            continue

        state = prepare_state_for_turn(
            previous_state=state,
            user_input=user_input,
            show_drafts=show_drafts,
        )
        try:
            result = asyncio.run(invoke_odin_graph_with_ui_fn(graph=graph, state=state, console=console))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Graph execution failed:[/red] {exc}")
            if debug:
                raise
            continue

        state = append_assistant_turn(
            state=result,
            final_synthesis=result.get("final_synthesis", "No synthesis produced."),
        )
        if show_drafts:
            print_worker_drafts(console=console, state=state)
        else:
            console.print()
        final_text = _normalize_for_render(state["chat_history"][-1]["content"])
        console.print(Markdown(final_text))


def handle_show_drafts_command(*, command: str, show_drafts: bool, console: Console) -> bool:
    """
    Process `/show-drafts` command and return updated draft-visibility state.

    Supported commands:
        - `/show-drafts`
        - `/show-drafts on`
        - `/show-drafts off`
    """
    parts = command.strip().split(maxsplit=1)
    if len(parts) == 1:
        status = "on" if show_drafts else "off"
        console.print(f"\nDraft visibility is currently [bold]{status}[/bold].")
        console.print("Use [bold]/show-drafts on[/bold] or [bold]/show-drafts off[/bold].")
        return show_drafts

    value = parts[1].strip().lower()
    if value not in {"on", "off"}:
        console.print("\nInvalid value. Use [bold]/show-drafts on[/bold] or [bold]/show-drafts off[/bold].")
        return show_drafts

    next_value = value == "on"
    if next_value == show_drafts:
        status = "on" if show_drafts else "off"
        console.print(f"\nDraft visibility is already [bold]{status}[/bold].")
        return show_drafts

    status = "on" if next_value else "off"
    console.print(f"\nDraft visibility set to [bold]{status}[/bold].")
    return next_value


def handle_mode_command(*, command: str, current_mode: str, console: Console) -> str:
    """
    Process `/mode` chat commands and return the selected mode.

    Args:
        command: Raw user command text.
        current_mode: Active mode before processing the command.
        console: Rich console for command feedback.

    Returns:
        Updated active mode. If selection is invalid or unwired, the current mode
        is preserved.
    """
    parts = command.strip().split(maxsplit=1)
    if len(parts) == 1:
        modes = ", ".join(
            f"{mode}{' (available)' if mode in WIRED_MODES else ' (planned)'}"
            for mode in ALL_MODES
        )
        console.print(f"\nCurrent mode: [bold]{current_mode}[/bold]")
        console.print(f"Available modes: {modes}")
        console.print("Use [bold]/mode <name>[/bold] to switch modes.")
        return current_mode

    requested_mode = parts[1].strip().lower()
    if requested_mode not in ALL_MODES:
        console.print(f"\nUnknown mode: {requested_mode}")
        console.print(f"Use [bold]/mode <name>[/bold] with one of: {', '.join(ALL_MODES)}")
        return current_mode
    if requested_mode not in WIRED_MODES:
        console.print(
            f"\n{requested_mode} mode is planned and not yet wired in Phase 1. "
            f"Remaining on {current_mode}."
        )
        return current_mode
    if requested_mode == current_mode:
        console.print(f"\nAlready in {current_mode} mode.")
        return current_mode

    console.print(f"\nSwitched mode to [bold]{requested_mode}[/bold].")
    return requested_mode


def handle_config_command(*, command: str, console: Console, resolve_env_path_fn) -> None:
    """
    Process `/config` commands for runtime flag configuration.

    Supported commands:
        - `/config`
        - `/config set <KEY> <VALUE>`
    """
    parts = command.strip().split(maxsplit=3)
    env_path = resolve_env_path_fn(console=console)

    if len(parts) == 1:
        console.print(f"\nConfig file: [bold]{env_path}[/bold]")
        for key in CONFIGURABLE_FLAGS:
            raw = os.getenv(key, "").strip()
            current = raw if raw else "(unset)"
            console.print(f"- {key} = {current}")
        console.print("Set values with: [bold]/config set <KEY> <VALUE>[/bold]")
        return

    if len(parts) < 4 or parts[1].lower() != "set":
        console.print(
            "\nInvalid config command. Use [bold]/config[/bold] or "
            "[bold]/config set <KEY> <VALUE>[/bold]."
        )
        return

    key = parts[2].strip().upper()
    value = parts[3].strip()
    if key not in CONFIGURABLE_FLAGS:
        console.print(
            f"\nUnsupported key: {key}. "
            f"Supported keys: {', '.join(CONFIGURABLE_FLAGS)}"
        )
        return

    normalized = normalize_flag_value(value)
    if normalized is None:
        console.print(
            "\nUnsupported value. Use one of: 1, 0, true, false, yes, no, on, off."
        )
        return

    template = env_path.read_text(encoding="utf-8") if env_path.exists() else read_env_template(TEMPLATE_ENV_PATH)
    rendered = set_env_value(template, key, normalized)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(rendered, encoding="utf-8")
    os.environ[key] = normalized
    console.print(f"\nUpdated [bold]{key}[/bold]={normalized} in {env_path}")


def print_worker_drafts(*, console: Console, state: OdinState) -> None:
    """Render Muninn/Huginn worker drafts before Odin's final verdict."""
    drafts = state.get("parallel_drafts", [])
    console.print()
    if not drafts:
        console.print("[dim]No worker drafts available for this turn.[/dim]")
        return
    console.print("[bold]Council drafts[/bold]")
    for index, draft in enumerate(drafts):
        if index > 0:
            console.print()
        worker_id = str(draft.get("worker_id", "worker")).title()
        model = str(draft.get("model", "unknown"))
        content = _normalize_for_render(str(draft.get("draft", ""))) or "(empty draft)"
        console.print(f"[bold]{worker_id}[/bold] ({model})")
        console.print(Markdown(content))
    console.print()


def _normalize_for_render(content: str) -> str:
    """Trim surrounding blank space to keep REPL spacing consistent."""
    return content.strip()


def prepare_state_for_turn(*, previous_state: OdinState | None, user_input: str, show_drafts: bool) -> OdinState:
    """
    Build the input state for the next Odin turn.

    Args:
        previous_state: State from prior turn, if any.
        user_input: Current user message text.
        show_drafts: Runtime toggle for draft visibility.

    Returns:
        Updated state carrying prior context, resetting per-turn draft buffer,
        and appending the new user message to `chat_history`.
    """
    if previous_state is None:
        state = initialize_odin_state(user_input)
    else:
        state = {
            **previous_state,
            "query": user_input,
            "parallel_drafts": [],
        }
    state["show_drafts"] = show_drafts
    return append_chat_message(state=state, role="user", content=user_input)


def append_assistant_turn(*, state: OdinState, final_synthesis: str) -> OdinState:
    """
    Append final assistant synthesis into persisted chat history.

    Args:
        state: Current Odin state after graph execution.
        final_synthesis: Final judge output text.

    Returns:
        New state object with appended assistant message.
    """
    return append_chat_message(state=state, role="assistant", content=final_synthesis)


def append_chat_message(*, state: OdinState, role: str, content: str) -> OdinState:
    """
    Append one chat message to state history immutably.

    Args:
        state: Base state.
        role: Message role label (e.g., user, assistant).
        content: Message content.

    Returns:
        New state with extended `chat_history`.
    """
    history = list(state.get("chat_history", []))
    history.append(ChatMessage(role=role, content=content))
    return {
        **state,
        "chat_history": history,
    }
