"""Council REPL loop and slash-command handlers."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from open_council.cli.constants import ALL_MODES, CONFIGURABLE_FLAGS, TEMPLATE_ENV_PATH, WIRED_MODES
from open_council.config.env_files import normalize_flag_value, read_env_template, set_env_value
from open_council.runtime.checkpoints import new_thread_id, touch_thread
from open_council.state.artemis import ArtemisState, initialize_artemis_state
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
    thread_id: str | None = None,
    graph_config: dict[str, Any] | None = None,
    invoke_artemis_graph_with_ui_fn=None,
    artemis_graph_builder=None,
    initial_mode: str = "odin",
    initial_history: list[ChatMessage] | None = None,
) -> None:
    """
    Run interactive chat REPL for wired modes (Odin / Artemis).

    Kept name `run_odin_repl` for main.py compatibility; supports Artemis when
    builders/invokers are provided. Each thread belongs to one mode: `/mode` starts a
    new thread, and `initial_history` seeds a resumed thread's conversation.
    """
    current_mode = initial_mode if initial_mode in WIRED_MODES else "odin"
    odin_graph = graph_builder()
    artemis_graph = artemis_graph_builder() if artemis_graph_builder is not None else None
    odin_state: OdinState | None = None
    artemis_state: ArtemisState | None = None
    if initial_history:
        # Placeholder query; prepare_*_state_for_turn replaces it on the next turn.
        if current_mode == "artemis":
            artemis_state = {**initialize_artemis_state("(resumed)"), "chat_history": list(initial_history)}
        else:
            odin_state = {**initialize_odin_state("(resumed)"), "chat_history": list(initial_history)}
    show_drafts = initial_show_drafts
    interrupt_state = {"armed": False}
    active_thread = thread_id
    config = graph_config

    mode_label = current_mode.title()
    console.print(
        f"{mode_label} ready. Type your query, '/mode', '/config', '/show-drafts', "
        "or '/exit' / '/quit' to stop."
    )
    if active_thread:
        console.print(f"[dim]Thread: {active_thread}[/dim]")
    console.print()

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
            next_mode = handle_mode_command(
                command=user_input,
                current_mode=current_mode,
                console=console,
            )
            if next_mode != current_mode:
                current_mode = next_mode
                odin_state = None
                artemis_state = None
                if active_thread:
                    active_thread = new_thread_id()
                    config = {"configurable": {"thread_id": active_thread}}
                    console.print(
                        f"[dim]Started a new {current_mode} thread: {active_thread}[/dim]"
                    )
                else:
                    console.print(f"[dim]Switched active session to {current_mode}.[/dim]")
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
        if current_mode not in WIRED_MODES:
            console.print(
                f"\n{current_mode} mode is selected but not wired yet. "
                "Use /mode odin or /mode artemis."
            )
            continue

        try:
            if current_mode == "artemis":
                if artemis_graph is None or invoke_artemis_graph_with_ui_fn is None:
                    console.print("\nArtemis graph is not available in this build.")
                    continue
                artemis_state = prepare_artemis_state_for_turn(
                    previous_state=artemis_state,
                    user_input=user_input,
                    show_drafts=show_drafts,
                )
                result = asyncio.run(
                    invoke_artemis_graph_with_ui_fn(
                        graph=artemis_graph,
                        state=artemis_state,
                        console=console,
                        config=config,
                    )
                )
                artemis_state = append_assistant_turn_artemis(
                    state=result,
                    final_synthesis=result.get("final_synthesis", "No synthesis produced."),
                )
                if active_thread:
                    touch_thread(thread_id=active_thread, mode="artemis", title=user_input)
                if show_drafts:
                    print_artemis_drafts(console=console, state=artemis_state)
                else:
                    console.print()
                final_text = _normalize_for_render(artemis_state["chat_history"][-1]["content"])
                console.print("[bold]Artemis Research Verdict[/bold]")
                console.print(Markdown(final_text))
                console.print()
            else:
                odin_state = prepare_state_for_turn(
                    previous_state=odin_state,
                    user_input=user_input,
                    show_drafts=show_drafts,
                )
                result = asyncio.run(
                    invoke_odin_graph_with_ui_fn(
                        graph=odin_graph,
                        state=odin_state,
                        console=console,
                        config=config,
                    )
                )
                odin_state = append_assistant_turn(
                    state=result,
                    final_synthesis=result.get("final_synthesis", "No synthesis produced."),
                )
                if active_thread:
                    touch_thread(thread_id=active_thread, mode="odin", title=user_input)
                if show_drafts:
                    print_worker_drafts(console=console, state=odin_state)
                else:
                    console.print()
                final_text = _normalize_for_render(odin_state["chat_history"][-1]["content"])
                console.print("[bold]Odin's Final Verdict[/bold]")
                console.print(Markdown(final_text))
                console.print()
        except KeyboardInterrupt:
            interrupt_state["armed"] = True
            console.print("\n[dim]Current turn cancelled. Press Ctrl+C again to exit, or type /exit.[/dim]")
            continue
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Graph execution failed:[/red] {exc}")
            if debug:
                raise
            continue


def handle_show_drafts_command(*, command: str, show_drafts: bool, console: Console) -> bool:
    """Process `/show-drafts` command and return updated draft-visibility state."""
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
    """Process `/mode` chat commands and return the selected mode."""
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
            f"\n{requested_mode} mode is planned and not yet wired. "
            f"Remaining on {current_mode}."
        )
        return current_mode
    if requested_mode == current_mode:
        console.print(f"\nAlready in {current_mode} mode.")
        return current_mode

    console.print(f"\nSwitched mode to [bold]{requested_mode}[/bold].")
    return requested_mode


def handle_config_command(*, command: str, console: Console, resolve_env_path_fn) -> None:
    """Process `/config` commands for runtime flag configuration."""
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
    worker_order = {"muninn": 0, "huginn": 1}
    ordered_drafts = sorted(
        drafts,
        key=lambda item: worker_order.get(str(item.get("worker_id", "")).lower(), 99),
    )
    console.print("[bold]Council drafts[/bold]")
    role_labels = {
        "muninn": "Thesis/Constructor",
        "huginn": "Antithesis/Deconstructor",
    }
    for index, draft in enumerate(ordered_drafts):
        if index > 0:
            console.print()
            console.print("[dim]---[/dim]")
            console.print()
        worker_key = str(draft.get("worker_id", "worker")).lower()
        worker_id = worker_key.title()
        role_label = role_labels.get(worker_key, "Worker")
        model = str(draft.get("model", "unknown"))
        content = _normalize_for_render(str(draft.get("draft", ""))) or "(empty draft)"
        console.print(f"[bold]{worker_id}[/bold] ({role_label}) - {model}")
        console.print(Markdown(content))
    console.print()
    console.print("[bold]End of Council drafts[/bold]")
    console.print()


def print_artemis_drafts(*, console: Console, state: ArtemisState) -> None:
    """Render Artemis specialist draft and critic notes before final verdict."""
    console.print()
    console.print("[bold]Artemis research trail[/bold]")
    plan = _normalize_for_render(str(state.get("plan", "")))
    draft = _normalize_for_render(str(state.get("specialist_draft", "")))
    feedback = _normalize_for_render(str(state.get("critic_feedback", "")))
    if plan:
        console.print("[bold]Plan[/bold]")
        console.print(Markdown(plan))
        console.print()
    if draft:
        console.print("[bold]Specialist draft[/bold]")
        console.print(Markdown(draft))
        console.print()
    if feedback:
        console.print(
            f"[bold]Critic[/bold] ({state.get('critic_verdict', 'n/a')}, "
            f"revisions={state.get('revision_count', 0)})"
        )
        console.print(Markdown(feedback))
        console.print()
    console.print("[bold]End of Artemis research trail[/bold]")
    console.print()


def _normalize_for_render(content: str) -> str:
    """Trim surrounding blank space to keep REPL spacing consistent."""
    return content.strip()


def prepare_state_for_turn(*, previous_state: OdinState | None, user_input: str, show_drafts: bool) -> OdinState:
    """Build the input state for the next Odin turn."""
    if previous_state is None:
        state = initialize_odin_state(user_input)
    else:
        state = {
            **previous_state,
            "query": user_input,
            "parallel_drafts": [],
            "final_synthesis": "",
        }
    state["show_drafts"] = show_drafts
    return append_chat_message(state=state, role="user", content=user_input)


def prepare_artemis_state_for_turn(
    *,
    previous_state: ArtemisState | None,
    user_input: str,
    show_drafts: bool,
) -> ArtemisState:
    """Build the input state for the next Artemis turn."""
    if previous_state is None:
        state = initialize_artemis_state(user_input)
    else:
        state = {
            **previous_state,
            "query": user_input,
            "plan": "",
            "search_queries": [],
            "search_hits": [],
            "extracted_pages": [],
            "specialist_draft": "",
            "critic_verdict": "",
            "critic_feedback": "",
            "critic_accepted": False,
            "revision_count": 0,
            "final_synthesis": "",
        }
    state["show_drafts"] = show_drafts
    return append_chat_message_artemis(state=state, role="user", content=user_input)


def append_assistant_turn(*, state: OdinState, final_synthesis: str) -> OdinState:
    """Append final assistant synthesis into persisted chat history."""
    return append_chat_message(state=state, role="assistant", content=final_synthesis)


def append_assistant_turn_artemis(*, state: ArtemisState, final_synthesis: str) -> ArtemisState:
    return append_chat_message_artemis(state=state, role="assistant", content=final_synthesis)


def append_chat_message(*, state: OdinState, role: str, content: str) -> OdinState:
    """Append one chat message to Odin state history immutably."""
    history = list(state.get("chat_history", []))
    history.append(ChatMessage(role=role, content=content))
    return {
        **state,
        "chat_history": history,
    }


def append_chat_message_artemis(*, state: ArtemisState, role: str, content: str) -> ArtemisState:
    history = list(state.get("chat_history", []))
    history.append(ChatMessage(role=role, content=content))
    return {
        **state,
        "chat_history": history,
    }
