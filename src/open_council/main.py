"""CLI entrypoint for Open Council."""

import argparse
import asyncio
import contextlib
import shutil
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from open_council.core.fake import fake_mode_enabled
from open_council.core.llm import configure_litellm_logging
from open_council.cli.constants import (
    ALL_MODES,
    GLOBAL_ENV_PATH,
    LOCAL_ENV_PATH,
    TEMPLATE_ENV_PATH,
    UPDATE_COMMAND,
    WIRED_MODES,
)
from open_council.cli.graph_ui import (
    invoke_artemis_graph_with_ui as _invoke_artemis_graph_with_ui_impl,
    invoke_odin_graph_with_ui as _invoke_odin_graph_with_ui_impl,
)
from open_council.cli.prompting import (
    prompt_with_exit_controls as _prompt_with_exit_controls_impl,
    without_echoctl as _without_echoctl_impl,
)
from open_council.cli.repl import run_odin_repl as _run_odin_repl_impl
from open_council.config.bootstrap import (
    ensure_env_file_with_wizard as _ensure_env_file_with_wizard_impl,
    load_env_file as _load_env_file_impl,
)
from open_council.graphs.artemis_graph import build_artemis_graph
from open_council.graphs.executive_graph import build_odin_graph
from open_council.runtime.checkpoints import (
    ensure_thread_id,
    get_thread_mode,
    list_threads,
    open_async_sqlite_saver,
    restore_chat_history,
)
from open_council.runtime.provider_readiness import (
    OllamaReadiness,
    get_ollama_readiness as _get_ollama_readiness_impl,
    http_get_json as _http_get_json_impl,
    print_ollama_status as _print_ollama_status_impl,
    print_provider_readiness_summary as _print_provider_readiness_summary_impl,
)
from open_council.runtime.update_check import (
    is_falsey_env as _is_falsey_env_impl,
    is_truthy_env as _is_truthy_env_impl,
    run_git_command as _run_git_command_impl,
    run_update_command as _run_update_command_impl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="council", description="Open Council CLI")
    parser.add_argument(
        "--mode",
        choices=ALL_MODES,
        default=None,
        help="Council mode to run (default: odin, or the mode of the --thread being resumed).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose runtime diagnostics.")
    parser.add_argument(
        "--show-drafts",
        action="store_true",
        help="Show worker drafts before the final response when supported by the active mode.",
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="Resume a thread from `council history` (opens in that thread's mode), or create one with this id.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("history", help="List recent checkpoint threads.")
    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.mode_explicit = args.mode is not None
    args.mode = args.mode or "odin"
    return args


def app(argv: Sequence[str] | None = None) -> None:
    args = parse_cli_args(argv)
    console = Console()

    if getattr(args, "command", None) == "history":
        print_thread_history(console=console)
        return

    resumed_mode = get_thread_mode(args.thread.strip()) if (args.thread or "").strip() else None
    if resumed_mode is not None and resumed_mode != args.mode:
        if args.mode_explicit:
            console.print(
                f"[yellow]Thread {args.thread.strip()} belongs to {resumed_mode} mode; "
                f"opening it in {resumed_mode} instead of {args.mode}.[/yellow]"
            )
        args.mode = resumed_mode

    configure_litellm_logging(debug=args.debug)
    console.print("Open Council starting...")
    console.print(f"Mode: {args.mode}")
    console.print(f"Debug: {'enabled' if args.debug else 'disabled'}")
    console.print(f"Show drafts: {'enabled' if args.show_drafts else 'disabled'}")
    if args.mode not in WIRED_MODES:
        console.print(f"{args.mode} mode is planned and not yet wired.")
        return

    env_path = resolve_env_path(console=console)
    if not ensure_env_file_with_wizard(console=console, env_path=env_path, template_path=TEMPLATE_ENV_PATH):
        return
    _load_env_file(env_path)
    print_provider_readiness_summary(console=console)
    if fake_mode_enabled():
        console.print("[yellow]Fake mode enabled: canned LLM/search responses, no API calls.[/yellow]")
    maybe_print_update_notice(console=console)

    thread_id = ensure_thread_id(args.thread)
    graph_config = {"configurable": {"thread_id": thread_id}}
    initial_history: list[dict[str, str]] = []
    if resumed_mode is not None:
        initial_history = asyncio.run(_load_resumed_history(mode=args.mode, config=graph_config))
        turns = sum(1 for message in initial_history if message.get("role") == "user")
        console.print(f"[dim]Resumed thread {thread_id} ({args.mode}, {turns} previous turn(s)).[/dim]")
        last_question = next(
            (m["content"] for m in reversed(initial_history) if m.get("role") == "user"),
            "",
        )
        if last_question:
            console.print(f"[dim]Last question: {_truncate(last_question, 100)}[/dim]")
    else:
        console.print(f"[dim]Checkpoint thread: {thread_id}[/dim]")

    run_odin_repl(
        console=console,
        debug=args.debug,
        initial_show_drafts=args.show_drafts,
        thread_id=thread_id,
        graph_config=graph_config,
        initial_mode=args.mode,
        initial_history=initial_history,
    )


async def _load_resumed_history(*, mode: str, config: dict) -> list[dict[str, str]]:
    builder = build_artemis_graph if mode == "artemis" else build_odin_graph
    try:
        async with open_async_sqlite_saver() as checkpointer:
            if checkpointer is None:
                return []
            snapshot = await builder(checkpointer=checkpointer).aget_state(config)
    except Exception:  # noqa: BLE001
        return []
    return restore_chat_history(getattr(snapshot, "values", None) or {})


def _truncate(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def run_odin_repl(
    console: Console,
    *,
    debug: bool = False,
    initial_show_drafts: bool = False,
    thread_id: str | None = None,
    graph_config: dict | None = None,
    initial_mode: str = "odin",
    initial_history: list[dict[str, str]] | None = None,
) -> None:
    _run_odin_repl_impl(
        console=console,
        debug=debug,
        initial_show_drafts=initial_show_drafts,
        graph_builder=build_odin_graph,
        artemis_graph_builder=build_artemis_graph,
        prompt_with_exit_controls_fn=_prompt_with_exit_controls,
        invoke_odin_graph_with_ui_fn=_invoke_odin_graph_with_ui,
        invoke_artemis_graph_with_ui_fn=_invoke_artemis_graph_with_ui,
        resolve_env_path_fn=resolve_env_path,
        thread_id=thread_id,
        graph_config=graph_config,
        initial_mode=initial_mode,
        initial_history=initial_history,
    )


async def _invoke_odin_graph_with_ui(*, graph, state, console: Console, config=None):
    async with open_async_sqlite_saver() as checkpointer:
        if checkpointer is not None and config:
            graph = build_odin_graph(checkpointer=checkpointer)
        return await _invoke_odin_graph_with_ui_impl(graph=graph, state=state, console=console, config=config)


async def _invoke_artemis_graph_with_ui(*, graph, state, console: Console, config=None):
    async with open_async_sqlite_saver() as checkpointer:
        if checkpointer is not None and config:
            graph = build_artemis_graph(checkpointer=checkpointer)
        return await _invoke_artemis_graph_with_ui_impl(graph=graph, state=state, console=console, config=config)


def print_thread_history(*, console: Console) -> None:
    threads = list_threads(limit=50)
    if not threads:
        console.print("No checkpoint threads found yet.")
        return
    table = Table(title="Open Council thread history")
    table.add_column("Thread ID", no_wrap=True, min_width=36)
    table.add_column("Mode")
    table.add_column("First question")
    table.add_column("Updated")
    for item in threads:
        updated = datetime.fromtimestamp(item.updated_at).strftime("%Y-%m-%d %H:%M:%S")
        table.add_row(item.thread_id, item.mode, _truncate(item.title, 60) or "-", updated)
    console.print(table)
    console.print("[dim]Resume with: council --thread <id>[/dim]")


def resolve_env_path(*, console: Console) -> Path:
    if GLOBAL_ENV_PATH.exists():
        return GLOBAL_ENV_PATH
    if LOCAL_ENV_PATH.exists():
        console.print(
            "[dim]Using local .env for now. "
            "Run setup again to migrate to ~/.open-council/.env.[/dim]"
        )
        return LOCAL_ENV_PATH
    return GLOBAL_ENV_PATH


def _load_env_file(env_path: Path) -> None:
    _load_env_file_impl(env_path)


def ensure_env_file_with_wizard(*, console: Console, env_path: Path = Path(".env"), template_path: Path = Path(".env.example")) -> bool:
    return _ensure_env_file_with_wizard_impl(
        console=console,
        env_path=env_path,
        template_path=template_path,
        prompt_with_exit_controls_fn=_prompt_with_exit_controls,
        get_ollama_readiness_fn=get_ollama_readiness,
        print_ollama_status_fn=_print_ollama_status,
    )


def get_ollama_readiness() -> OllamaReadiness:
    return _get_ollama_readiness_impl(which_fn=shutil.which, http_get_json_fn=_http_get_json)


def _http_get_json(url: str, *, timeout_seconds: float):
    return _http_get_json_impl(url, timeout_seconds=timeout_seconds)


def _print_ollama_status(*, console: Console, status: OllamaReadiness) -> None:
    _print_ollama_status_impl(console=console, status=status)


def print_provider_readiness_summary(*, console: Console) -> None:
    _print_provider_readiness_summary_impl(console=console, get_ollama_readiness_fn=get_ollama_readiness)


def maybe_print_update_notice(*, console: Console) -> None:
    if _is_falsey_env("OPEN_COUNCIL_UPDATE_CHECK"):
        return
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        return
    current_sha = _run_git_command(repo_root, "rev-parse", "HEAD")
    remote_line = _run_git_command(repo_root, "ls-remote", "origin", "refs/heads/main")
    if not current_sha or not remote_line:
        return
    if current_sha.strip() == remote_line.split(maxsplit=1)[0].strip():
        return
    if _is_truthy_env("OPEN_COUNCIL_AUTO_UPDATE"):
        console.print("[yellow]Update available.[/yellow] Auto-update is enabled; applying update now...")
        if _run_update_command(repo_root):
            console.print("[green]Auto-update complete.[/green] Restart council to use the latest version.")
            return
        console.print("[yellow]Auto-update failed.[/yellow] To update now, exit application and run:\n"
                      f"[dim]{UPDATE_COMMAND}[/dim]\nThen restart council to use the latest version.")
        return
    console.print("[yellow]Update available.[/yellow] To update now, exit application and run:\n"
                  f"[dim]{UPDATE_COMMAND}[/dim]\nThen restart council to use the latest version.")


def _run_git_command(repo_root: Path, *args: str) -> str | None:
    return _run_git_command_impl(repo_root, *args)


def _run_update_command(repo_root: Path) -> bool:
    return _run_update_command_impl(repo_root)


def _is_truthy_env(name: str) -> bool:
    return _is_truthy_env_impl(name)


def _is_falsey_env(name: str) -> bool:
    return _is_falsey_env_impl(name)


def _prompt_with_exit_controls(*, prompt: str, console: Console, interrupt_state: dict[str, bool], default: str | None) -> str | None:
    return _prompt_with_exit_controls_impl(
        prompt=prompt,
        console=console,
        interrupt_state=interrupt_state,
        default=default,
        ask_fn=Prompt.ask,
        without_echoctl_fn=_without_echoctl,
    )


@contextlib.contextmanager
def _without_echoctl():
    with _without_echoctl_impl():
        yield
