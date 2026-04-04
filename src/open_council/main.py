"""CLI entrypoint for Open Council."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import termios
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

from open_council.graphs.executive_graph import build_odin_graph
from open_council.state.executive import ChatMessage, OdinState, initialize_odin_state

GLOBAL_CONFIG_DIR = Path.home() / ".open-council"
GLOBAL_ENV_PATH = GLOBAL_CONFIG_DIR / ".env"
LOCAL_ENV_PATH = Path(".env")
TEMPLATE_ENV_PATH = Path(__file__).resolve().parents[2] / ".env.example"


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level Open Council CLI parser.

    Returns:
        Configured `argparse.ArgumentParser` with mode/debug options.
    """
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
    """
    Parse command-line arguments for `council`.

    Args:
        argv: Optional explicit argument list for tests or embedding.

    Returns:
        Parsed argparse namespace.
    """
    return build_parser().parse_args(argv)


def app(argv: Sequence[str] | None = None) -> None:
    """
    Program entrypoint for Open Council CLI.

    Args:
        argv: Optional argument sequence; defaults to `sys.argv` behavior.

    Behavior:
        - Parses mode/debug flags.
        - Runs first-run wizard when `.env` is missing.
        - Launches Odin REPL for supported mode.
    """
    args = parse_cli_args(argv)
    console = Console()
    console.print("Open Council starting...")
    console.print(f"Mode: {args.mode}")
    console.print(f"Debug: {'enabled' if args.debug else 'disabled'}")
    if args.mode != "odin":
        console.print(f"{args.mode} mode is planned and not yet wired in Phase 1.")
        return

    env_path = resolve_env_path(console=console)
    if not ensure_env_file_with_wizard(
        console=console,
        env_path=env_path,
        template_path=TEMPLATE_ENV_PATH,
    ):
        return
    _load_env_file(env_path)
    print_provider_readiness_summary(console=console)
    run_odin_repl(console=console, debug=args.debug)


def run_odin_repl(console: Console, *, debug: bool = False) -> None:
    """
    Run the interactive Odin chat REPL loop.

    Args:
        console: Rich console used for all terminal output.
        debug: If True, re-raise graph execution exceptions after printing.

    Behavior:
        - Reads user turns via Rich prompt.
        - Enforces slash-based exit commands.
        - Executes one graph turn per user query.
        - Persists and reuses state across turns.
    """
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
    """
    Build the input state for the next Odin turn.

    Args:
        previous_state: State from prior turn, if any.
        user_input: Current user message text.

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
    return _append_chat_message(state=state, role="user", content=user_input)


def _append_assistant_turn(*, state: OdinState, final_synthesis: str) -> OdinState:
    """
    Append final assistant synthesis into persisted chat history.

    Args:
        state: Current Odin state after graph execution.
        final_synthesis: Final judge output text.

    Returns:
        New state object with appended assistant message.
    """
    return _append_chat_message(state=state, role="assistant", content=final_synthesis)


def _append_chat_message(*, state: OdinState, role: str, content: str) -> OdinState:
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


async def _invoke_odin_graph_with_ui(*, graph: Any, state: OdinState, console: Console) -> OdinState:
    """
    Execute one Odin graph turn while rendering transient Rich spinners.

    Args:
        graph: Compiled graph runnable exposing `astream`/`ainvoke`.
        state: Input graph state for this turn.
        console: Rich console for transient progress rendering.

    Returns:
        Merged resulting state after processing streamed node updates.
    """
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
    """
    Extract node name from one streamed update chunk.

    Args:
        chunk: Streamed graph update payload.

    Returns:
        Node key when available, otherwise `None`.
    """
    if isinstance(chunk, dict) and chunk:
        return next(iter(chunk))
    return None


def _extract_node_update(chunk: Any, node_name: str | None) -> dict[str, Any]:
    """
    Extract node-specific state update payload from stream chunk.

    Args:
        chunk: Streamed update payload.
        node_name: Node key previously extracted from the chunk.

    Returns:
        Node update dictionary, or empty dict when unavailable.
    """
    if node_name is None:
        return {}
    node_payload = chunk.get(node_name)
    if isinstance(node_payload, dict):
        return node_payload
    return {}


def _merge_state_update(state: OdinState, update: dict[str, Any]) -> OdinState:
    """
    Merge streamed state delta into the current state snapshot.

    Args:
        state: Current merged state.
        update: State delta emitted by one graph node.

    Returns:
        Updated merged state honoring list-append semantics for
        `parallel_drafts`.
    """
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
    """
    Ensure `.env` exists, guiding first-run setup when missing.

    Args:
        console: Rich console for wizard output.
        env_path: Destination `.env` file path.
        template_path: `.env.example` template path.

    Returns:
        `True` when setup completes or file already exists.
        `False` when the user exits during the wizard.
    """
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

    ollama_status = get_ollama_readiness()
    _print_ollama_status(console=console, status=ollama_status)

    template = _read_env_template(template_path)
    rendered = _set_env_value(template, "GROQ_API_KEY", groq_api_key) if groq_api_key else template
    rendered = (
        _set_env_value(rendered, "GEMINI_API_KEY", gemini_api_key)
        if gemini_api_key
        else rendered
    )
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(rendered, encoding="utf-8")
    console.print(f"[green]Created {env_path}[/green]. Update it anytime as needed.")
    return True


def resolve_env_path(*, console: Console) -> Path:
    """
    Resolve canonical runtime env file path with temporary local fallback.

    Resolution order:
        1. `~/.open-council/.env` if present.
        2. Local `./.env` fallback if present (migration bridge).
        3. Canonical global path for wizard creation.
    """
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
    """Load environment variables from the resolved env file path."""
    load_dotenv(dotenv_path=env_path, override=True)


@dataclass(slots=True)
class OllamaReadiness:
    """Status payload for local Ollama installation/readiness checks."""

    state: str
    message: str
    base_url: str
    model: str


def get_ollama_readiness() -> OllamaReadiness:
    """
    Check whether Ollama is installed, reachable, and model-ready.

    Returns:
        `OllamaReadiness` status in one of:
        - `not_installed`
        - `installed_not_running`
        - `running_model_missing`
        - `ready`
    """
    ollama_path = shutil.which("ollama")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    configured_model = os.getenv("OLLAMA_MODEL", "ollama/llama3.1").strip()
    normalized_model = _normalize_ollama_model_name(configured_model)

    if not ollama_path:
        return OllamaReadiness(
            state="not_installed",
            message="Ollama not detected in PATH.",
            base_url=base_url,
            model=normalized_model,
        )

    tags_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        payload = _http_get_json(tags_url, timeout_seconds=2.0)
    except (TimeoutError, URLError, OSError, ValueError):
        return OllamaReadiness(
            state="installed_not_running",
            message=(
                f"Ollama binary detected at {ollama_path}, but server is not reachable at {base_url}."
            ),
            base_url=base_url,
            model=normalized_model,
        )

    available_models = _extract_ollama_model_names(payload)
    if not _has_ollama_model(available_models, normalized_model):
        return OllamaReadiness(
            state="running_model_missing",
            message=(
                f"Ollama server is reachable at {base_url}, but model '{normalized_model}' is missing."
            ),
            base_url=base_url,
            model=normalized_model,
        )

    return OllamaReadiness(
        state="ready",
        message=f"Ollama is ready at {base_url} with model '{normalized_model}'.",
        base_url=base_url,
        model=normalized_model,
    )


def print_provider_readiness_summary(*, console: Console) -> None:
    """Print a short readiness summary for configured providers."""
    groq_ready = _has_real_api_key(os.getenv("GROQ_API_KEY", ""))
    gemini_ready = _has_real_api_key(os.getenv("GEMINI_API_KEY", ""))
    ollama_status = get_ollama_readiness()

    console.print("\n[bold]Provider readiness[/bold]")
    console.print(f"- Groq API key: {'ready' if groq_ready else 'missing'}")
    console.print(f"- Gemini API key: {'ready' if gemini_ready else 'missing'}")
    console.print(f"- Ollama: {ollama_status.state}")
    if ollama_status.state != "ready":
        console.print(f"  [dim]{ollama_status.message}[/dim]")


def _print_ollama_status(*, console: Console, status: OllamaReadiness) -> None:
    """Render wizard-facing Ollama setup guidance based on readiness state."""
    if status.state == "ready":
        console.print(f"[green]{status.message}[/green]")
        return
    if status.state == "running_model_missing":
        console.print(f"[yellow]{status.message}[/yellow]")
        console.print(f"[dim]Run: ollama pull {status.model}[/dim]")
        return
    if status.state == "installed_not_running":
        console.print(f"[yellow]{status.message}[/yellow]")
        console.print("[dim]Start Ollama with: ollama serve[/dim]")
        return
    console.print(
        "[yellow]Ollama was not detected in PATH.[/yellow] "
        "You can still use Groq/Gemini and add Ollama later."
    )


def _has_real_api_key(value: str) -> bool:
    """Return True when an API key appears present and non-placeholder."""
    cleaned = value.strip().strip('"').strip("'")
    if not cleaned:
        return False
    lower = cleaned.lower()
    return "your_" not in lower and "_here" not in lower


def _normalize_ollama_model_name(value: str) -> str:
    """Normalize litellm-style `ollama/<model>` names to `<model>`."""
    model = value.strip()
    if "/" in model:
        return model.split("/", maxsplit=1)[1]
    return model


def _http_get_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    """Perform a GET request and return a JSON object payload."""
    request = Request(url=url, method="GET")
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read().decode("utf-8")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object payload.")
    return parsed


def _extract_ollama_model_names(payload: dict[str, Any]) -> set[str]:
    """Extract Ollama model names from `/api/tags` payload."""
    names: set[str] = set()
    models = payload.get("models")
    if not isinstance(models, list):
        return names
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str):
            names.add(name.strip())
    return names


def _has_ollama_model(available_models: set[str], model: str) -> bool:
    """Check model availability, including `:latest` alias support."""
    if model in available_models:
        return True
    latest_alias = f"{model}:latest"
    if latest_alias in available_models:
        return True
    return False


def _read_env_template(template_path: Path) -> str:
    """
    Read `.env.example` contents or fallback template defaults.

    Args:
        template_path: Template file path.

    Returns:
        Template text used to generate `.env`.
    """
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return 'GROQ_API_KEY=""\nGEMINI_API_KEY=""\nOLLAMA_BASE_URL="http://localhost:11434"\n'


def _set_env_value(template: str, key: str, value: str) -> str:
    """
    Upsert one key-value assignment in env-file text.

    Args:
        template: Existing env-file text.
        key: Environment variable name.
        value: Variable value to write.

    Returns:
        Updated env-file text with the target key replaced or appended.
    """
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
    """
    Prompt user input with graceful exit and interrupt semantics.

    Args:
        prompt: Prompt label shown to user.
        console: Rich console for guidance output.
        interrupt_state: Mutable two-step Ctrl+C state flag.
        default: Optional default value for prompt input.

    Returns:
        Trimmed user input, or `None` when caller should exit application.
    """
    while True:
        try:
            with _without_echoctl():
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


@contextlib.contextmanager
def _without_echoctl():
    """
    Temporarily disable terminal control-character echo (e.g., `^C`).

    This keeps interrupt keystrokes from being rendered in the prompt UI while
    still allowing `KeyboardInterrupt` semantics to work as expected.
    """
    if not sys.stdin.isatty():
        yield
        return

    fd = sys.stdin.fileno()
    attrs = termios.tcgetattr(fd)
    updated = attrs[:]
    updated[3] = updated[3] & ~termios.ECHOCTL
    try:
        termios.tcsetattr(fd, termios.TCSANOW, updated)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
