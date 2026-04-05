"""Environment path resolution, loading, and first-run wizard."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from open_council.cli.constants import GLOBAL_ENV_PATH, LOCAL_ENV_PATH
from open_council.config.env_files import read_env_template, set_env_value


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


def load_env_file(env_path: Path) -> None:
    """Load environment variables from the resolved env file path."""
    load_dotenv(dotenv_path=env_path, override=True)


def ensure_env_file_with_wizard(
    *,
    console: Console,
    env_path: Path = Path(".env"),
    template_path: Path = Path(".env.example"),
    prompt_with_exit_controls_fn,
    get_ollama_readiness_fn,
    print_ollama_status_fn,
) -> bool:
    """
    Ensure `.env` exists, guiding first-run setup when missing.

    Args:
        console: Rich console for wizard output.
        env_path: Destination `.env` file path.
        template_path: `.env.example` template path.
        prompt_with_exit_controls_fn: Prompt helper callback.
        get_ollama_readiness_fn: Ollama readiness probe callback.
        print_ollama_status_fn: Ollama status print callback.

    Returns:
        `True` when setup completes or file already exists.
        `False` when the user exits during the wizard.
    """
    if env_path.exists():
        console.print(f"[dim]Using existing config: {env_path}[/dim]")
        return True

    console.print("[yellow].env not found. Starting first-run setup.[/yellow]")
    interrupt_state = {"armed": False}

    groq_api_key = prompt_with_exit_controls_fn(
        prompt="Paste your GROQ_API_KEY (press Enter to skip)",
        console=console,
        interrupt_state=interrupt_state,
        default="",
    )
    if groq_api_key is None:
        return False
    gemini_api_key = prompt_with_exit_controls_fn(
        prompt="Paste your GEMINI_API_KEY (press Enter to skip)",
        console=console,
        interrupt_state=interrupt_state,
        default="",
    )
    if gemini_api_key is None:
        return False

    ollama_status = get_ollama_readiness_fn()
    print_ollama_status_fn(console=console, status=ollama_status)

    template = read_env_template(template_path)
    rendered = set_env_value(template, "GROQ_API_KEY", groq_api_key) if groq_api_key else template
    rendered = set_env_value(rendered, "GEMINI_API_KEY", gemini_api_key) if gemini_api_key else rendered
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(rendered, encoding="utf-8")
    console.print(f"[green]Created {env_path}[/green]. Update it anytime as needed.")
    return True
