"""Startup update checks and optional auto-update logic."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from rich.console import Console

from open_council.cli.constants import UPDATE_COMMAND


def maybe_print_update_notice(
    *,
    console: Console,
    run_git_command_fn=None,
    run_update_command_fn=None,
    is_truthy_env_fn=None,
    is_falsey_env_fn=None,
) -> None:
    """
    Print a non-blocking update notice when local checkout lags behind `origin/main`.

    This check is best-effort:
    - Skips when update checks are disabled via `OPEN_COUNCIL_UPDATE_CHECK=0`.
    - Supports opt-in auto-update via `OPEN_COUNCIL_AUTO_UPDATE=1`.
    - Skips when running outside a git checkout.
    - Silently ignores git/network errors.
    """
    if run_git_command_fn is None:
        run_git_command_fn = run_git_command
    if run_update_command_fn is None:
        run_update_command_fn = run_update_command
    if is_truthy_env_fn is None:
        is_truthy_env_fn = is_truthy_env
    if is_falsey_env_fn is None:
        is_falsey_env_fn = is_falsey_env

    if is_falsey_env_fn("OPEN_COUNCIL_UPDATE_CHECK"):
        return

    repo_root = Path(__file__).resolve().parents[3]
    if not (repo_root / ".git").exists():
        return

    current_sha = run_git_command_fn(repo_root, "rev-parse", "HEAD")
    remote_line = run_git_command_fn(repo_root, "ls-remote", "origin", "refs/heads/main")
    if not current_sha or not remote_line:
        return

    remote_sha = remote_line.split(maxsplit=1)[0].strip()
    if current_sha.strip() == remote_sha:
        return

    if is_truthy_env_fn("OPEN_COUNCIL_AUTO_UPDATE"):
        console.print(
            "[yellow]Update available.[/yellow] "
            "Auto-update is enabled; applying update now..."
        )
        if run_update_command_fn(repo_root):
            console.print(
                "[green]Auto-update complete.[/green] "
                "Restart council to use the latest version."
            )
            return
        console.print(
            "[yellow]Auto-update failed.[/yellow] "
            "To update now, exit application and run:\n"
            f"[dim]{UPDATE_COMMAND}[/dim]\n"
            "Then restart council to use the latest version."
        )
        return

    console.print(
        "[yellow]Update available.[/yellow] "
        "To update now, exit application and run:\n"
        f"[dim]{UPDATE_COMMAND}[/dim]\n"
        "Then restart council to use the latest version."
    )


def run_git_command(repo_root: Path, *args: str) -> str | None:
    """Run a git command with a short timeout and return trimmed stdout."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    output = completed.stdout.strip()
    return output or None


def run_update_command(repo_root: Path) -> bool:
    """
    Execute the installer update command in the repository root.

    Returns:
        `True` when update command exits successfully, else `False`.
    """
    try:
        subprocess.run(
            ["/bin/sh", "-lc", UPDATE_COMMAND],
            check=True,
            cwd=repo_root,
            timeout=120,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return True


def is_truthy_env(name: str) -> bool:
    """Return True when env var is explicitly enabled."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def is_falsey_env(name: str) -> bool:
    """Return True when env var is explicitly disabled."""
    return os.getenv(name, "").strip().lower() in {"0", "false", "no", "off"}
