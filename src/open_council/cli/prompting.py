"""Terminal prompt and interrupt-handling utilities."""

from __future__ import annotations

import contextlib
import sys

import termios
from rich.console import Console
from rich.prompt import Prompt


def prompt_with_exit_controls(
    *,
    prompt: str,
    console: Console,
    interrupt_state: dict[str, bool],
    default: str | None,
    ask_fn=None,
    without_echoctl_fn=None,
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
    if ask_fn is None:
        ask_fn = Prompt.ask
    if without_echoctl_fn is None:
        without_echoctl_fn = without_echoctl

    while True:
        try:
            with without_echoctl_fn():
                raw_input = ask_fn(prompt, default=default)
                if raw_input is None:
                    user_input = ""
                else:
                    user_input = str(raw_input).strip()
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
def without_echoctl():
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
