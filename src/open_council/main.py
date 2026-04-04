"""CLI entrypoint for Open Council."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from rich.console import Console


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
