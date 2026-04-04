from __future__ import annotations

from open_council.main import app, parse_cli_args


def test_parse_cli_args_defaults() -> None:
    args = parse_cli_args([])
    assert args.mode == "odin"
    assert args.debug is False


def test_parse_cli_args_custom_mode_and_debug() -> None:
    args = parse_cli_args(["--mode", "leviathan", "--debug"])
    assert args.mode == "leviathan"
    assert args.debug is True


def test_app_prints_mode_and_debug(capsys) -> None:
    app(["--mode", "odin", "--debug"])
    output = capsys.readouterr().out

    assert "Open Council starting..." in output
    assert "Mode: odin" in output
    assert "Debug: enabled" in output


def test_app_warns_for_unwired_mode(capsys) -> None:
    app(["--mode", "artemis"])
    output = capsys.readouterr().out

    assert "Mode: artemis" in output
    assert "artemis mode is planned and not yet wired in Phase 1." in output
