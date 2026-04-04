from __future__ import annotations

from typing import Any

from open_council.main import app, parse_cli_args


class _DummyGraph:
    def __init__(self) -> None:
        self.received_states: list[dict[str, Any]] = []

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.received_states.append(dict(state))
        return {
            **state,
            "parallel_drafts": [{"worker_id": "muninn", "model": "m", "draft": "d"}],
            "final_synthesis": f"verdict for: {state['query']}",
        }


def test_parse_cli_args_defaults() -> None:
    args = parse_cli_args([])
    assert args.mode == "odin"
    assert args.debug is False


def test_parse_cli_args_custom_mode_and_debug() -> None:
    args = parse_cli_args(["--mode", "leviathan", "--debug"])
    assert args.mode == "leviathan"
    assert args.debug is True


def test_app_prints_mode_and_debug(capsys, monkeypatch) -> None:
    dummy_graph = _DummyGraph()
    prompt_values = iter(["/quit"])

    def _stub_prompt(_: str) -> str:
        return next(prompt_values)

    from open_council import main

    monkeypatch.setattr(main.Prompt, "ask", _stub_prompt)
    monkeypatch.setattr(main, "build_odin_graph", lambda: dummy_graph)

    app(["--mode", "odin", "--debug"])
    output = capsys.readouterr().out

    assert "Open Council starting..." in output
    assert "Mode: odin" in output
    assert "Debug: enabled" in output
    assert "Odin ready." in output
    assert "Exiting Open Council." in output


def test_app_warns_for_unwired_mode(capsys) -> None:
    app(["--mode", "artemis"])
    output = capsys.readouterr().out

    assert "Mode: artemis" in output
    assert "artemis mode is planned and not yet wired in Phase 1." in output


def test_app_persists_state_across_turns(capsys, monkeypatch) -> None:
    dummy_graph = _DummyGraph()
    prompt_values = iter(["first question", "second question", "/exit"])

    def _stub_prompt(_: str) -> str:
        return next(prompt_values)

    from open_council import main

    monkeypatch.setattr(main.Prompt, "ask", _stub_prompt)
    monkeypatch.setattr(main, "build_odin_graph", lambda: dummy_graph)

    app(["--mode", "odin"])
    output = capsys.readouterr().out

    assert "verdict for: first question" in output
    assert "verdict for: second question" in output
    assert len(dummy_graph.received_states) == 2
    assert dummy_graph.received_states[0]["query"] == "first question"
    assert dummy_graph.received_states[1]["query"] == "second question"

    second_history = dummy_graph.received_states[1]["chat_history"]
    assert second_history[0]["role"] == "user"
    assert second_history[0]["content"] == "first question"
    assert second_history[1]["role"] == "assistant"
    assert second_history[1]["content"] == "verdict for: first question"
    assert second_history[2]["role"] == "user"
    assert second_history[2]["content"] == "second question"


def test_app_requires_slash_exit_commands(capsys, monkeypatch) -> None:
    dummy_graph = _DummyGraph()
    prompt_values = iter(["exit", "quit", "/exit"])

    def _stub_prompt(_: str) -> str:
        return next(prompt_values)

    from open_council import main

    monkeypatch.setattr(main.Prompt, "ask", _stub_prompt)
    monkeypatch.setattr(main, "build_odin_graph", lambda: dummy_graph)

    app(["--mode", "odin"])
    output = capsys.readouterr().out

    assert "To exit, use /exit." in output
    assert "To quit, use /quit." in output
    assert "Exiting Open Council." in output
    assert len(dummy_graph.received_states) == 0
