from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

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

    async def astream(self, state: dict[str, Any], stream_mode: str = "updates"):
        _ = stream_mode
        self.received_states.append(dict(state))
        yield {
            "muninn_worker": {
                "parallel_drafts": [{"worker_id": "muninn", "model": "m", "draft": "d"}]
            }
        }
        yield {
            "huginn_worker": {
                "parallel_drafts": [{"worker_id": "huginn", "model": "m", "draft": "d"}]
            }
        }
        yield {"odin_judge": {"final_synthesis": f"verdict for: {state['query']}"}}


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

    def _stub_prompt(_: str, default: str | None = None) -> str:
        _ = default
        return next(prompt_values)

    from open_council import main

    monkeypatch.setattr(main.Prompt, "ask", _stub_prompt)
    monkeypatch.setattr(main, "build_odin_graph", lambda: dummy_graph)
    monkeypatch.setattr(main, "resolve_env_path", lambda console: Path("/tmp/mock.env"))
    monkeypatch.setattr(main, "ensure_env_file_with_wizard", lambda **_: True)
    monkeypatch.setattr(main, "_load_env_file", lambda env_path: None)
    monkeypatch.setattr(main, "print_provider_readiness_summary", lambda console: None)

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

    def _stub_prompt(_: str, default: str | None = None) -> str:
        _ = default
        return next(prompt_values)

    from open_council import main

    monkeypatch.setattr(main.Prompt, "ask", _stub_prompt)
    monkeypatch.setattr(main, "build_odin_graph", lambda: dummy_graph)
    monkeypatch.setattr(main, "resolve_env_path", lambda console: Path("/tmp/mock.env"))
    monkeypatch.setattr(main, "ensure_env_file_with_wizard", lambda **_: True)
    monkeypatch.setattr(main, "_load_env_file", lambda env_path: None)
    monkeypatch.setattr(main, "print_provider_readiness_summary", lambda console: None)

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

    def _stub_prompt(_: str, default: str | None = None) -> str:
        _ = default
        return next(prompt_values)

    from open_council import main

    monkeypatch.setattr(main.Prompt, "ask", _stub_prompt)
    monkeypatch.setattr(main, "build_odin_graph", lambda: dummy_graph)
    monkeypatch.setattr(main, "resolve_env_path", lambda console: Path("/tmp/mock.env"))
    monkeypatch.setattr(main, "ensure_env_file_with_wizard", lambda **_: True)
    monkeypatch.setattr(main, "_load_env_file", lambda env_path: None)
    monkeypatch.setattr(main, "print_provider_readiness_summary", lambda console: None)

    app(["--mode", "odin"])
    output = capsys.readouterr().out

    assert "To exit, use /exit." in output
    assert "To quit, use /quit." in output
    assert "Exiting Open Council." in output
    assert len(dummy_graph.received_states) == 0


def test_first_run_wizard_creates_env_file(tmp_path, monkeypatch, capsys) -> None:
    from open_council import main
    from rich.console import Console

    template_path = tmp_path / ".env.example"
    env_path = tmp_path / ".env"
    template_path.write_text(
        'GROQ_API_KEY="your_groq_api_key_here"\nGEMINI_API_KEY="your_gemini_api_key_here"\n',
        encoding="utf-8",
    )

    prompt_values = iter(["groq-key-123", "gem-key-456"])

    def _stub_prompt(_: str, default: str = "") -> str:
        _ = default
        return next(prompt_values)

    monkeypatch.setattr(main.Prompt, "ask", _stub_prompt)
    monkeypatch.setattr(main.shutil, "which", lambda _: "/usr/local/bin/ollama")

    main.ensure_env_file_with_wizard(
        console=Console(),
        env_path=env_path,
        template_path=template_path,
    )

    output = capsys.readouterr().out
    content = env_path.read_text(encoding="utf-8")
    assert "Ollama binary detected" in output
    assert 'GROQ_API_KEY="groq-key-123"' in content
    assert 'GEMINI_API_KEY="gem-key-456"' in content


def test_first_run_wizard_skips_when_env_exists(tmp_path, monkeypatch) -> None:
    from open_council import main
    from rich.console import Console

    env_path = tmp_path / ".env"
    template_path = tmp_path / ".env.example"
    env_path.write_text("EXISTING=1\n", encoding="utf-8")
    template_path.write_text('GROQ_API_KEY="x"\n', encoding="utf-8")

    with patch.object(main.Prompt, "ask", side_effect=AssertionError("Should not prompt")):
        result = main.ensure_env_file_with_wizard(
            console=Console(),
            env_path=env_path,
            template_path=template_path,
        )

    assert result is True
    assert env_path.read_text(encoding="utf-8") == "EXISTING=1\n"


def test_first_run_wizard_allows_exit_command(tmp_path, monkeypatch, capsys) -> None:
    from open_council import main
    from rich.console import Console

    env_path = tmp_path / ".env"
    template_path = tmp_path / ".env.example"
    template_path.write_text('GROQ_API_KEY="x"\nGEMINI_API_KEY="y"\n', encoding="utf-8")

    monkeypatch.setattr(main.Prompt, "ask", lambda prompt, default="": "/exit")

    result = main.ensure_env_file_with_wizard(
        console=Console(),
        env_path=env_path,
        template_path=template_path,
    )

    output = capsys.readouterr().out
    assert result is False
    assert "Exiting Open Council." in output
    assert env_path.exists() is False


def test_first_run_wizard_keyboard_interrupt_twice_exits(tmp_path, monkeypatch, capsys) -> None:
    from open_council import main
    from rich.console import Console

    env_path = tmp_path / ".env"
    template_path = tmp_path / ".env.example"
    template_path.write_text('GROQ_API_KEY="x"\nGEMINI_API_KEY="y"\n', encoding="utf-8")

    calls = {"count": 0}

    def _interrupting_prompt(prompt: str, default: str = "") -> str:
        _ = prompt, default
        calls["count"] += 1
        raise KeyboardInterrupt()

    monkeypatch.setattr(main.Prompt, "ask", _interrupting_prompt)

    result = main.ensure_env_file_with_wizard(
        console=Console(),
        env_path=env_path,
        template_path=template_path,
    )
    output = capsys.readouterr().out

    assert result is False
    assert "Press Ctrl+C again to exit, or type /exit." in output
    assert "Exiting Open Council." in output


def test_resolve_env_path_uses_global_when_present(tmp_path, monkeypatch, capsys) -> None:
    from open_council import main
    from rich.console import Console

    global_env = tmp_path / ".open-council" / ".env"
    global_env.parent.mkdir(parents=True, exist_ok=True)
    global_env.write_text("GROQ_API_KEY=x\n", encoding="utf-8")
    local_env = tmp_path / ".env"
    local_env.write_text("GROQ_API_KEY=local\n", encoding="utf-8")

    monkeypatch.setattr(main, "GLOBAL_ENV_PATH", global_env)
    monkeypatch.setattr(main, "LOCAL_ENV_PATH", local_env)

    resolved = main.resolve_env_path(console=Console())
    output = capsys.readouterr().out

    assert resolved == global_env
    assert "Using local .env for now" not in output


def test_resolve_env_path_falls_back_to_local_with_hint(tmp_path, monkeypatch, capsys) -> None:
    from open_council import main
    from rich.console import Console

    global_env = tmp_path / ".open-council" / ".env"
    local_env = tmp_path / ".env"
    local_env.write_text("GROQ_API_KEY=local\n", encoding="utf-8")

    monkeypatch.setattr(main, "GLOBAL_ENV_PATH", global_env)
    monkeypatch.setattr(main, "LOCAL_ENV_PATH", local_env)

    resolved = main.resolve_env_path(console=Console())
    output = capsys.readouterr().out

    assert resolved == local_env
    assert "Using local .env for now" in output


def test_get_ollama_readiness_not_installed(monkeypatch) -> None:
    from open_council import main

    monkeypatch.setattr(main.shutil, "which", lambda _: None)
    status = main.get_ollama_readiness()

    assert status.state == "not_installed"


def test_get_ollama_readiness_installed_not_running(monkeypatch) -> None:
    from open_council import main

    monkeypatch.setattr(main.shutil, "which", lambda _: "/usr/local/bin/ollama")

    def _boom(url: str, *, timeout_seconds: float):
        _ = url, timeout_seconds
        raise OSError("connection refused")

    monkeypatch.setattr(main, "_http_get_json", _boom)
    status = main.get_ollama_readiness()

    assert status.state == "installed_not_running"


def test_get_ollama_readiness_model_missing(monkeypatch) -> None:
    from open_council import main

    monkeypatch.setattr(main.shutil, "which", lambda _: "/usr/local/bin/ollama")
    monkeypatch.setattr(
        main,
        "_http_get_json",
        lambda url, timeout_seconds: {"models": [{"name": "llama2:latest"}]},
    )
    status = main.get_ollama_readiness()

    assert status.state == "running_model_missing"


def test_get_ollama_readiness_ready(monkeypatch) -> None:
    from open_council import main

    monkeypatch.setattr(main.shutil, "which", lambda _: "/usr/local/bin/ollama")
    monkeypatch.setattr(
        main,
        "_http_get_json",
        lambda url, timeout_seconds: {"models": [{"name": "llama3.1:latest"}]},
    )
    status = main.get_ollama_readiness()

    assert status.state == "ready"
