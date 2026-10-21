from __future__ import annotations

from pathlib import Path

import pytest

from open_council import main
from open_council.runtime import checkpoints
from open_council.runtime.checkpoints import list_threads, restore_chat_history


@pytest.fixture()
def fake_session(tmp_path, monkeypatch):
    """Run `app()` with fake LLM/search, real graphs, and a temp checkpoint DB."""
    monkeypatch.setenv("OPEN_COUNCIL_FAKE_LLM", "1")
    monkeypatch.setattr(checkpoints, "DEFAULT_CHECKPOINT_PATH", tmp_path / "checkpoints.sqlite")
    monkeypatch.setattr(
        "open_council.tools.search_cache.DEFAULT_CACHE_PATH", tmp_path / "search_cache.sqlite"
    )
    monkeypatch.setattr(main, "resolve_env_path", lambda console: Path("/tmp/mock.env"))
    monkeypatch.setattr(main, "ensure_env_file_with_wizard", lambda **_: True)
    monkeypatch.setattr(main, "_load_env_file", lambda env_path: None)
    monkeypatch.setattr(main, "print_provider_readiness_summary", lambda console: None)
    monkeypatch.setattr(main, "maybe_print_update_notice", lambda console: None)

    def _run(argv: list[str], inputs: list[str]) -> None:
        values = iter(inputs)
        monkeypatch.setattr(main.Prompt, "ask", lambda _prompt, default=None: next(values))
        main.app(argv)

    return _run


def test_resume_restores_artemis_history_and_mode(fake_session, capsys) -> None:
    fake_session(["--mode", "artemis"], ["first research question", "/exit"])
    threads = list_threads()
    assert len(threads) == 1
    thread = threads[0]
    assert thread.mode == "artemis"
    assert thread.title == "first research question"
    capsys.readouterr()

    fake_session(["--thread", thread.thread_id], ["follow up question", "/exit"])
    output = capsys.readouterr().out

    assert "Mode: artemis" in output
    assert f"Resumed thread {thread.thread_id} (artemis, 1 previous" in output
    assert "Last question: first research question" in output
    assert "Artemis Research Verdict" in output
    # Prior question + prior answer + current question reach the synthesizer.
    assert "History messages seen: 3" in output


def test_mode_switch_starts_new_thread(fake_session, capsys) -> None:
    fake_session(["--mode", "odin"], ["odin question", "/mode artemis", "artemis question", "/exit"])
    output = capsys.readouterr().out

    assert "Started a new artemis thread:" in output
    modes = {thread.mode: thread.title for thread in list_threads()}
    assert modes == {"odin": "odin question", "artemis": "artemis question"}


def test_explicit_conflicting_mode_opens_thread_in_its_own_mode(fake_session, capsys) -> None:
    fake_session(["--mode", "artemis"], ["research question", "/exit"])
    thread_id = list_threads()[0].thread_id
    capsys.readouterr()

    fake_session(["--mode", "odin", "--thread", thread_id], ["/exit"])
    output = capsys.readouterr().out

    assert f"Thread {thread_id} belongs to artemis mode" in output
    assert "Mode: artemis" in output


def test_restore_chat_history_appends_answer_or_drops_cancelled_question() -> None:
    completed = {
        "chat_history": [{"role": "user", "content": "q1"}],
        "final_synthesis": "a1",
    }
    assert restore_chat_history(completed) == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]

    cancelled = {
        "chat_history": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        "final_synthesis": "",
    }
    assert restore_chat_history(cancelled) == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    assert restore_chat_history({}) == []
