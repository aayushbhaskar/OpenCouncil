from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from open_council.core.llm import LiteLLMClient, configure_litellm_logging


def _fake_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.mark.asyncio
async def test_complete_short_circuits_on_primary_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def _stub_acompletion(*, model: str, **_: object) -> SimpleNamespace:
        calls.append(model)
        return _fake_response("primary-ok")

    monkeypatch.setattr("open_council.core.llm.acompletion", _stub_acompletion)

    client = LiteLLMClient()
    result = await client.complete([{"role": "user", "content": "hello"}])

    assert result.ok is True
    assert result.content == "primary-ok"
    assert len(calls) == 1
    assert calls[0].startswith("groq/")


@pytest.mark.asyncio
async def test_complete_falls_back_in_expected_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def _stub_acompletion(*, model: str, **_: object) -> SimpleNamespace:
        calls.append(model)
        if model.startswith("groq/") or model.startswith("openrouter/"):
            raise RuntimeError(f"failure-{model}")
        return _fake_response("ollama-ok")

    monkeypatch.setattr("open_council.core.llm.acompletion", _stub_acompletion)

    client = LiteLLMClient()
    result = await client.complete([{"role": "user", "content": "hello"}])

    assert result.ok is True
    assert result.provider == "ollama"
    assert result.content == "ollama-ok"
    assert len(calls) == 3
    assert calls[0].startswith("groq/")
    assert calls[1].startswith("openrouter/")
    assert calls[2].startswith("ollama/")


@pytest.mark.asyncio
async def test_complete_returns_safe_error_when_all_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub_acompletion(*, model: str, **_: object) -> SimpleNamespace:
        raise RuntimeError(f"failure-{model}")

    monkeypatch.setattr("open_council.core.llm.acompletion", _stub_acompletion)

    client = LiteLLMClient()
    result = await client.complete([{"role": "user", "content": "hello"}])

    assert result.ok is False
    assert result.error is not None
    assert "All fallback providers failed" in result.error
    assert len(result.attempts) == 3


@pytest.mark.asyncio
async def test_complete_prints_simple_retry_message(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    calls: list[str] = []

    async def _stub_acompletion(*, model: str, **_: object) -> SimpleNamespace:
        calls.append(model)
        if model.startswith("groq/"):
            raise RuntimeError("groq-down")
        return _fake_response("openrouter-ok")

    monkeypatch.setattr("open_council.core.llm.acompletion", _stub_acompletion)
    monkeypatch.setenv("OPEN_COUNCIL_DEBUG", "0")

    client = LiteLLMClient()
    result = await client.complete([{"role": "user", "content": "hello"}])
    output = capsys.readouterr().out

    assert result.ok is True
    assert result.provider == "openrouter"
    assert "Provider retry: groq unavailable, trying openrouter..." in output
    assert calls[0].startswith("groq/")
    assert calls[1].startswith("openrouter/")


@pytest.mark.asyncio
async def test_complete_honors_provider_model_override_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def _stub_acompletion(*, model: str, **_: object) -> SimpleNamespace:
        calls.append(model)
        if model == "openrouter/custom":
            raise RuntimeError("custom-down")
        return _fake_response("ok")

    monkeypatch.setattr("open_council.core.llm.acompletion", _stub_acompletion)

    client = LiteLLMClient()
    result = await client.complete(
        [{"role": "user", "content": "hello"}],
        provider_models=[
            ("openrouter", "openrouter/custom"),
            ("groq", "groq/fallback"),
        ],
    )

    assert result.ok is True
    assert calls == ["openrouter/custom", "groq/fallback"]


@pytest.mark.asyncio
async def test_openrouter_requests_low_reasoning_effort_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def _stub_acompletion(*, model: str, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return _fake_response("ok")

    monkeypatch.setattr("open_council.core.llm.acompletion", _stub_acompletion)
    monkeypatch.delenv("OPENROUTER_REASONING_EFFORT", raising=False)

    client = LiteLLMClient()
    await client.complete(
        [{"role": "user", "content": "hello"}],
        provider_models=[("openrouter", "openrouter/minimax/minimax-m3.1-flash:free")],
    )
    assert captured["extra_body"] == {"reasoning": {"effort": "low"}}

    captured.clear()
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "")
    await client.complete(
        [{"role": "user", "content": "hello"}],
        provider_models=[("openrouter", "openrouter/minimax/minimax-m3.1-flash:free")],
    )
    assert "extra_body" not in captured


@pytest.mark.asyncio
async def test_fake_mode_skips_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fail(**_: object) -> SimpleNamespace:
        raise AssertionError("fake mode must not call providers")

    monkeypatch.setattr("open_council.core.llm.acompletion", _fail)
    monkeypatch.setenv("OPEN_COUNCIL_FAKE_LLM", "1")

    result = await LiteLLMClient().complete(
        [
            {"role": "system", "content": "You are the Artemis Critic."},
            {"role": "user", "content": "Specialist draft:\n[fake first pass] draft"},
        ]
    )
    assert result.ok is True
    assert result.provider == "fake"
    assert result.content.startswith("VERDICT: REJECT")


@pytest.mark.asyncio
async def test_complete_hard_times_out_provider_call(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _slow_acompletion(*, model: str, **_: object) -> SimpleNamespace:
        _ = model
        await __import__("asyncio").sleep(6.2)
        return _fake_response("late")

    monkeypatch.setattr("open_council.core.llm.acompletion", _slow_acompletion)
    monkeypatch.setenv("LITELLM_TIMEOUT_SECONDS", "0.01")

    client = LiteLLMClient()
    result = await client.complete([{"role": "user", "content": "hello"}])

    assert result.ok is False
    assert result.error is not None


def test_configure_litellm_logging_toggles_flags() -> None:
    import litellm

    litellm.set_verbose = False
    configure_litellm_logging(debug=False)
    assert litellm.suppress_debug_info is True
    assert "LITELLM_LOG" not in os.environ

    configure_litellm_logging(debug=True)
    assert litellm.suppress_debug_info is False
    assert os.environ.get("LITELLM_LOG") == "DEBUG"
