from __future__ import annotations

from types import SimpleNamespace

import pytest

from open_council.core.llm import LiteLLMClient


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
        if model.startswith("groq/") or model.startswith("gemini/"):
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
    assert calls[1].startswith("gemini/")
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
