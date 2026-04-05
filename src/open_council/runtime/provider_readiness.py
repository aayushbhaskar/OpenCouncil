"""Provider readiness checks and summary rendering."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from rich.console import Console


@dataclass(slots=True)
class OllamaReadiness:
    """Status payload for local Ollama installation/readiness checks."""

    state: str
    message: str
    base_url: str
    model: str


def get_ollama_readiness(
    *,
    which_fn=shutil.which,
    http_get_json_fn=None,
) -> OllamaReadiness:
    """
    Check whether Ollama is installed, reachable, and model-ready.

    Returns:
        `OllamaReadiness` status in one of:
        - `not_installed`
        - `installed_not_running`
        - `running_model_missing`
        - `ready`
    """
    if http_get_json_fn is None:
        http_get_json_fn = http_get_json

    ollama_path = which_fn("ollama")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    configured_model = os.getenv("OLLAMA_MODEL", "ollama/llama3.1").strip()
    normalized_model = normalize_ollama_model_name(configured_model)

    if not ollama_path:
        return OllamaReadiness(
            state="not_installed",
            message="Ollama not detected in PATH.",
            base_url=base_url,
            model=normalized_model,
        )

    tags_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        payload = http_get_json_fn(tags_url, timeout_seconds=2.0)
    except (TimeoutError, URLError, OSError, ValueError):
        return OllamaReadiness(
            state="installed_not_running",
            message=(
                f"Ollama binary detected at {ollama_path}, but server is not reachable at {base_url}."
            ),
            base_url=base_url,
            model=normalized_model,
        )

    available_models = extract_ollama_model_names(payload)
    if not has_ollama_model(available_models, normalized_model):
        return OllamaReadiness(
            state="running_model_missing",
            message=(
                f"Ollama server is reachable at {base_url}, but model '{normalized_model}' is missing."
            ),
            base_url=base_url,
            model=normalized_model,
        )

    return OllamaReadiness(
        state="ready",
        message=f"Ollama is ready at {base_url} with model '{normalized_model}'.",
        base_url=base_url,
        model=normalized_model,
    )


def print_provider_readiness_summary(
    *,
    console: Console,
    get_ollama_readiness_fn=get_ollama_readiness,
) -> None:
    """Print a short readiness summary for configured providers."""
    groq_ready = has_real_api_key(os.getenv("GROQ_API_KEY", ""))
    gemini_ready = has_real_api_key(os.getenv("GEMINI_API_KEY", ""))
    ollama_status = get_ollama_readiness_fn()

    console.print("\n[bold]Provider readiness[/bold]")
    console.print(f"- Groq API key: {'ready' if groq_ready else 'missing'}")
    console.print(f"- Gemini API key: {'ready' if gemini_ready else 'missing'}")
    console.print(f"- Ollama: {ollama_status.state}")
    if ollama_status.state != "ready":
        console.print(f"  [dim]{ollama_status.message}[/dim]")


def print_ollama_status(*, console: Console, status: OllamaReadiness) -> None:
    """Render wizard-facing Ollama setup guidance based on readiness state."""
    if status.state == "ready":
        console.print(f"[green]{status.message}[/green]")
        return
    if status.state == "running_model_missing":
        console.print(f"[yellow]{status.message}[/yellow]")
        console.print(f"[dim]Run: ollama pull {status.model}[/dim]")
        return
    if status.state == "installed_not_running":
        console.print(f"[yellow]{status.message}[/yellow]")
        console.print("[dim]Start Ollama with: ollama serve[/dim]")
        return
    console.print(
        "[yellow]Ollama was not detected in PATH.[/yellow] "
        "You can still use Groq/Gemini and add Ollama later."
    )


def has_real_api_key(value: str) -> bool:
    """Return True when an API key appears present and non-placeholder."""
    cleaned = value.strip().strip('"').strip("'")
    if not cleaned:
        return False
    lower = cleaned.lower()
    return "your_" not in lower and "_here" not in lower


def normalize_ollama_model_name(value: str) -> str:
    """Normalize litellm-style `ollama/<model>` names to `<model>`."""
    model = value.strip()
    if "/" in model:
        return model.split("/", maxsplit=1)[1]
    return model


def http_get_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    """Perform a GET request and return a JSON object payload."""
    request = Request(url=url, method="GET")
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read().decode("utf-8")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object payload.")
    return parsed


def extract_ollama_model_names(payload: dict[str, Any]) -> set[str]:
    """Extract Ollama model names from `/api/tags` payload."""
    names: set[str] = set()
    models = payload.get("models")
    if not isinstance(models, list):
        return names
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str):
            names.add(name.strip())
    return names


def has_ollama_model(available_models: set[str], model: str) -> bool:
    """Check model availability, including `:latest` alias support."""
    if model in available_models:
        return True
    latest_alias = f"{model}:latest"
    if latest_alias in available_models:
        return True
    return False
