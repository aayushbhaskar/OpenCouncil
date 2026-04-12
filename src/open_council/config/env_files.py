"""Helpers for reading and writing `.env`-style configuration text."""

from __future__ import annotations

import re
from pathlib import Path


def read_env_template(template_path: Path) -> str:
    """
    Read `.env.example` contents or fallback template defaults.

    Args:
        template_path: Template file path.

    Returns:
        Template text used to generate `.env`.
    """
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return (
        'GROQ_API_KEY=""\n'
        'OPENROUTER_API_KEY=""\n'
        'GEMINI_API_KEY=""\n'
        'OLLAMA_BASE_URL="http://localhost:11434"\n'
    )


def set_env_value(template: str, key: str, value: str) -> str:
    """
    Upsert one key-value assignment in env-file text.

    Args:
        template: Existing env-file text.
        key: Environment variable name.
        value: Variable value to write.

    Returns:
        Updated env-file text with the target key replaced or appended.
    """
    pattern = rf"(?m)^{re.escape(key)}=.*$"
    replacement = f'{key}="{value}"'
    if re.search(pattern, template):
        return re.sub(pattern, replacement, template)
    suffix = "" if template.endswith("\n") else "\n"
    return f"{template}{suffix}{replacement}\n"


def normalize_flag_value(value: str) -> str | None:
    """Normalize boolean-like env values to `1`/`0`, or return None."""
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return "1"
    if lowered in {"0", "false", "no", "off"}:
        return "0"
    return None
