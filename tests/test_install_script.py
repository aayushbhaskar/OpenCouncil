from __future__ import annotations

from pathlib import Path


def test_install_script_contains_required_steps() -> None:
    script_path = Path("install.sh")
    content = script_path.read_text(encoding="utf-8")

    assert "set -euo pipefail" in content
    assert "REPO_URL=" in content
    assert ".open-council-app" in content
    assert "python3 -m venv" in content
    assert "pip\" install --quiet -e" in content
    assert "~/.local/bin" in content or ".local/bin" in content
    assert "ln -sf" in content
