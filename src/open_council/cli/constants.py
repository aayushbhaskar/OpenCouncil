"""Shared CLI constants and canonical paths."""

from __future__ import annotations

from pathlib import Path

ALL_MODES = ("odin", "artemis", "leviathan")
WIRED_MODES = frozenset({"odin"})
UPDATE_COMMAND = "curl -fsSL https://aayushbhaskar.github.io/OpenCouncil/install.sh | bash"
CONFIGURABLE_FLAGS = (
    "OPEN_COUNCIL_UPDATE_CHECK",
    "OPEN_COUNCIL_AUTO_UPDATE",
)

GLOBAL_CONFIG_DIR = Path.home() / ".open-council"
GLOBAL_ENV_PATH = GLOBAL_CONFIG_DIR / ".env"
LOCAL_ENV_PATH = Path(".env")
TEMPLATE_ENV_PATH = Path(__file__).resolve().parents[3] / ".env.example"
