from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import WORKSPACE_ROOT, ensure_workspace_dirs


def _settings_path() -> Path:
    ensure_workspace_dirs()
    return (WORKSPACE_ROOT / "ui_settings.json").resolve()


def load_ui_settings() -> dict[str, Any]:
    path = _settings_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def save_ui_settings(patch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise TypeError("patch must be a dict")
    data = load_ui_settings()
    data.update(patch)
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data

