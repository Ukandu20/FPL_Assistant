from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import CONFIG_ROOT, LEGACY_CONFIG_ROOT


def iter_config_candidates(relative_path: str | Path) -> list[Path]:
    rel = Path(relative_path)
    return [
        CONFIG_ROOT / rel,
        LEGACY_CONFIG_ROOT / rel,
    ]


def resolve_config_path(relative_path: str | Path, *, required: bool = True) -> Path:
    candidates = iter_config_candidates(relative_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        joined = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find config '{relative_path}'. Checked: {joined}")
    return candidates[0]


def load_json_config(relative_path: str | Path, *, default: Any = None) -> Any:
    path = resolve_config_path(relative_path, required=default is None)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

