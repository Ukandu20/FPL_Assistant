from __future__ import annotations

from pathlib import Path

from fpl_assistant.platform.paths import artifact_dir, test_run_dir


def get_test_run_dir(name: str) -> Path:
    return test_run_dir(name)


def get_test_soccerdata_dir() -> Path:
    return artifact_dir("cache", "soccerdata_test")

