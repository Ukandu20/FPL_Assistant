from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Could not determine repository root for FPL Assistant.")


REPO_ROOT = find_repo_root()
SRC_ROOT = REPO_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "fpl_assistant"
CONFIG_ROOT = REPO_ROOT / "config"
LEGACY_CONFIG_ROOT = REPO_ROOT / "data" / "config"
DATA_ROOT = REPO_ROOT / "data"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


def artifact_path(*parts: str, create_parent: bool = False) -> Path:
    path = ARTIFACTS_ROOT.joinpath(*parts)
    if create_parent:
        ensure_dir(path.parent)
    return path


def artifact_dir(*parts: str) -> Path:
    return ensure_dir(ARTIFACTS_ROOT.joinpath(*parts))


def cache_dir(*parts: str) -> Path:
    return artifact_dir("cache", *parts)


def test_run_dir(name: str) -> Path:
    return artifact_dir("test_runs", name)

