"""Canonical filesystem and season discovery for the FPL Streamlit app."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
FPL_ROOT = PROCESSED_ROOT / "fpl"
UNDERSTAT_ROOT = PROCESSED_ROOT / "understat"
PREDICTIONS_ROOT = PROJECT_ROOT / "data" / "predictions"
PRICE_CATEGORY_CONFIG_PATH = PROJECT_ROOT / "config" / "fpl_price_categories.json"


def discover_leagues(root: Path = FPL_ROOT) -> list[str]:
    """Return provider leagues with deterministic ordering."""
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def discover_seasons(
    league: str,
    *,
    root: Path = FPL_ROOT,
    required_path: str = "season/cleaned_players.csv",
) -> list[str]:
    """Return newest-first seasons that satisfy a provider data contract."""
    league_dir = root / league
    if not league_dir.is_dir():
        return []
    seasons = [
        path.name
        for path in league_dir.iterdir()
        if path.is_dir() and (path / required_path).is_file()
    ]
    return sorted(seasons, reverse=True)


def current_season(
    league: str,
    *,
    root: Path = FPL_ROOT,
    required_path: str = "season/cleaned_players.csv",
) -> str | None:
    """Resolve the current season as the newest season with required data."""
    seasons = discover_seasons(
        league, root=root, required_path=required_path
    )
    return seasons[0] if seasons else None


def fpl_season_path(league: str, season: str) -> Path:
    return FPL_ROOT / league / season / "season" / "cleaned_players.csv"


def fpl_gameweeks_path(league: str, season: str) -> Path:
    return FPL_ROOT / league / season / "gws" / "merged_gws.csv"


def understat_team_season_path(league: str, season: str) -> Path:
    return UNDERSTAT_ROOT / league / season / "team_season.csv"


def file_version(path: Path) -> tuple[int, int] | None:
    """Return a lightweight cache key that changes with file contents."""
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size
