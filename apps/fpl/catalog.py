"""Canonical filesystem and season discovery for the FPL Streamlit app."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
FPL_ROOT = PROCESSED_ROOT / "fpl"
RAW_FPL_ROOT = RAW_ROOT / "fpl"
UNDERSTAT_ROOT = PROCESSED_ROOT / "understat"
WHOSCORED_ROOT = PROCESSED_ROOT / "whoscored"
PREDICTIONS_ROOT = PROJECT_ROOT / "data" / "predictions"
ARCHETYPE_ROOT = PROCESSED_ROOT / "archetypes"
FIXTURE_REGISTRY_ROOT = PROCESSED_ROOT / "registry" / "fixtures"
PRICE_CATEGORY_CONFIG_PATH = PROJECT_ROOT / "config" / "fpl_price_categories.json"
PLAYER_IMAGE_CONFIG_PATH = PROJECT_ROOT / "config" / "fpl_image_assets.json"


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


def fpl_player_profiles_path(league: str, season: str) -> Path:
    return FPL_ROOT / league / season / "analytics" / "player_profiles.csv"


def fpl_raw_players_path(league: str, season: str) -> Path:
    return RAW_FPL_ROOT / league / season / "players_raw.csv"


def fpl_raw_fixtures_path(league: str, season: str) -> Path:
    return RAW_FPL_ROOT / league / season / "season" / "fixtures.csv"


def fpl_raw_teams_path(league: str, season: str) -> Path:
    return RAW_FPL_ROOT / league / season / "season" / "teams.csv"


def fpl_fixture_metadata_path(league: str, season: str) -> Path:
    return (
        RAW_FPL_ROOT
        / league
        / season
        / "season"
        / "fixture_metadata_per_team_resolved.csv"
    )


def fixture_calendar_path(season: str) -> Path:
    return FIXTURE_REGISTRY_ROOT / season / "fixture_calendar.csv"


def expected_points_root(season: str) -> Path:
    return PREDICTIONS_ROOT / "expected_points" / season


def latest_archetype_snapshot(
    season: str | None = None, *, root: Path = ARCHETYPE_ROOT
) -> Path | None:
    """Return the newest complete immutable archetype snapshot for a season."""
    candidates: list[tuple[datetime, Path]] = []
    for path in root.glob("model_version=*/snapshot=*") if root.is_dir() else ():
        if not (path / "archetypes.jsonl").is_file():
            continue
        stamp = path.name.removeprefix("snapshot=")
        try:
            snapshot = datetime.strptime(stamp, "%Y-%m-%dT%H-%M-%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        if season is not None:
            try:
                start_year, end_year = (int(value) for value in season.split("-"))
            except (TypeError, ValueError):
                continue
            season_start = datetime(start_year, 7, 1, tzinfo=timezone.utc)
            season_end = datetime(end_year, 7, 1, tzinfo=timezone.utc)
            if not season_start <= snapshot < season_end:
                continue
        candidates.append((snapshot, path))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def understat_team_season_path(league: str, season: str) -> Path:
    return UNDERSTAT_ROOT / league / season / "team_season.csv"


def whoscored_roles_path(league: str, season: str) -> Path:
    return WHOSCORED_ROOT / league / season / "player_season" / "roles.csv"


def file_version(path: Path) -> tuple[int, int] | None:
    """Return a lightweight cache key that changes with file contents."""
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size
