"""Publish season-level player production profiles for the application."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fpl_assistant.domain.player_profiles import build_player_profiles


PROJECT_ROOT = Path(__file__).resolve().parents[5]


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def profile_output_path(project_root: Path, league: str, season: str) -> Path:
    return (
        project_root
        / "data"
        / "processed"
        / "fpl"
        / league
        / season
        / "analytics"
        / "player_profiles.csv"
    )


def publish_player_profiles(
    league: str,
    season: str,
    *,
    project_root: Path = PROJECT_ROOT,
    minimum_minutes: int = 450,
    established_minutes: int = 900,
    prior_minutes: int = 900,
) -> pd.DataFrame:
    """Build and write the player-profile artifact for one league-season."""
    fpl_path = (
        project_root
        / "data"
        / "processed"
        / "fpl"
        / league
        / season
        / "season"
        / "cleaned_players.csv"
    )
    if not fpl_path.is_file():
        raise FileNotFoundError(f"FPL season file does not exist: {fpl_path}")

    whoscored_root = (
        project_root
        / "data"
        / "processed"
        / "whoscored"
        / league
        / season
        / "player_season"
    )
    profiles = build_player_profiles(
        pd.read_csv(fpl_path),
        shooting=_read_optional(whoscored_root / "shooting.csv"),
        passing=_read_optional(whoscored_root / "passing.csv"),
        defense=_read_optional(whoscored_root / "defense.csv"),
        keepers=_read_optional(whoscored_root / "keepers.csv"),
        season=season,
        minimum_minutes=minimum_minutes,
        established_minutes=established_minutes,
        prior_minutes=prior_minutes,
    )
    output_path = profile_output_path(project_root, league, season)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output_path, index=False)
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="ENG-Premier League")
    parser.add_argument("--season", required=True)
    parser.add_argument("--minimum-minutes", type=int, default=450)
    parser.add_argument("--established-minutes", type=int, default=900)
    parser.add_argument("--prior-minutes", type=int, default=900)
    args = parser.parse_args()
    profiles = publish_player_profiles(
        args.league,
        args.season,
        minimum_minutes=args.minimum_minutes,
        established_minutes=args.established_minutes,
        prior_minutes=args.prior_minutes,
    )
    print(
        f"Published {len(profiles):,} player profiles for "
        f"{args.league} {args.season}."
    )


if __name__ == "__main__":
    main()

