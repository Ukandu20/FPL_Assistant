"""Shared, version-keyed data access for FPL Streamlit pages."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from apps.fpl.catalog import (
    ARCHETYPE_ROOT,
    PREDICTIONS_ROOT,
    file_version,
    fpl_gameweeks_path,
    fpl_raw_fixtures_path,
    fpl_raw_teams_path,
    fpl_season_path,
    latest_archetype_snapshot,
    whoscored_roles_path,
)
from fpl_assistant.apps.viewmodels.player_card import latest_forecast_path


@st.cache_data(show_spinner=False)
def read_csv(path: str, version: tuple[int, int] | None = None) -> pd.DataFrame:
    del version
    try:
        return pd.read_csv(path, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_table(path: str, version: tuple[int, int] | None = None) -> pd.DataFrame:
    del version
    source = Path(path)
    try:
        return pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(
            source, low_memory=False
        )
    except (FileNotFoundError, pd.errors.EmptyDataError, ImportError, ValueError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_jsonl(path: str, version: tuple[int, int] | None = None) -> pd.DataFrame:
    del version
    try:
        return pd.read_json(path, lines=True)
    except (FileNotFoundError, ValueError):
        return pd.DataFrame()


def season_players(league: str, season: str) -> pd.DataFrame:
    path = fpl_season_path(league, season)
    return read_csv(str(path), file_version(path))


def gameweeks(league: str, season: str) -> pd.DataFrame:
    path = fpl_gameweeks_path(league, season)
    return read_csv(str(path), file_version(path))


def raw_fixtures(league: str, season: str) -> pd.DataFrame:
    path = fpl_raw_fixtures_path(league, season)
    return read_csv(str(path), file_version(path))


def raw_teams(league: str, season: str) -> pd.DataFrame:
    path = fpl_raw_teams_path(league, season)
    return read_csv(str(path), file_version(path))


def set_piece_roles(league: str, season: str) -> pd.DataFrame:
    path = whoscored_roles_path(league, season)
    return read_csv(str(path), file_version(path))


def forecast(season: str) -> tuple[pd.DataFrame, Path | None]:
    path = latest_forecast_path(PREDICTIONS_ROOT / "expected_points", season)
    if path is None:
        return pd.DataFrame(), None
    return read_table(str(path), file_version(path)), path


def archetypes(season: str) -> tuple[pd.DataFrame, Path | None]:
    snapshot = latest_archetype_snapshot(season)
    if snapshot is None:
        return pd.DataFrame(), None
    path = snapshot / "archetypes.jsonl"
    return read_jsonl(str(path), file_version(path)), path


def archetype_snapshots(season: str) -> list[Path]:
    """Return complete snapshots for a season, newest first."""
    try:
        start_year, end_year = (int(value) for value in season.split("-"))
    except (TypeError, ValueError):
        return []
    season_start = pd.Timestamp(year=start_year, month=7, day=1, tz="UTC")
    season_end = pd.Timestamp(year=end_year, month=7, day=1, tz="UTC")
    candidates: list[Path] = []
    for path in ARCHETYPE_ROOT.glob("model_version=*/snapshot=*") if ARCHETYPE_ROOT.is_dir() else ():
        artifact = path / "archetypes.jsonl"
        if not artifact.is_file():
            continue
        try:
            stamp = pd.Timestamp(
                datetime.strptime(
                    path.name.removeprefix("snapshot="), "%Y-%m-%dT%H-%M-%SZ"
                ).replace(tzinfo=timezone.utc)
            )
        except ValueError:
            continue
        if season_start <= stamp < season_end:
            candidates.append(path)
    return sorted(candidates, key=lambda item: item.name, reverse=True)


def previous_archetypes(season: str) -> tuple[pd.DataFrame, Path | None]:
    snapshots = archetype_snapshots(season)
    if len(snapshots) < 2:
        return pd.DataFrame(), None
    path = snapshots[1] / "archetypes.jsonl"
    return read_jsonl(str(path), file_version(path)), path


def player_archetype_history(season: str, player_id: object) -> pd.DataFrame:
    """Return active labels for one player across every published snapshot."""
    rows: list[pd.DataFrame] = []
    for snapshot in reversed(archetype_snapshots(season)):
        path = snapshot / "archetypes.jsonl"
        frame = read_jsonl(str(path), file_version(path))
        if frame.empty or "player_id" not in frame or "active_label" not in frame:
            continue
        active = frame.loc[
            frame["player_id"].astype(str).eq(str(player_id))
            & frame["active_label"].fillna(False).astype(bool)
        ].copy()
        if active.empty:
            continue
        active["Snapshot"] = snapshot.name.removeprefix("snapshot=")
        rows.append(active)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
