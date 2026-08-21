from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import pytest

from fpl_assistant.providers.fpl.clean.assign_game_ids import (
    attach_fixture_calendar_game_ids,
    process_season,
)


def _case_dir(name: str) -> Path:
    path = Path(".tmp") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_fixture_calendar_fills_game_ids_from_official_fpl_fixture():
    tmp_path = _case_dir("assign_ids_fixture")
    calendar = tmp_path / "fixture_calendar.csv"
    pd.DataFrame(
        [
            {"fpl_id": 10, "fbref_id": "match-a", "team_id": "ars"},
            {"fpl_id": 10, "fbref_id": "match-a", "team_id": "che"},
            {"fpl_id": 11, "fbref_id": "match-b", "team_id": "liv"},
            {"fpl_id": 11, "fbref_id": "match-b", "team_id": "mun"},
        ]
    ).to_csv(calendar, index=False)
    frame = pd.DataFrame(
        [
            {"fixture": 10, "game_id": pd.NA},
            {"fixture": "11", "game_id": pd.NA},
            {"fixture": 12, "game_id": "existing"},
        ]
    )

    result = attach_fixture_calendar_game_ids(frame, calendar)

    assert result["game_id"].tolist() == ["match-a", "match-b", "existing"]
    assert result["match_id"].tolist() == ["match-a", "match-b", "existing"]


def test_fixture_calendar_replaces_stale_game_id_with_fbref_truth():
    tmp_path = _case_dir("assign_ids_replace")
    calendar = tmp_path / "fixture_calendar.csv"
    pd.DataFrame([{"fpl_id": 10, "fbref_id": "fbref-match"}]).to_csv(
        calendar, index=False
    )

    result = attach_fixture_calendar_game_ids(
        pd.DataFrame([{"fixture": 10, "game_id": "stale-match"}]), calendar
    )

    assert result.loc[0, "game_id"] == "fbref-match"
    assert result.loc[0, "match_id"] == "fbref-match"


def test_fixture_calendar_rejects_conflicting_game_ids():
    tmp_path = _case_dir("assign_ids_conflict")
    calendar = tmp_path / "fixture_calendar.csv"
    pd.DataFrame(
        [
            {"fpl_id": 10, "fbref_id": "match-a"},
            {"fpl_id": 10, "fbref_id": "match-b"},
        ]
    ).to_csv(calendar, index=False)

    with pytest.raises(ValueError, match="multiple canonical game IDs"):
        attach_fixture_calendar_game_ids(
            pd.DataFrame([{"fixture": 10}]),
            calendar,
        )


def test_fixture_calendar_prefers_provider_neutral_match_id():
    tmp_path = _case_dir("assign_ids_canonical")
    calendar = tmp_path / "fixture_calendar.csv"
    pd.DataFrame(
        [{"fpl_id": 10, "match_id": "canonical", "fbref_id": "legacy-alias"}]
    ).to_csv(calendar, index=False)

    result = attach_fixture_calendar_game_ids(
        pd.DataFrame([{"fixture": 10}]), calendar
    )

    assert result.loc[0, "game_id"] == "canonical"
    assert result.loc[0, "match_id"] == "canonical"


def test_process_season_uses_fixture_calendar_without_fbref_files():
    tmp_path = _case_dir("assign_ids_without_fbref")
    season_dir = tmp_path / "fpl" / "2026-2027"
    gws_dir = season_dir / "gws"
    gws_dir.mkdir(parents=True)
    row = {
        "fixture": 10,
        "round": 1,
        "was_home": True,
        "team_id": "ars-id",
        "opp_id": "che-id",
        "team_code": "ARS",
        "opp_code": "CHE",
        "kickoff_time": "2026-08-21T19:00:00Z",
    }
    pd.DataFrame([row]).to_csv(gws_dir / "merged_gws.csv", index=False)
    pd.DataFrame([row]).to_csv(gws_dir / "gw1.csv", index=False)

    fixture_root = tmp_path / "registry" / "fixtures"
    calendar_dir = fixture_root / "2026-2027"
    calendar_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"fpl_id": 10, "match_id": "canonical-match", "team_id": "ars-id"},
            {"fpl_id": 10, "match_id": "canonical-match", "team_id": "che-id"},
        ]
    ).to_csv(calendar_dir / "fixture_calendar.csv", index=False)

    process_season(
        proc_season_dir=season_dir,
        fbref_root=tmp_path / "missing-fbref",
        fixture_calendar_root=fixture_root,
        league="ENG-Premier League",
        summary_name="summary.csv",
        tz_name="UTC",
    )

    merged = pd.read_csv(gws_dir / "merged_gws.csv")
    matches = pd.read_csv(season_dir / "matches" / "matches.csv")
    assert merged.loc[0, "match_id"] == "canonical-match"
    assert matches.loc[0, "match_id"] == "canonical-match"
