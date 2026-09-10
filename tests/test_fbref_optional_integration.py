from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from fpl_assistant.pipelines.integrate.calendar_builder import (
    load_minutes,
    write_empty_minutes_calendar,
)
from fpl_assistant.qa.assurance import (
    validate_fixture_calendar,
    validate_player_minutes_calendar,
)


def _case_dir(name: str) -> Path:
    path = Path(".tmp") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_player_match_loader_uses_whoscored_without_fbref():
    tmp_path = _case_dir("player_calendar_without_fbref")
    season = "2026-2027"
    player_match = tmp_path / "whoscored" / season / "player_match"
    player_match.mkdir(parents=True)
    key = {"match_id": "match-1", "player_id": "player-1", "team_id": "team-1"}

    pd.DataFrame(
        [{
            **key,
            "player": "Example Player",
            "minutes": 90,
            "yellow_cards": 0,
            "red_cards": 0,
            "fpl_pos": "MID",
            "shots_total": 2,
            "shots_on_target": 1,
        }]
    ).to_csv(player_match / "summary.csv", index=False)
    pd.DataFrame(
        [{**key, "shots_on_target_against": 0, "saves": 0, "save_pct": 0}]
    ).to_csv(player_match / "keepers.csv", index=False)
    pd.DataFrame(
        [{
            **key,
            "blocks": 1,
            "tackles_won": 2,
            "interceptions": 1,
            "clearances": 0,
        }]
    ).to_csv(player_match / "defense.csv", index=False)
    pd.DataFrame(
        [{**key, "recoveries": 4, "penalties_won": 0, "own_goals": 0}]
    ).to_csv(player_match / "misc.csv", index=False)

    result = load_minutes(
        tmp_path / "fixtures" / season,
        tmp_path / "fbref-does-not-exist",
        tmp_path / "whoscored",
    )

    assert result.loc[0, "match_id"] == "match-1"
    assert result.loc[0, "fbref_id"] == "match-1"
    assert result.loc[0, "minutes"] == 90


def test_provider_calendar_builder_cannot_overwrite_modelling_registry():
    tmp_path = _case_dir("observed_calendar_ownership")
    season_dir = tmp_path / "2026-2027"
    season_dir.mkdir()
    modelling = season_dir / "player_fixture_calendar.csv"
    modelling.write_text("sentinel\nkeep\n", encoding="utf-8")

    write_empty_minutes_calendar(season_dir, include_price=False)

    assert modelling.read_text(encoding="utf-8") == "sentinel\nkeep\n"
    assert (season_dir / "player_fixture_calendar_observed.csv").is_file()


def test_assurance_accepts_provider_neutral_calendars_without_price():
    tmp_path = _case_dir("assurance_without_fbref")
    season_dir = tmp_path / "2026-2027"
    season_dir.mkdir()
    fixture_rows = []
    for is_home, team_id, opponent_id, team in (
        (1, "ars-id", "che-id", "ARS"),
        (0, "che-id", "ars-id", "CHE"),
    ):
        fixture_rows.append(
            {
                "fpl_id": 10,
                "match_id": "match-1",
                "gw_orig": 1,
                "date_sched": "2026-08-21",
                "date_played": "2026-08-21",
                "team": team,
                "team_id": team_id,
                "opponent_id": opponent_id,
                "is_home": is_home,
                "home": "ARS",
                "away": "CHE",
                "home_id": "ars-id",
                "away_id": "che-id",
                "status": "finished",
                "sched_missing": 0,
                "venue": "Example Stadium",
            }
        )
    pd.DataFrame(fixture_rows).to_csv(season_dir / "fixture_calendar.csv", index=False)

    pd.DataFrame(
        [{
            "match_id": "match-1",
            "fpl_id": 10,
            "gw_orig": 1,
            "date_played": "2026-08-21",
            "team_id": "ars-id",
            "team": "ARS",
            "venue": "Example Stadium",
            "was_home": 1,
            "player_id": "player-1",
            "player": "Example Player",
            "pos": "MID",
            "minutes": 90,
            "days_since_last": 0,
            "is_active": 1,
            "is_starter": 1,
            "starter_source": "fpl",
            "xp": 2.0,
            "total_points": 2,
            "bonus": 0,
            "bps": 5,
            "clean_sheets": 0,
            "team_gf": 1,
            "team_ga": 0,
            "fdr_home": 3,
            "fdr_away": 3,
        }]
    ).to_csv(season_dir / "player_fixture_calendar.csv", index=False)

    fixture = validate_fixture_calendar(season_dir)
    players = validate_player_minutes_calendar(season_dir, 0.99, 0.98, -50, 200)

    assert fixture.loc[0, "match_id"] == "match-1"
    assert players.loc[0, "match_id"] == "match-1"
