import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clubelo_pipeline.enrich.add_fbref_pl_matches import add_fbref_match_columns


def test_add_fbref_match_columns_matches_next_day_rows_and_skips_interleaved_non_league_rows():
    clubelo_df = pd.DataFrame(
        [
            {"from": "2015-09-13", "country": "ENG", "level": "1"},
            {"from": "2015-09-16", "country": "ENG", "level": "1"},
            {"from": "2015-09-17", "country": "ENG", "level": "1"},
            {"from": "2015-09-21", "country": "ENG", "level": "1"},
            {"from": "2015-09-24", "country": "ENG", "level": "1"},
        ]
    )
    schedule_df = pd.DataFrame(
        [
            {
                "team": "MUN",
                "season": "2015-2016",
                "league": "ENG-Premier League",
                "date": "2015-09-12",
                "game_id": "g1",
                "home": "MUN",
                "away": "LIV",
                "opponent": "LIV",
                "round": "Matchweek 5",
                "result": "W",
                "is_home": "1",
            },
            {
                "team": "MUN",
                "season": "2015-2016",
                "league": "ENG-Premier League",
                "date": "2015-09-20",
                "game_id": "g2",
                "home": "SOU",
                "away": "MUN",
                "opponent": "SOU",
                "round": "Matchweek 6",
                "result": "L",
                "is_home": "0",
            },
        ]
    )

    out = add_fbref_match_columns(clubelo_df, schedule_df, team_code="MUN")

    assert out.loc[0, "fbref_match_id"] == "g1"
    assert out.loc[0, "fbref_match_date"] == "2015-09-12"
    assert out.loc[0, "fbref_match_home"] == "MUN"
    assert out.loc[0, "fbref_match_away"] == "LIV"
    assert out.loc[0, "fbref_match_delta_days"] == "1"

    assert out.loc[3, "fbref_match_id"] == "g2"
    assert out.loc[3, "fbref_match_date"] == "2015-09-20"
    assert out.loc[3, "fbref_match_opponent"] == "SOU"
    assert out.loc[3, "fbref_match_is_home"] == "0"
    assert out.loc[3, "fbref_match_delta_days"] == "1"

    assert out.loc[1, "fbref_match_id"] == ""
    assert out.loc[2, "fbref_match_id"] == ""
    assert out.loc[4, "fbref_match_id"] == ""


def test_add_fbref_match_columns_prefers_same_day_rows_and_replaces_existing_columns():
    clubelo_df = pd.DataFrame(
        [
            {
                "from": "2015-10-04",
                "country": "ENG",
                "level": "1",
                "fbref_match_id": "old",
            },
            {"from": "2015-10-05", "country": "ENG", "level": "1"},
        ]
    )
    schedule_df = pd.DataFrame(
        [
            {
                "team": "MUN",
                "season": "2015-2016",
                "league": "ENG-Premier League",
                "date": "2015-10-04",
                "game_id": "g3",
                "home": "ARS",
                "away": "MUN",
                "round": "Matchweek 8",
                "result": "L",
            }
        ]
    )

    out = add_fbref_match_columns(clubelo_df, schedule_df, team_code="MUN")

    assert out.loc[0, "fbref_match_id"] == "g3"
    assert out.loc[0, "fbref_match_date"] == "2015-10-04"
    assert out.loc[0, "fbref_match_opponent"] == "ARS"
    assert out.loc[0, "fbref_match_is_home"] == "0"
    assert out.loc[0, "fbref_match_delta_days"] == "0"
    assert out.loc[1, "fbref_match_id"] == ""
