import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clubelo_pipeline.enrich.add_pl_season_bands import add_season_band_columns


def test_add_season_band_columns_bands_only_eng_level_1_rows():
    seasons_df = pd.DataFrame(
        [
            {
                "season": "1992-93",
                "competition": "Premier League",
                "start_date": pd.Timestamp("1992-08-15"),
                "finish_date": pd.Timestamp("1993-05-11"),
            }
        ]
    )
    df = pd.DataFrame(
        [
            {"from": "1992-08-15", "country": "ENG", "level": "1", "team": "Man United"},
            {"from": "1992-08-15", "country": "ENG", "level": "2", "team": "Leeds"},
            {"from": "1992-08-15", "country": "ESP", "level": "1", "team": "Barcelona"},
            {"from": "1992-07-01", "country": "ENG", "level": "1", "team": "Arsenal"},
        ]
    )

    out = add_season_band_columns(df, seasons_df, country_filter="ENG", level_filter="1")

    assert out.loc[0, "season"] == "1992-93"
    assert out.loc[0, "league"] == "ENG-Premier League"
    assert out.loc[1, "season"] == ""
    assert out.loc[1, "league"] == ""
    assert out.loc[2, "season"] == ""
    assert out.loc[3, "season"] == ""


def test_add_season_band_columns_uses_latest_matching_start_date():
    seasons_df = pd.DataFrame(
        [
            {
                "season": "1992-93",
                "competition": "Premier League",
                "start_date": pd.Timestamp("1992-08-15"),
                "finish_date": pd.Timestamp("1993-05-11"),
            },
            {
                "season": "1993-94",
                "competition": "Premier League",
                "start_date": pd.Timestamp("1993-08-14"),
                "finish_date": pd.Timestamp("1994-05-08"),
            },
        ]
    )
    df = pd.DataFrame(
        [
            {"from": "1993-05-01", "country": "ENG", "level": "1"},
            {"from": "1993-08-14", "country": "ENG", "level": "1"},
        ]
    )

    out = add_season_band_columns(df, seasons_df, country_filter="ENG", level_filter="1")

    assert out.loc[0, "season"] == "1992-93"
    assert out.loc[0, "league"] == "ENG-Premier League"
    assert out.loc[1, "season"] == "1993-94"
    assert out.loc[1, "league"] == "ENG-Premier League"
