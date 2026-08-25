from __future__ import annotations

import pandas as pd

from fpl_assistant.providers.fpl.clean.gw_stats_cleaner import clean_gw_df


def test_clean_gw_df_removes_only_exact_source_duplicates():
    source = pd.DataFrame(
        [
            {
                "name": "Example Player",
                "team": "Arsenal",
                "opponent_team": 2,
                "was_home": True,
                "round": 1,
                "fixture": 10,
                "minutes": 90,
            },
            {
                "name": "Example Player",
                "team": "Arsenal",
                "opponent_team": 2,
                "was_home": True,
                "round": 1,
                "fixture": 10,
                "minutes": 90,
            },
            {
                "name": "Example Player",
                "team": "Arsenal",
                "opponent_team": 2,
                "was_home": True,
                "round": 1,
                "fixture": 11,
                "minutes": 45,
            },
        ]
    )

    cleaned, unmatched, unmatched_rows = clean_gw_df(
        source,
        "2025-26",
        {},
        {},
        {},
        {2: "Chelsea"},
        {2: "CHE"},
        {2: "chelsea"},
        {"arsenal": "ARS", "chelsea": "CHE"},
        {"arsenal": "arsenal", "chelsea": "chelsea"},
        {"ARS": "arsenal", "CHE": "chelsea"},
        1,
    )

    assert len(cleaned) == 2
    assert cleaned["fixture"].tolist() == [10, 11]
    assert len(unmatched) == 2
    assert len(unmatched_rows) == 2


def test_official_gw_position_beats_a_later_registry_position():
    source = pd.DataFrame(
        [{
            "name": "Example Player",
            "position": "MID",
            "team": "Arsenal",
            "opponent_team": 2,
            "was_home": True,
            "round": 1,
        }]
    )
    master = {
        "player-1": {
            "name": "Example Player",
            "career": {
                "2025-2026": {"fpl_position": "MID"},
                "2026-2027": {"fpl_position": "FWD"},
            },
        }
    }

    cleaned, _, _ = clean_gw_df(
        source,
        "2025-26",
        master,
        {"example player": "player-1"},
        {},
        {2: "Chelsea"},
        {2: "CHE"},
        {2: "chelsea"},
        {"arsenal": "ARS", "chelsea": "CHE"},
        {"arsenal": "arsenal", "chelsea": "chelsea"},
        {"ARS": "arsenal", "CHE": "chelsea"},
        1,
    )

    assert cleaned.loc[0, "fpl_pos"] == "MID"


def test_official_element_roster_fills_missing_canonical_team_id():
    source = pd.DataFrame(
        [{
            "element": 607,
            "name": "Luka Lynch",
            "team": "Coventry",
            "opponent_team": 1,
            "was_home": False,
            "round": 1,
            "fixture": 1,
        }]
    )
    element_roster = {
        607: {
            "player_id": "40e4fe8b",
            "name": "Luka Lynch",
            "team": "COV",
            "team_id": "ce4d980327c7",
            "fpl_pos": "MID",
        }
    }

    cleaned, unmatched, _ = clean_gw_df(
        source,
        "2026-2027",
        {},
        {},
        {},
        {1: "Arsenal"},
        {1: "ARS"},
        {1: "1dd1f33c"},
        {"coventry": "COV", "arsenal": "ARS"},
        {"arsenal": "1dd1f33c"},
        {"ARS": "1dd1f33c"},
        1,
        element_roster,
    )

    assert unmatched == []
    assert cleaned.loc[0, "team_code"] == "COV"
    assert cleaned.loc[0, "team_id"] == "ce4d980327c7"
    assert cleaned.loc[0, "away_id"] == "ce4d980327c7"
