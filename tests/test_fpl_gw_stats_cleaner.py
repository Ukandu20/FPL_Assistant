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
