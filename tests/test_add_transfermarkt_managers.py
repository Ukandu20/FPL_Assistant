import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clubelo_pipeline.enrich.add_transfermarkt_managers import add_manager_columns


def test_add_manager_columns_assigns_current_and_historical_managers():
    clubelo_df = pd.DataFrame(
        [
            {"from": "2024-10-28"},
            {"from": "2024-11-11"},
            {"from": "2026-01-13"},
            {"from": "1980-01-01"},
        ]
    )
    managers_df = pd.DataFrame(
        [
            {
                "name": "Rúben Amorim",
                "nationality": "Portugal",
                "appointment_date": pd.Timestamp("2024-11-11"),
                "end_date": pd.Timestamp("2026-01-05"),
            },
            {
                "name": "Ruud van Nistelrooy",
                "nationality": "Netherlands",
                "appointment_date": pd.Timestamp("2024-10-28"),
                "end_date": pd.Timestamp("2024-11-11"),
            },
            {
                "name": "Michael Carrick",
                "nationality": "England",
                "appointment_date": pd.Timestamp("2026-01-13"),
                "end_date": pd.NaT,
            },
        ]
    )

    out = add_manager_columns(clubelo_df, managers_df)

    assert out.loc[0, "manager"] == "Ruud van Nistelrooy"
    assert out.loc[1, "manager"] == "Rúben Amorim"
    assert out.loc[2, "manager"] == "Michael Carrick"
    assert out.loc[2, "manager_nationality"] == "England"
    assert out.loc[3, "manager"] == ""


def test_add_manager_columns_replaces_existing_manager_columns():
    clubelo_df = pd.DataFrame(
        [{"from": "2024-11-11", "manager": "Old Name", "manager_nationality": "Old Nat"}]
    )
    managers_df = pd.DataFrame(
        [
            {
                "name": "Rúben Amorim",
                "nationality": "Portugal",
                "appointment_date": pd.Timestamp("2024-11-11"),
                "end_date": pd.Timestamp("2026-01-05"),
            }
        ]
    )

    out = add_manager_columns(clubelo_df, managers_df)

    assert out.loc[0, "manager"] == "Rúben Amorim"
    assert out.loc[0, "manager_nationality"] == "Portugal"
