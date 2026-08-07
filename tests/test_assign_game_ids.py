from __future__ import annotations

import pandas as pd
import pytest

from fpl_assistant.providers.fpl.clean.assign_game_ids import (
    attach_fixture_calendar_game_ids,
)


def test_fixture_calendar_fills_game_ids_from_official_fpl_fixture(tmp_path):
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


def test_fixture_calendar_replaces_stale_game_id_with_fbref_truth(tmp_path):
    calendar = tmp_path / "fixture_calendar.csv"
    pd.DataFrame([{"fpl_id": 10, "fbref_id": "fbref-match"}]).to_csv(
        calendar, index=False
    )

    result = attach_fixture_calendar_game_ids(
        pd.DataFrame([{"fixture": 10, "game_id": "stale-match"}]), calendar
    )

    assert result.loc[0, "game_id"] == "fbref-match"
    assert result.loc[0, "match_id"] == "fbref-match"


def test_fixture_calendar_rejects_conflicting_game_ids(tmp_path):
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
