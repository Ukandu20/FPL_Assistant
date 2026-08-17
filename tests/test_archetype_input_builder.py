from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_assistant.archetypes.input_builder import (
    _apply_current_roster_context,
    _deduplicate_fpl,
    _understat_shot_evidence,
    discover_joinable_seasons,
)


def test_discover_joinable_seasons_requires_all_three_providers(
    tmp_path: Path,
) -> None:
    season = "2025-2026"
    paths = [
        tmp_path / "fpl" / "ENG-Premier League" / season / "gws" / "merged_gws.csv",
        tmp_path / "understat" / "ENG-Premier League" / season / "player_match.csv",
        tmp_path / "whoscored" / "ENG-Premier League" / season / "player_match" / "summary.csv",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    incomplete = tmp_path / "fpl" / "ENG-Premier League" / "2024-2025" / "gws" / "merged_gws.csv"
    incomplete.parent.mkdir(parents=True)
    incomplete.touch()

    assert discover_joinable_seasons(processed_root=tmp_path) == [season]


def test_fpl_duplicate_resolution_excludes_cross_identity_collisions() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["m1", "m1", "m2", "m2"],
            "player_id": ["same", "same", "collision", "collision"],
            "name": ["A", "A", "B", "C"],
            "team": ["AAA", "AAA", "BBB", "CCC"],
            "minutes": [0, 90, 60, 30],
            "total_points": [0, 5, 2, 1],
        }
    )

    resolved, audit = _deduplicate_fpl(frame, "2025-2026")

    assert resolved[["game_id", "player_id", "minutes"]].to_dict("records") == [
        {"game_id": "m1", "player_id": "same", "minutes": 90}
    ]
    assert {row["audit_type"] for row in audit} == {
        "duplicate_resolved", "ambiguous_identity_excluded"
    }


def test_understat_non_penalty_evidence_excludes_penalty_shots(
    tmp_path: Path,
) -> None:
    path = tmp_path / "understat" / "EPL" / "2025" / "shot_events.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "player_id": [10, 10, 10],
            "xg": [0.2, 0.76, 0.3],
            "situation": ["Open Play", "Penalty", "Open Play"],
            "result": ["Goal", "Goal", "Missed Shot"],
        }
    ).to_csv(path, index=False)

    evidence = _understat_shot_evidence("2025-2026", raw_root=tmp_path).iloc[0]

    assert evidence["npxg"] == 0.5
    assert evidence["non_penalty_goals"] == 1


def test_current_roster_context_preserves_historical_availability_rows() -> None:
    matches = pd.DataFrame(
        {
            "player_id": ["p1"], "team_id": ["old"],
            "fpl_position": ["MID"], "historical_fpl_position": ["MID"],
            "kickoff_utc": [pd.Timestamp("2026-05-01", tz="UTC")],
            "minutes": [90], "availability_status": ["unknown"],
        }
    )
    roster = pd.DataFrame(
        {
            "player_id": ["p1"], "team_id": ["new"], "fpl_pos": ["FWD"],
            "status": ["i"], "news": ["hamstring"],
        }
    )

    result = _apply_current_roster_context(
        matches, roster, as_of=pd.Timestamp("2026-08-17", tz="UTC")
    ).iloc[0]

    assert result["availability_status"] == "unknown"
    assert result["current_availability_status"] == "injured"
    assert result["transferred"]
    assert result["position_changed"]
