from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fpl_assistant.archetypes.clean_sheets import score_clean_sheet_specialist
from fpl_assistant.archetypes.persistence import persist_snapshot
from fpl_assistant.archetypes.pipeline import build_archetype_snapshot
from tests.test_archetype_scoring import _observations


def test_clean_sheet_specialist_uses_sixty_minute_opportunities() -> None:
    rows: list[dict[str, object]] = []
    for player_index in range(5):
        for appearance in range(10):
            rows.append(
                {
                    "player_id": f"d{player_index}",
                    "kickoff_utc": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=7 * appearance),
                    "fpl_position": "DEF", "minutes": 90, "started": True,
                    "clean_sheet": appearance < (player_index * 2),
                    "goals_conceded": max(0, 4 - player_index),
                    "xga": 1.5, "team_defence_index_pre_match": 0.8 + player_index * 0.1,
                }
            )
    result = score_clean_sheet_specialist(
        pd.DataFrame(rows), as_of="2026-05-01T00:00:00Z"
    ).set_index("player_id")
    assert result.at["d4", "score_0_100"] == 100
    assert result.at["d4", "active_label"]
    assert result.at["d4", "eligible_appearances"] == 10


def test_pipeline_emits_exactly_one_position_composite() -> None:
    result = build_archetype_snapshot(
        _observations(),
        as_of="2026-05-01T00:00:00Z",
        current_season="2025-2026",
    )
    composites = result.archetypes[result.archetypes["family"].eq("Production Composite")]
    assert composites.groupby("player_id").size().le(1).all()
    high = composites[composites["player_id"].eq("high")].iloc[0]
    assert high["display_name"] == "Complete Midfielder"
    assert "SWEEPER" not in result.archetypes["archetype_id"].tolist()
    assert "DISTRIBUTOR" not in result.archetypes["archetype_id"].tolist()


def test_versioned_snapshot_is_deterministic_and_immutable(tmp_path: Path) -> None:
    result = build_archetype_snapshot(
        _observations(),
        as_of="2026-05-01T00:00:00Z",
        current_season="2025-2026",
    )
    target = persist_snapshot(
        result.archetypes, result.team_ratings,
        output_root=tmp_path, snapshot_date="2026-05-01T00:00:00Z",
        model_version="1.0.0",
        evidence_tables=result.evidence_tables,
    )
    assert (target / "archetypes.jsonl").is_file()
    assert (target / "team_ratings.jsonl").is_file()
    assert (target / "manifest.json").is_file()
    assert (target / "player_match_evidence.jsonl").is_file()
    assert (target / "production_component_evidence.jsonl").is_file()
    assert set(result.evidence_tables) == {
        "player_match_evidence",
        "production_component_evidence",
        "family_calculation_evidence",
        "team_match_evidence",
        "player_value_evidence",
    }
    manifest = pd.read_json(target / "manifest.json", typ="series")
    assert "production_component_evidence.jsonl" in manifest["artifacts"]
    assert persist_snapshot(
        result.archetypes, result.team_ratings,
        output_root=tmp_path, snapshot_date="2026-05-01T00:00:00Z",
        model_version="1.0.0",
        evidence_tables=result.evidence_tables,
    ) == target

    numerically_equivalent_evidence = {
        name: frame.copy() for name, frame in result.evidence_tables.items()
    }
    production_evidence = numerically_equivalent_evidence[
        "production_component_evidence"
    ]
    production_evidence.loc[
        production_evidence.index[0], "temporal_combined_raw"
    ] += 1e-14
    assert persist_snapshot(
        result.archetypes,
        result.team_ratings,
        output_root=tmp_path,
        snapshot_date="2026-05-01T00:00:00Z",
        model_version="1.0.0",
        evidence_tables=numerically_equivalent_evidence,
    ) == target

    changed = result.archetypes.copy()
    changed.loc[0, "score_0_100"] = 12.3
    with pytest.raises(FileExistsError, match="would be overwritten"):
        persist_snapshot(
            changed, result.team_ratings,
            output_root=tmp_path, snapshot_date="2026-05-01T00:00:00Z",
            model_version="1.0.0",
            evidence_tables=result.evidence_tables,
        )


def test_pipeline_adds_own_and_opponent_pre_match_context_to_evidence() -> None:
    observations = _observations()
    observations["opponent_id"] = "opponent"
    team_rows: list[dict[str, object]] = []
    for row in observations[["match_id", "team_id", "kickoff_utc"]].drop_duplicates().itertuples(index=False):
        team_rows.append(
            {
                "match_id": row.match_id, "season": "2025-2026",
                "kickoff_utc": row.kickoff_utc, "home_team_id": row.team_id,
                "away_team_id": "opponent", "home_goals": 1, "away_goals": 0,
                "home_xg": 1.2, "away_xg": 0.7,
            }
        )

    result = build_archetype_snapshot(
        observations,
        team_matches=pd.DataFrame(team_rows),
        as_of="2026-05-01T00:00:00Z",
        current_season="2025-2026",
    )
    evidence = result.evidence_tables["player_match_evidence"]

    assert evidence["team_attack"].notna().all()
    assert evidence["opponent_defence"].notna().all()
    assert evidence["opponent_elo_pre_match"].notna().all()
