from __future__ import annotations

import json

import pandas as pd

from fpl_assistant.providers.fpl.pipelines.clean_and_enrich import (
    attach_fpl_context,
    enrich_season,
    reset_preseason_carryover,
)
from fpl_assistant.providers.fpl.pipelines.prices_from_merged import process_season
from fpl_assistant.providers.fpl.paths import league_scoped_root
from fpl_assistant.testing.paths import get_test_run_dir


TEST_ROOT = get_test_run_dir("fpl_preseason_pipeline")


def _case_dir(name: str):
    path = TEST_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_attach_fpl_context_restores_official_ids_teams_and_positions():
    tmp_path = _case_dir("context")
    season_dir = tmp_path / "2026-2027"
    (season_dir / "season").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "first_name": "Ada",
                "second_name": "Example",
                "id": 17,
                "team": 7,
                "element_type": 3,
                "web_name": "Ada",
                "code": 12345,
                "opta_code": "p12345",
            }
        ]
    ).to_csv(season_dir / "players_raw.csv", index=False)
    pd.DataFrame(
        [{"id": 7, "short_name": "COV"}]
    ).to_csv(season_dir / "season" / "teams.csv", index=False)

    result = attach_fpl_context(
        pd.DataFrame([{"first_name": "Ada", "second_name": "Example"}]),
        season_dir,
    )

    row = result.iloc[0]
    assert row["fpl_element_id"] == 17
    assert row["fpl_team"] == "COV"
    assert row["fpl_element_type"] == 3
    assert row["fpl_code"] == 12345


def test_enrichment_generates_unique_preseason_ids_and_team_fallback():
    tmp_path = _case_dir("enrichment")
    raw_root = tmp_path / "raw"
    proc_root = tmp_path / "processed"
    season_dir = raw_root / "2026-2027"
    (season_dir / "season").mkdir(parents=True, exist_ok=True)

    cleaned = pd.DataFrame(
        [
            {
                "first_name": "João Pedro",
                "second_name": "Known",
                "now_cost": 75,
                "element_type": "FWD",
            },
            {
                "first_name": "João Pedro",
                "second_name": "New",
                "now_cost": 50,
                "element_type": "MID",
            },
        ]
    )
    cleaned.to_csv(season_dir / "season" / "cleaned_players.csv", index=False)
    pd.DataFrame(
        [
            {
                "first_name": "João Pedro",
                "second_name": "Known",
                "id": 1,
                "team": 1,
                "element_type": 4,
                "web_name": "João Pedro",
                "code": 1001,
                "opta_code": "p1001",
            },
            {
                "first_name": "João Pedro",
                "second_name": "New",
                "id": 2,
                "team": 2,
                "element_type": 3,
                "web_name": "Costinha",
                "code": 1002,
                "opta_code": "p1002",
            },
        ]
    ).to_csv(season_dir / "players_raw.csv", index=False)
    pd.DataFrame(
        [
            {"id": 1, "short_name": "CHE"},
            {"id": 2, "short_name": "COV"},
        ]
    ).to_csv(season_dir / "season" / "teams.csv", index=False)

    enrich_season(
        season_dir=season_dir,
        out_root=proc_root,
        pid2rec={"known123": {"name": "João Pedro", "career": {}}},
        key2pid={"joao pedro": "known123"},
        overrides={"joao pedro known": "known123"},
        team_ids={"CHE": "team-che"},
        generate_missing_ids=True,
        threshold=85,
        fail_if_unmatched_pct=0,
    )

    result = pd.read_csv(
        proc_root / "2026-2027" / "season" / "cleaned_players.csv"
    )
    assert result["player_id"].notna().all()
    assert result["player_id"].is_unique
    assert result["team_id"].notna().all()
    assert result["fpl_pos"].tolist() == ["FWD", "MID"]
    assert result.loc[result["team"] == "COV", "team_id_source"].item() == (
        "generated_from_fpl_code"
    )


def test_price_export_uses_preseason_roster_as_opening_gw1():
    tmp_path = _case_dir("prices")
    season_dir = tmp_path / "fpl" / "2026-2027"
    (season_dir / "season").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "player_id": "player-a",
                "now_cost": 155,
                "team_id": "team-a",
                "fpl_pos": "FWD",
            },
            {
                "player_id": "player-b",
                "now_cost": 40,
                "team_id": "team-b",
                "fpl_pos": "GKP",
            },
        ]
    ).to_csv(season_dir / "season" / "cleaned_players.csv", index=False)

    json_dir = tmp_path / "prices"
    parquet_dir = tmp_path / "prices_parquet"
    process_season(season_dir, json_dir, parquet_dir)

    registry = json.loads(
        (json_dir / "2026-2027.json").read_text(encoding="utf-8")
    )
    assert registry == {
        "player-a": {"1": 15.5},
        "player-b": {"1": 4.0},
    }
    assert (
        (parquet_dir / "2026-2027.parquet").is_file()
        or (parquet_dir / "2026-2027.csv").is_file()
    )


def test_league_scoped_root_accepts_provider_or_scoped_root():
    provider_root = TEST_ROOT / "paths" / "fpl"
    scoped_root = provider_root / "ENG-Premier League"

    assert league_scoped_root(provider_root) == scoped_root
    assert league_scoped_root(scoped_root) == scoped_root


def test_preseason_carryover_is_reset_but_roster_fields_are_retained():
    tmp_path = _case_dir("carryover")
    season_dir = tmp_path / "2026-2027"
    (season_dir / "season").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"started": False, "finished": False, "kickoff_time": "2026-08-21T19:00:00Z"}]
    ).to_csv(season_dir / "season" / "fixtures.csv", index=False)
    roster = pd.DataFrame(
        [{
            "name": "Example Player",
            "team": "ARS",
            "now_cost": 75,
            "selected_by_percent": 12.3,
            "minutes": 2500,
            "total_points": 180,
            "goals_scored": 12,
        }]
    )

    result, reset_columns = reset_preseason_carryover(roster, season_dir)

    assert set(reset_columns) == {"goals_scored", "minutes", "total_points"}
    assert result.loc[0, "minutes"] == 0
    assert result.loc[0, "total_points"] == 0
    assert result.loc[0, "now_cost"] == 75
    assert result.loc[0, "selected_by_percent"] == 12.3
    assert result.loc[0, "performance_data_status"] == "prior_season_carryover_reset"


def test_started_season_totals_are_not_reset():
    tmp_path = _case_dir("started")
    season_dir = tmp_path / "2026-2027"
    (season_dir / "season").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"started": True, "finished": False}]).to_csv(
        season_dir / "season" / "fixtures.csv", index=False
    )
    roster = pd.DataFrame([{"minutes": 90, "total_points": 8}])

    result, reset_columns = reset_preseason_carryover(roster, season_dir)

    assert reset_columns == []
    assert result.loc[0, "total_points"] == 8
