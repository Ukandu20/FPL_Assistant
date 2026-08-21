from __future__ import annotations

import json

import pandas as pd
import pytest

from fpl_assistant.providers.fpl.pipelines.clean_and_enrich import (
    attach_fpl_context,
    enrich_season,
    load_fpl_code_registry,
    publish_fpl_roster_registry,
    register_generated_players,
    reset_preseason_carryover,
)
from fpl_assistant.canonical.identity import stable_canonical_id
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
    generated = result[result["player_id_source"].eq("generated_from_fpl_code_duplicate")]
    assert generated["player_id"].str.len().eq(8).all()
    assert generated["player_id"].item() == stable_canonical_id(
        "player", "fpl", "1002", length=8
    )


def test_canonical_name_identity_wins_over_historical_generated_fallback():
    tmp_path = _case_dir("historical_generated_reconciliation")
    raw_root = tmp_path / "raw"
    proc_root = tmp_path / "processed"
    season_dir = raw_root / "2026-2027"
    (season_dir / "season").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"first_name": "Kostas", "second_name": "Tsimikas"}]
    ).to_csv(season_dir / "season" / "cleaned_players.csv", index=False)
    pd.DataFrame(
        [
            {
                "first_name": "Kostas",
                "second_name": "Tsimikas",
                "id": 1,
                "team": 1,
                "element_type": 2,
                "web_name": "Tsimikas",
                "code": 12345,
            }
        ]
    ).to_csv(season_dir / "players_raw.csv", index=False)
    pd.DataFrame([{"id": 1, "short_name": "LIV"}]).to_csv(
        season_dir / "season" / "teams.csv", index=False
    )

    enrich_season(
        season_dir=season_dir,
        out_root=proc_root,
        pid2rec={"6285d4dc": {"name": "Kostas Tsimikas", "career": {}}},
        key2pid={"kostas tsimikas": "6285d4dc"},
        overrides={},
        team_ids={"LIV": "team-liv"},
        generate_missing_ids=True,
        threshold=85,
        fail_if_unmatched_pct=0,
        fpl_code_to_pid={"12345": "historical12"},
        historically_generated_codes={"12345"},
    )

    result = pd.read_csv(
        proc_root / "2026-2027" / "season" / "cleaned_players.csv"
    )
    assert result.loc[0, "player_id"] == "6285d4dc"
    assert result.loc[0, "player_id_source"] == "master_or_override"


def test_generated_players_are_promoted_to_all_player_registries():
    tmp_path = _case_dir("registry_promotion")
    registry = tmp_path / "registry"
    bridge_dir = registry / "bridges"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    compatibility_master = tmp_path / "master_fpl_players.json"
    (registry / "master_players.json").write_text(
        json.dumps({"known": {"name": "Known Player", "career": {}}}),
        encoding="utf-8",
    )
    (registry / "_id_lookup_players.json").write_text(
        json.dumps({"known player": "known"}), encoding="utf-8"
    )
    (registry / "master_fpl.json").write_text(
        json.dumps({"known": {"player_id": "known", "name": "Known Player", "career": {}}}),
        encoding="utf-8",
    )
    compatibility_master.write_text("{}", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "entity_type": "player",
                "provider": "whoscored",
                "provider_id": "1",
                "provider_name": "Known Player",
                "canonical_id": "known",
                "valid_from": "",
                "valid_to": "",
                "match_method": "exact",
                "match_confidence": 1.0,
                "review_status": "approved",
            }
        ]
    ).to_csv(bridge_dir / "player_ids.csv", index=False)
    players = pd.DataFrame(
        [
            {
                "player_id": "abc12345",
                "name": "New Player",
                "first_name": "New",
                "second_name": "Player",
                "fpl_code": "98765",
                "fpl_element_id": 55,
                "team": "COV",
                "team_id": "team-cov",
                "fpl_pos": "MID",
            }
        ]
    )

    audit = register_generated_players(
        players,
        pd.Series([True]),
        season="2026-2027",
        league="ENG-Premier League",
        registry_root=registry,
        compatibility_master_path=compatibility_master,
    )
    # Registration is idempotent and must not duplicate the provider bridge.
    register_generated_players(
        players,
        pd.Series([True]),
        season="2026-2027",
        league="ENG-Premier League",
        registry_root=registry,
        compatibility_master_path=compatibility_master,
    )

    master = json.loads((registry / "master_players.json").read_text(encoding="utf-8"))
    lookup = json.loads((registry / "_id_lookup_players.json").read_text(encoding="utf-8"))
    master_fpl = json.loads((registry / "master_fpl.json").read_text(encoding="utf-8"))
    compatibility = json.loads(compatibility_master.read_text(encoding="utf-8"))
    bridges = pd.read_csv(bridge_dir / "player_ids.csv", dtype=str)
    fpl_bridge = bridges[bridges["provider"].eq("fpl")]

    assert audit.loc[0, "registry_status"] == "registered"
    assert master["abc12345"]["career"]["2026-2027"]["team_id"] == "team-cov"
    assert lookup["new player"] == "abc12345"
    assert master_fpl["abc12345"]["career"]["2026-27"]["fpl_position"] == "MID"
    assert compatibility["abc12345"]["player_id"] == "abc12345"
    assert len(fpl_bridge) == 1
    assert fpl_bridge.iloc[0]["provider_id"] == "98765"
    assert fpl_bridge.iloc[0]["canonical_id"] == "abc12345"


def test_fpl_roster_publication_upserts_team_and_player_membership():
    tmp_path = _case_dir("fpl_roster_registry")
    registry = tmp_path / "registry"
    registry.mkdir(parents=True, exist_ok=True)
    (registry / "master_players.json").write_text(
        json.dumps(
            {
                "player-a": {
                    "name": "Canonical A",
                    "career": {"2025-2026": {"team": "OLD", "team_id": "old-team"}},
                }
            }
        ),
        encoding="utf-8",
    )
    (registry / "master_teams.json").write_text(
        json.dumps(
            {
                "team-a": {
                    "name": "ARS",
                    "career": {"2025-2026": {"league": "ENG-Premier League", "players": []}},
                }
            }
        ),
        encoding="utf-8",
    )
    (registry / "master_fpl.json").write_text("{}", encoding="utf-8")
    (registry / "_id_lookup_players.json").write_text("{}", encoding="utf-8")
    (registry / "_id_lookup_teams.json").write_text(
        json.dumps({"ars": "team-a"}), encoding="utf-8"
    )
    roster = pd.DataFrame(
        [
            {
                "player_id": "player-a",
                "name": "FPL Name A",
                "team": "ARS",
                "team_id": "team-a",
                "fpl_pos": "MID",
            },
            {
                "player_id": "player-b",
                "name": "Player B",
                "team": "COV",
                "team_id": "team-cov",
                "fpl_pos": "FWD",
            },
        ]
    )

    audit = publish_fpl_roster_registry(
        roster,
        season="2026-2027",
        league="ENG-Premier League",
        registry_root=registry,
    )
    publish_fpl_roster_registry(
        roster,
        season="2026-2027",
        league="ENG-Premier League",
        registry_root=registry,
    )

    master_players = json.loads(
        (registry / "master_players.json").read_text(encoding="utf-8")
    )
    master_teams = json.loads(
        (registry / "master_teams.json").read_text(encoding="utf-8")
    )
    master_fpl = json.loads((registry / "master_fpl.json").read_text(encoding="utf-8"))
    team_lookup = json.loads(
        (registry / "_id_lookup_teams.json").read_text(encoding="utf-8")
    )

    assert audit["players_published"] == 2
    assert audit["teams_published"] == 2
    assert master_players["player-a"]["name"] == "Canonical A"
    assert "2025-2026" in master_players["player-a"]["career"]
    assert master_players["player-a"]["career"]["2026-2027"]["team_id"] == "team-a"
    assert master_players["player-b"]["career"]["2026-2027"]["fpl_position"] == "FWD"
    assert master_fpl["player-b"]["career"]["2026-27"]["team"] == "COV"
    assert master_teams["team-a"]["career"]["2026-2027"]["players"] == [
        {"id": "player-a", "name": "Canonical A"}
    ]
    assert master_teams["team-cov"]["career"]["2026-2027"]["players"] == [
        {"id": "player-b", "name": "Player B"}
    ]
    assert team_lookup["cov"] == "team-cov"


def test_fpl_code_registry_reuses_prior_season_id_but_excludes_target_output():
    tmp_path = _case_dir("fpl_code_history")
    registry = tmp_path / "registry"
    (registry / "bridges").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["provider", "provider_id", "canonical_id", "match_method"]).to_csv(
        registry / "bridges" / "player_ids.csv", index=False
    )
    processed = tmp_path / "processed"
    for season, player_id in (("2025-2026", "legacy12char"), ("2026-2027", "wrong12char")):
        path = processed / season / "season"
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "fpl_code": "519440",
                    "player_id": player_id,
                    "player_id_source": "generated_from_fpl_code",
                }
            ]
        ).to_csv(path / "cleaned_players.csv", index=False)

    mapping, generated = load_fpl_code_registry(
        registry, processed, target_season="2026-2027"
    )

    assert mapping["519440"] == "legacy12char"
    assert "519440" in generated


def test_fpl_code_registry_prefers_later_corrected_historical_mapping():
    tmp_path = _case_dir("fpl_code_history_correction")
    registry = tmp_path / "registry"
    (registry / "bridges").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["provider", "provider_id", "canonical_id", "match_method"]).to_csv(
        registry / "bridges" / "player_ids.csv", index=False
    )
    processed = tmp_path / "processed"
    for season, player_id in (("2023-2024", "wrong001"), ("2025-2026", "right001")):
        path = processed / season / "season"
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{
                "fpl_code": "628204",
                "player_id": player_id,
                "player_id_source": "master_or_override",
            }]
        ).to_csv(path / "cleaned_players.csv", index=False)

    mapping, generated = load_fpl_code_registry(
        registry, processed, target_season="2026-2027"
    )

    assert mapping["628204"] == "right001"
    assert "628204" not in generated


def test_fpl_code_registry_preserves_current_eight_character_id_policy():
    tmp_path = _case_dir("fpl_code_id_length")
    registry = tmp_path / "registry"
    (registry / "bridges").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["provider", "provider_id", "canonical_id", "match_method"]).to_csv(
        registry / "bridges" / "player_ids.csv", index=False
    )
    processed = tmp_path / "processed"
    for season, player_id in (("2023-2024", "9e868b71"), ("2025-2026", "9e868b7118fa")):
        path = processed / season / "season"
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{
                "fpl_code": "214285",
                "player_id": player_id,
                "player_id_source": "master_or_override",
            }]
        ).to_csv(path / "cleaned_players.csv", index=False)

    mapping, _ = load_fpl_code_registry(
        registry, processed, target_season="2026-2027"
    )

    assert mapping["214285"] == "9e868b71"


def test_generated_registration_refuses_name_collision():
    tmp_path = _case_dir("registry_collision")
    registry = tmp_path / "registry"
    (registry / "bridges").mkdir(parents=True, exist_ok=True)
    (registry / "master_players.json").write_text("{}", encoding="utf-8")
    (registry / "master_fpl.json").write_text("{}", encoding="utf-8")
    (registry / "_id_lookup_players.json").write_text(
        json.dumps({"new player": "different-id"}), encoding="utf-8"
    )
    players = pd.DataFrame(
        [
            {
                "player_id": "abc12345",
                "name": "New Player",
                "fpl_code": "98765",
                "team": "COV",
                "team_id": "team-cov",
                "fpl_pos": "MID",
            }
        ]
    )

    with pytest.raises(ValueError, match="Generated name collision"):
        register_generated_players(
            players,
            pd.Series([True]),
            season="2026-2027",
            league="ENG-Premier League",
            registry_root=registry,
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
