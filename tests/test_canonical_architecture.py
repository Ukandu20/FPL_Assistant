from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_assistant.canonical.contracts import (
    ColumnContract,
    TableContract,
    validate_contract,
)
from fpl_assistant.canonical.dnp import build_complete_player_fixture_panel
from fpl_assistant.canonical.facts import (
    DEFAULT_PLAYER_SOURCE_POLICY,
    build_canonical_facts,
)
from fpl_assistant.canonical.features import build_feature_snapshot
from fpl_assistant.canonical.identity import build_identity_registry
from fpl_assistant.canonical.manifest import RunManifest, determine_run_mode
from fpl_assistant.canonical.matches import build_match_registry
from fpl_assistant.canonical.staging import stage_match_facts
from fpl_assistant.canonical.whoscored_defense import (
    aggregate_whoscored_defensive_events,
)
from fpl_assistant.providers.fbref.capabilities import (
    LEVEL_PLAYER_MATCH,
    capability_document,
    validate_requested_stats,
    write_coverage_manifest,
)


def _artifact_dir(name: str) -> Path:
    path = Path(".test-artifacts") / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_fbref_reduced_capability_contract_rejects_removed_stats():
    assert validate_requested_stats(
        LEVEL_PLAYER_MATCH, ["summary", "keeper"]
    ) == ["summary", "keepers"]
    with pytest.raises(ValueError, match="Unsupported"):
        validate_requested_stats(LEVEL_PLAYER_MATCH, ["defense"])

    document = capability_document()
    assert document["capabilities"]["player_match"] == ["summary", "keepers"]
    assert "passing" in document["historical_unavailable"]

    path = write_coverage_manifest(
        _artifact_dir("fbref") / "coverage.json",
        league="ENG-Premier League",
        season="2026-2027",
        records=[],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["provider"] == "fbref"
    assert payload["capabilities"]["supplementary"] == ["lineups", "events"]


def test_identity_registry_builds_persistent_dimensions_and_bridges():
    records = pd.DataFrame(
        [
            {
                "provider": "fpl",
                "provider_id": "1",
                "provider_name": "Manchester United",
                "canonical_id": "team001",
                "canonical_name": "Manchester United",
            },
            {
                "provider": "fbref",
                "provider_id": "abc",
                "provider_name": "Manchester Utd",
            },
        ]
    )
    result = build_identity_registry(
        records,
        entity_type="team",
        aliases={"Manchester Utd": "team001"},
    )

    assert result.dimensions["canonical_id"].tolist() == ["team001"]
    assert set(result.bridges["provider"]) == {"fpl", "fbref"}
    assert result.bridges["canonical_id"].nunique() == 1
    assert result.review.empty

    rerun = build_identity_registry(
        pd.DataFrame(
            [
                {
                    "provider": "understat",
                    "provider_id": "89",
                    "provider_name": "Manchester United",
                }
            ]
        ),
        entity_type="team",
        existing_dimensions=result.dimensions,
        existing_bridges=result.bridges,
    )
    bridge = rerun.bridges[rerun.bridges["provider"].eq("understat")].iloc[0]
    assert bridge["canonical_id"] == "team001"
    assert bridge["match_method"] == "unique_normalized_name"


def _team_bridges() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entity_type": "team",
                "provider": provider,
                "provider_id": provider_id,
                "canonical_id": canonical_id,
            }
            for provider, provider_id, canonical_id in [
                ("fpl", "1", "ars"),
                ("fpl", "2", "che"),
                ("fbref", "Arsenal", "ars"),
                ("fbref", "Chelsea", "che"),
            ]
        ]
    )


def test_match_registry_reconciles_provider_ids_into_one_match():
    schedules = pd.DataFrame(
        [
            {
                "provider": "fpl",
                "provider_match_id": "100",
                "competition": "ENG-Premier League",
                "season": "2026-2027",
                "kickoff_utc": "2026-08-15T15:00:00Z",
                "home_provider_team_id": "1",
                "away_provider_team_id": "2",
                "gameweek": 1,
                "status": "scheduled",
            },
            {
                "provider": "fbref",
                "provider_match_id": "fb-100",
                "competition": "ENG-Premier League",
                "season": "2026-2027",
                "kickoff_utc": "2026-08-15T15:05:00Z",
                "home_provider_team_id": "Arsenal",
                "away_provider_team_id": "Chelsea",
                "round": "Matchweek 1",
            },
        ]
    )
    result = build_match_registry(schedules, team_bridges=_team_bridges())

    assert len(result.matches) == 1
    assert len(result.bridges) == 2
    assert result.bridges["match_id"].nunique() == 1
    assert result.matches.iloc[0]["gameweek"] == 1
    assert result.unresolved.empty


def test_facts_use_deterministic_precedence_and_preserve_provenance():
    records = pd.DataFrame(
        [
            {
                "provider": "fpl",
                "provider_record_id": "f1",
                "retrieved_at": "2026-08-16T00:00:00Z",
                "match_id": "m1",
                "player_id": "p1",
                "minutes": 90,
                "xg": None,
            },
            {
                "provider": "understat",
                "provider_record_id": "u1",
                "retrieved_at": "2026-08-16T01:00:00Z",
                "match_id": "m1",
                "player_id": "p1",
                "minutes": 89,
                "xg": 0.42,
            },
            {
                "provider": "fbref",
                "provider_record_id": "b1",
                "retrieved_at": "2026-08-16T02:00:00Z",
                "match_id": "m1",
                "player_id": "p1",
                "minutes": 90,
                "xg": 0.4,
            },
        ]
    )
    result = build_canonical_facts(
        records,
        entity_column="player_id",
        source_policy=DEFAULT_PLAYER_SOURCE_POLICY,
    )
    fact = result.facts.iloc[0]
    assert fact["minutes"] == 90
    assert fact["xg"] == 0.42
    sources = result.provenance.set_index("metric")["provider"]
    assert sources["minutes"] == "fpl"
    assert sources["xg"] == "understat"
    assert set(result.conflicts["metric"]) >= {"minutes", "xg"}


def test_provider_rows_are_staged_through_identity_and_match_bridges():
    raw = pd.DataFrame(
        [{"fixture": 100, "element": 22, "mins": 90, "goals_scored": 1}]
    )
    entity_bridges = pd.DataFrame(
        [{"provider": "fpl", "provider_id": "22", "canonical_id": "p22"}]
    )
    match_bridges = pd.DataFrame(
        [{"provider": "fpl", "provider_match_id": "100", "match_id": "m100"}]
    )
    staged = stage_match_facts(
        raw,
        provider="fpl",
        entity_type="player",
        entity_bridges=entity_bridges,
        match_bridges=match_bridges,
        provider_entity_id_column="element",
        provider_match_id_column="fixture",
        metric_columns={"mins": "minutes", "goals_scored": "goals"},
        retrieved_at="2026-08-16T00:00:00Z",
    )
    assert staged.loc[0, "player_id"] == "p22"
    assert staged.loc[0, "match_id"] == "m100"
    assert staged.loc[0, "minutes"] == 90
    assert staged.loc[0, "provider_record_id"]


def test_complete_player_fixture_panel_creates_real_dnp_rows():
    roster = pd.DataFrame(
        [
            {"season": "2026-2027", "player_id": "p1", "team_id": "ars"},
            {"season": "2026-2027", "player_id": "p2", "team_id": "ars"},
        ]
    )
    fixtures = pd.DataFrame(
        [
            {
                "season": "2026-2027",
                "gameweek": 1,
                "match_id": "m1",
                "kickoff_utc": "2026-08-15T15:00:00Z",
                "home_team_id": "ars",
                "away_team_id": "che",
                "status": "complete",
            },
            {
                "season": "2026-2027",
                "gameweek": 2,
                "match_id": "m2",
                "kickoff_utc": "2026-08-22T15:00:00Z",
                "home_team_id": "ars",
                "away_team_id": "liv",
                "status": "scheduled",
            },
        ]
    )
    observations = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "player_id": "p1",
                "minutes": 90,
                "started": True,
                "named_on_bench": False,
            }
        ]
    )
    panel = build_complete_player_fixture_panel(
        roster,
        fixtures,
        observations=observations,
        as_of_timestamp="2026-08-18T12:00:00Z",
    )

    dnp = panel[(panel["match_id"] == "m1") & (panel["player_id"] == "p2")].iloc[0]
    assert dnp["minutes"] == 0
    assert bool(dnp["did_not_play"]) is True
    assert dnp["observation_status"] == "not_in_matchday_squad"

    future = panel[panel["match_id"] == "m2"]
    assert future["minutes"].isna().all()
    assert future["did_not_play"].isna().all()


def test_whoscored_events_generate_defensive_components_and_audit():
    events = pd.DataFrame(
        [
            ["m1", "p1", "ars", "Tackle", "Successful", [], None],
            ["m1", "p1", "ars", "Interception", "Successful", [], None],
            ["m1", "p1", "ars", "Clearance", "Successful", [], None],
            ["m1", "p1", "ars", "BallRecovery", "Successful", [], None],
            ["m1", "p1", "ars", "BlockedPass", "Successful", [], None],
            ["m1", "gk1", "ars", "Save", "Successful", [], None],
            ["m1", "p2", "che", "Pass", "Successful", [], None],
        ],
        columns=[
            "match_id",
            "player_id",
            "team_id",
            "type",
            "outcome_type",
            "qualifiers",
            "related_player_id",
        ],
    )
    result = aggregate_whoscored_defensive_events(events)
    p1 = result.player_match[result.player_match["player_id"].eq("p1")].iloc[0]
    assert p1["tackles"] == 1
    assert p1["tackles_won"] == 1
    assert p1["defensive_contributions_def"] == 4
    assert p1["defensive_contributions_outfield"] == 5
    assert result.event_audit["event_count"].sum() == len(events)
    assert (~result.event_audit["recognized"]).any()


def test_feature_contract_prevents_time_leakage_and_manifest_hashes():
    tmp_path = _artifact_dir("manifest")
    contract = TableContract(
        name="minutes_features",
        version="1.0.0",
        key=("match_id", "player_id"),
        columns={
            "match_id": ColumnContract(nullable=False),
            "player_id": ColumnContract(nullable=False),
            "as_of_timestamp": ColumnContract(nullable=False),
        },
    )
    frame = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "player_id": "p1",
                "known_at": "2026-08-14T10:00:00Z",
            }
        ]
    )
    snapshot = build_feature_snapshot(
        frame,
        contract=contract,
        feature_version="minutes-v1",
        as_of_timestamp="2026-08-14T12:00:00Z",
    )
    assert snapshot.frame.iloc[0]["feature_version"] == "minutes-v1"
    assert validate_contract(snapshot.frame, contract) == []

    leaked = frame.assign(known_at="2026-08-14T13:00:00Z")
    with pytest.raises(ValueError, match="later than"):
        build_feature_snapshot(
            leaked,
            contract=contract,
            feature_version="minutes-v1",
            as_of_timestamp="2026-08-14T12:00:00Z",
        )

    artifact = tmp_path / "artifact.csv"
    artifact.write_text("a\n1\n", encoding="utf-8")
    manifest = RunManifest.create(run_id="gw01", mode="full")
    manifest.add_artifact(artifact, output=True)
    manifest.contracts["minutes_features"] = "1.0.0"
    manifest.finish()
    path = manifest.write(tmp_path / "manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["outputs"][0]["sha256"]
    assert payload["completed_at"]
    assert determine_run_mode(
        {"fpl": "ok", "understat": "ok", "clubelo": "ok"}
    ) == "standard"
    assert determine_run_mode(
        {
            "fpl": True,
            "understat": True,
            "clubelo": True,
            "fbref": True,
            "whoscored": True,
        }
    ) == "full"
