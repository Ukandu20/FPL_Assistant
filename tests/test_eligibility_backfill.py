from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd

from fpl_assistant.providers.fpl.pipelines.eligibility_backfill import (
    _resolve_and_publish_historical_identities,
    build_effective_dated_eligibility,
    combine_inseason_universes,
    enrich_player_attacking_stats,
    expand_player_fixture_calendar,
    normalize_fixture_calendar,
)


def _fixtures(status: str = "finished") -> pd.DataFrame:
    rows = []
    for fixture_id, date in ((1, "2026-08-21"), (2, "2026-08-28")):
        for is_home, team_id, opponent_id, team in (
            (1, "ars", "che", "ARS"),
            (0, "che", "ars", "CHE"),
        ):
            rows.append(
                {
                    "fpl_id": fixture_id,
                    "match_id": f"m{fixture_id}",
                    "fbref_id": f"m{fixture_id}",
                    "gw_orig": fixture_id,
                    "gw_played": fixture_id if status == "finished" else "",
                    "date_sched": date,
                    "date_played": date if status == "finished" else "",
                    "team": team,
                    "team_id": team_id,
                    "opponent_id": opponent_id,
                    "is_home": is_home,
                    "status": status,
                    "venue": "Example",
                    "gf": 1 if status == "finished" else "",
                    "ga": 0 if status == "finished" else "",
                }
            )
    return normalize_fixture_calendar(pd.DataFrame(rows), "2026-2027")


def test_expansion_adds_true_dnp_rows_beyond_matchday_squad():
    fixtures = _fixtures()
    universe = pd.DataFrame(
        [
            {
                "_fixture_key": str(fixture_id),
                "fixture": fixture_id,
                "player_id": player_id,
                "team_id": "ars",
                "name": name,
                "position": "MF",
                "minutes": minutes,
                "starts": int(minutes > 0),
                "total_points": points,
            }
            for fixture_id, player_id, name, minutes, points in (
                (1, "p1", "Starter", 90, 6),
                (1, "p2", "Eligible DNP", 0, 0),
                (2, "p1", "Starter", 90, 2),
                (2, "p2", "Eligible DNP", 0, 0),
            )
        ]
    )
    observed = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "fbref_id": "m1",
                "fpl_id": 1,
                "player_id": "p1",
                "team_id": "ars",
                "player": "Starter",
                "pos": "MF",
                "minutes": 90,
                "is_starter": 1,
            }
        ]
    )

    panel, audit = expand_player_fixture_calendar(
        "2026-2027",
        fixtures=fixtures,
        universe=universe,
        observed=observed,
        availability=None,
        information_timestamp="2027-06-01T00:00:00Z",
        eligibility_source="fpl_merged_gws_retrospective",
        timestamp_safe=False,
    )

    dnp = panel[(panel["match_id"] == "m1") & (panel["player_id"] == "p2")].iloc[0]
    assert float(dnp["minutes"]) == 0
    assert bool(dnp["did_not_play"]) is True
    assert dnp["observation_status"] == "not_in_matchday_squad"
    assert dnp["row_source"] == "fpl_eligible_dnp"
    assert audit["not_in_matchday_squad_rows"] == 2
    assert "gf" not in panel.columns
    assert "ga" not in panel.columns
    assert "xga" not in panel.columns
    assert float(panel.loc[0, "team_gf"]) == 1
    assert float(panel.loc[0, "team_ga"]) == 0

    eligibility = build_effective_dated_eligibility("2026-2027", panel, fixtures)
    p2 = eligibility[eligibility["player_id"] == "p2"].iloc[0]
    assert p2["fixtures_observed"] == 2
    assert p2["valid_from"] == "2026-08-21T00:00:00Z"
    assert p2["valid_until"] == "2026-08-28T00:00:00Z"
    assert bool(p2["timestamp_safe"]) is False


def test_pending_preseason_rows_keep_unknown_minutes_and_apply_snapshot_status():
    fixtures = _fixtures(status="scheduled")
    universe = pd.DataFrame(
        [
            {
                "_fixture_key": str(fixture_id),
                "fixture": fixture_id,
                "player_id": "p1",
                "team_id": "ars",
                "name": "Unavailable Player",
                "position": "FW",
                "minutes": pd.NA,
                "starts": pd.NA,
            }
            for fixture_id in (1, 2)
        ]
    )
    availability = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "confirmed_unavailable": True,
                "unavailable_reason": "Suspended",
            }
        ]
    )

    panel, audit = expand_player_fixture_calendar(
        "2026-2027",
        fixtures=fixtures,
        universe=universe,
        observed=pd.DataFrame(),
        availability=availability,
        information_timestamp="2026-08-20T12:00:00Z",
        eligibility_source="fpl_bootstrap_roster_snapshot",
        timestamp_safe=True,
    )

    assert panel["minutes"].isna().all()
    assert panel["did_not_play"].isna().all()
    assert panel["observation_status"].eq("fixture_pending").all()
    assert ~panel["eligible_for_fixture"].all()
    assert audit["pending_rows"] == 2


def test_understat_enrichment_separates_player_metrics_from_team_context():
    root = Path(".tmp") / f"understat_player_metrics_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    source = root / "player_match.csv"
    pd.DataFrame([
        {
            "match_id": "m1", "player_id": "p1", "goals": 1,
            "assists": 0, "xg": 0.72, "xa": 0.18,
        }
    ]).to_csv(source, index=False)
    panel = pd.DataFrame([
        {
            "match_id": "m1", "player_id": "p1", "minutes": 90,
            "observation_status": "played", "team_xg": 1.85,
            "team_xga": 0.56, "xg": 1.85,
        },
        {
            "match_id": "m1", "player_id": "p2", "minutes": 0,
            "observation_status": "not_in_matchday_squad", "team_xg": 1.85,
            "team_xga": 0.56, "xg": 1.85,
        },
    ])

    enriched, audit = enrich_player_attacking_stats(panel, source)

    active = enriched.set_index("player_id").loc["p1"]
    dnp = enriched.set_index("player_id").loc["p2"]
    assert active[["goals", "assists", "xg", "xa"]].tolist() == [1, 0, 0.72, 0.18]
    assert active[["team_xg", "team_xga"]].tolist() == [1.85, 0.56]
    assert dnp[["goals", "assists", "xg", "xa"]].tolist() == [0, 0, 0, 0]
    assert audit["player_attacking_provider_rows_matched"] == 1
    assert audit["player_attacking_rows_populated"] == 2


def test_inseason_universe_keeps_observed_rows_and_timestamp_safe_future_rows():
    fixtures = _fixtures()
    fixtures.loc[fixtures["fpl_id"].astype(str).eq("2"), ["status", "date_played"]] = ["scheduled", ""]
    fixtures = normalize_fixture_calendar(
        fixtures.drop(columns=["_fixture_key", "_fixture_time"]), "2026-2027"
    )
    historical = pd.DataFrame(
        [{
            "_fixture_key": "1", "fixture": 1, "player_id": "p1",
            "team_id": "ars", "name": "Player", "position": "MF",
            "minutes": 90, "starts": 1,
        }]
    )
    live = pd.DataFrame(
        [
            {
                "_fixture_key": str(fixture_id), "fixture": fixture_id,
                "player_id": "p1", "team_id": "ars", "name": "Player",
                "position": "MF", "minutes": pd.NA, "starts": pd.NA,
            }
            for fixture_id in (1, 2)
        ]
    )
    combined, audit = combine_inseason_universes(
        historical,
        {
            "source": "fpl_merged_gws_retrospective",
            "information_timestamp": "2026-08-25T00:00:00Z",
            "timestamp_safe": False,
        },
        live,
        {
            "source": "fpl_bootstrap_roster_snapshot",
            "information_timestamp": "2026-08-26T00:00:00Z",
            "timestamp_safe": True,
            "roster_players": 1,
        },
    )
    panel, panel_audit = expand_player_fixture_calendar(
        "2026-2027",
        fixtures=fixtures,
        universe=combined,
        observed=pd.DataFrame(),
        availability=None,
        information_timestamp=audit["information_timestamp"],
        eligibility_source=audit["source"],
        timestamp_safe=audit["timestamp_safe"],
    )

    by_fixture = panel.set_index("fpl_id")
    assert bool(by_fixture.loc[1, "eligibility_timestamp_safe"]) is False
    assert by_fixture.loc[1, "eligibility_source"] == "fpl_merged_gws_retrospective"
    assert float(by_fixture.loc[1, "minutes"]) == 90
    assert bool(by_fixture.loc[2, "eligibility_timestamp_safe"]) is True
    assert by_fixture.loc[2, "eligibility_source"] == "fpl_bootstrap_roster_snapshot"
    assert pd.isna(by_fixture.loc[2, "minutes"])
    assert panel_audit["pending_rows"] == 1

    eligibility = build_effective_dated_eligibility("2026-2027", panel, fixtures)
    assert len(eligibility) == 2
    assert set(eligibility["timestamp_safe"]) == {False, True}


def test_official_fpl_scores_mark_lagging_scheduled_fixture_as_completed():
    fixtures = _fixtures(status="scheduled").iloc[:2].copy()
    universe = pd.DataFrame(
        [
            {
                "_fixture_key": "1",
                "fixture": 1,
                "player_id": player_id,
                "team_id": "ars",
                "name": name,
                "position": "MF",
                "minutes": minutes,
                "starts": int(minutes > 0),
                "team_h_score": 3,
                "team_a_score": 0,
            }
            for player_id, name, minutes in (
                ("p1", "Starter", 90),
                ("p2", "Eligible DNP", 0),
            )
        ]
    )

    panel, audit = expand_player_fixture_calendar(
        "2026-2027",
        fixtures=fixtures,
        universe=universe,
        observed=pd.DataFrame(),
        availability=None,
        information_timestamp="2026-08-25T00:00:00Z",
        eligibility_source="fpl_merged_gws_retrospective",
        timestamp_safe=False,
    )

    assert panel.set_index("player_id")["observation_status"].to_dict() == {
        "p1": "played",
        "p2": "not_in_matchday_squad",
    }
    assert audit["played_rows"] == 1
    assert audit["not_in_matchday_squad_rows"] == 1
    assert audit["pending_rows"] == 0


def test_effective_membership_splits_when_team_fixture_sequence_has_a_gap():
    fixtures = _fixtures()
    panel = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "player_id": "p1",
                "player": "Player",
                "pos": "MF",
                "team_id": "ars",
                "confirmed_unavailable": False,
                "unavailable_reason": "",
                "information_timestamp": "2027-06-01T00:00:00Z",
                "eligibility_source": "test",
                "eligibility_timestamp_safe": False,
                "observation_status": "played",
            }
        ]
    )
    # Add a third team fixture while deliberately omitting the second from the panel.
    extra = fixtures.iloc[[0]].copy()
    extra["fpl_id"] = 3
    extra["match_id"] = "m3"
    extra["fbref_id"] = "m3"
    extra["date_sched"] = "2026-09-04"
    extra["date_played"] = "2026-09-04"
    fixtures = normalize_fixture_calendar(
        pd.concat([fixtures.drop(columns=["_fixture_key", "_fixture_time"]), extra.drop(columns=["_fixture_key", "_fixture_time"])], ignore_index=True),
        "2026-2027",
    )
    panel = pd.concat([panel, panel.assign(match_id="m3")], ignore_index=True)

    eligibility = build_effective_dated_eligibility("2026-2027", panel, fixtures)
    assert len(eligibility) == 2


def test_historical_identity_repair_separates_collided_fpl_elements():
    registry = Path(".tmp") / f"eligibility_identity_{uuid.uuid4().hex}"
    (registry / "bridges").mkdir(parents=True)
    for filename in (
        "master_players.json",
        "master_fpl.json",
        "master_teams.json",
        "_id_lookup_players.json",
        "overrides.json",
    ):
        (registry / filename).write_text("{}", encoding="utf-8")
    pd.DataFrame(
        columns=[
            "entity_type", "provider", "provider_id", "provider_name",
            "canonical_id", "valid_from", "valid_to", "match_method",
            "match_confidence", "review_status",
        ]
    ).to_csv(registry / "bridges" / "player_ids.csv", index=False)
    raw_players = pd.DataFrame(
        [
            {
                "id": 10,
                "code": 1000,
                "first_name": "First",
                "second_name": "Player",
                "element_type": 3,
            },
            {
                "id": 11,
                "code": 1001,
                "first_name": "Second",
                "second_name": "Player",
                "element_type": 4,
            },
        ]
    )
    # Simulate a bad historical cleaner assigning both elements one ID.
    fpl = pd.DataFrame(
        [
            {
                "element": "10", "player_id": "collision", "fixture": 1,
                "team_id": "ars", "team": "ARS", "minutes": 0,
                "kickoff_time": "2024-08-01T12:00:00Z", "position": "MF",
            },
            {
                "element": "11", "player_id": "collision", "fixture": 1,
                "team_id": "ars", "team": "ARS", "minutes": 90,
                "kickoff_time": "2024-08-01T12:00:00Z", "position": "FW",
            },
        ]
    )

    repaired, audit = _resolve_and_publish_historical_identities(
        "2024-2025",
        raw_players=raw_players,
        fpl=fpl,
        registry_root=registry,
        code_registry={},
    )

    assert repaired["player_id"].nunique() == 2
    assert audit["identity_generated_from_fpl_code"] == 2
    master = json.loads((registry / "master_players.json").read_text(encoding="utf-8"))
    bridges = pd.read_csv(registry / "bridges" / "player_ids.csv", dtype=str)
    assert set(repaired["player_id"]) <= set(master)
    assert bridges["provider_id"].nunique() == 2
