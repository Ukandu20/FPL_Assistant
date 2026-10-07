from __future__ import annotations

import json

import pandas as pd
import pytest

from fpl_assistant.apps.viewmodels.player_signal_profile import (
    build_player_signal_cards,
    current_season_signal_profiles,
)


def _archetypes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "selected",
                "fpl_position": "MID",
                "archetype_id": "GOAL_THREAT",
                "score_0_100": 88.0,
                "confidence_band": "High",
                "trend": "Rising",
                "status": "Established",
                "evidence_minutes": 1978,
                "eligible_appearances": 25,
            },
            {
                "player_id": "selected",
                "fpl_position": "MID",
                "archetype_id": "CREATOR",
                "score_0_100": 74.0,
                "confidence_band": "Medium",
                "trend": "Stable",
                "status": "Provisional",
                "evidence_minutes": 1978,
                "eligible_appearances": 25,
            },
            {
                "player_id": "selected",
                "fpl_position": "MID",
                "archetype_id": "DEFENSIVE_ENGINE",
                "score_0_100": 63.0,
                "confidence_band": "High",
                "trend": "Stable",
                "status": "Established",
                "evidence_minutes": 1978,
                "eligible_appearances": 25,
            },
        ]
    )


def _evidence_row(
    archetype_id: str,
    window: str,
    values: dict[str, float],
    z_scores: dict[str, float],
    weights: dict[str, float],
) -> dict[str, object]:
    return {
        "player_id": "selected",
        "archetype_id": archetype_id,
        "evidence_window": window,
        "metric_values": json.dumps(values),
        "position_z_scores": json.dumps(z_scores),
        "metric_weights": json.dumps(weights),
        "missing_core_fields": "[]",
        "applied_window_weights": json.dumps(
            {"previous_season": 0.35, "recent": 0.65}
        ),
    }


def _component_evidence() -> pd.DataFrame:
    rows = []
    definitions = {
        "GOAL_THREAT": (
            {"npxg": 0.54, "shots_in_box": 3.2, "shots_on_target": 1.31, "non_penalty_goals": 0.42},
            {"npxg": 1.34, "shots_in_box": 1.0, "shots_on_target": 0.81, "non_penalty_goals": 0.58},
            {"npxg": 0.45, "shots_in_box": 0.25, "shots_on_target": 0.2, "non_penalty_goals": 0.1},
        ),
        "CREATOR": (
            {"xa": 0.31, "key_passes": 2.4, "big_chances_created": 0.5, "shot_creating_actions": 3.1},
            {"xa": 1.1, "key_passes": 0.8, "big_chances_created": 0.5, "shot_creating_actions": 0.6},
            {"xa": 0.5, "key_passes": 0.25, "big_chances_created": 0.15, "shot_creating_actions": 0.1},
        ),
        "DEFENSIVE_ENGINE": (
            {"defcon_hit": 0.4, "tackles_won": 1.8, "interceptions": 1.1, "clearances": 2.7, "blocks": 0.8, "recoveries": 4.4},
            {"defcon_hit": 0.7, "tackles_won": 0.5, "interceptions": 0.2, "clearances": 0.4, "blocks": 0.1, "recoveries": 0.9},
            {"defcon_hit": 0.25, "tackles_won": 0.15, "interceptions": 0.15, "clearances": 0.15, "blocks": 0.15, "recoveries": 0.15},
        ),
    }
    for archetype_id, (values, z_scores, weights) in definitions.items():
        old_values = {key: value / 2 for key, value in values.items()}
        rows.append(
            _evidence_row(
                archetype_id, "previous_season", old_values, z_scores, weights
            )
        )
        rows.append(
            _evidence_row(archetype_id, "recent", values, z_scores, weights)
        )
    return pd.DataFrame(rows)


def _gameweeks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["selected", "selected", "peer"],
            "name": ["Selected", "Selected", "Peer"],
            "fpl_pos": ["MID", "MID", "MID"],
            "minutes": [90, 60, 90],
            "total_points": [10, 5, 4],
            "goals_scored": [1, 0, 0],
            "assists": [0, 1, 0],
            "clean_sheets": [0, 0, 0],
            "saves": [0, 0, 0],
            "bonus": [3, 1, 0],
        }
    )


def test_builds_four_cards_from_persisted_and_official_evidence() -> None:
    cards = build_player_signal_cards(
        _archetypes(),
        _component_evidence(),
        pd.Series(
            {
                "Points": 15,
                "Points Position Percentile": 100,
                "Bonus": 4,
                "Bonus Position Percentile": 100,
            }
        ),
        _gameweeks(),
        player_id="selected",
        player_name="Selected",
        fpl_position="MID",
    )

    assert [card["id"] for card in cards] == [
        "goal_threat",
        "assist_potential",
        "defensive_contribution",
        "fpl_output",
    ]
    goal = cards[0]
    assert goal["headline_value"] == 88
    assert goal["confidence_band"] == "High"
    assert goal["trend"] == "Rising"
    assert goal["evidence_window"] == "recent"
    assert goal["components"][0]["raw_value"] == 0.54
    assert goal["components"][0]["weight"] == 0.45
    assert goal["components"][0]["percentile"] == 100.0

    defensive = cards[2]
    assert defensive["components"][0]["display_value"] == "40.0%"

    output = cards[3]
    assert output["headline_value"] == 100
    assert output["eligible_appearances"] == 2
    output_components = {
        component["id"]: component for component in output["components"]
    }
    assert output_components["points_per_appearance"]["raw_value"] == 7.5
    assert output_components["points_per_90"]["raw_value"] == 9.0
    assert output_components["return_rate"]["display_value"] == "100.0%"
    assert output_components["haul_rate"]["display_value"] == "50.0%"
    assert output_components["blank_rate"]["display_value"] == "0.0%"


def test_goalkeeper_only_receives_fpl_output_card() -> None:
    cards = build_player_signal_cards(
        _archetypes(),
        _component_evidence(),
        pd.Series({"Points": 0}),
        pd.DataFrame(),
        player_id="keeper",
        player_name="Keeper",
        fpl_position="GKP",
    )

    assert [card["id"] for card in cards] == ["fpl_output"]
    assert cards[0]["status"] == "Insufficient appearances"


def test_component_percentile_is_ranked_against_same_position_peers() -> None:
    archetypes = _archetypes()
    peer_archetype = archetypes.iloc[[0]].copy()
    peer_archetype["player_id"] = "peer"
    archetypes = pd.concat([archetypes, peer_archetype], ignore_index=True)
    evidence = _component_evidence()
    peer_evidence = evidence.loc[
        evidence["archetype_id"].eq("GOAL_THREAT")
        & evidence["evidence_window"].eq("recent")
    ].copy()
    peer_evidence["player_id"] = "peer"
    peer_evidence["position_z_scores"] = json.dumps(
        {
            "npxg": 2.0,
            "shots_in_box": 2.0,
            "shots_on_target": 2.0,
            "non_penalty_goals": 2.0,
        }
    )
    evidence = pd.concat([evidence, peer_evidence], ignore_index=True)

    cards = build_player_signal_cards(
        archetypes,
        evidence,
        pd.Series({"Points": 15}),
        _gameweeks(),
        player_id="selected",
        player_name="Selected",
        fpl_position="MID",
    )

    assert cards[0]["components"][0]["percentile"] == 50.0


def test_missing_profile_artifact_keeps_explicit_unavailable_cards() -> None:
    cards = build_player_signal_cards(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.Series({"Points": 0}),
        pd.DataFrame(),
        player_id="new",
        player_name="New Player",
        fpl_position="FWD",
    )

    assert len(cards) == 4
    assert all(
        card["status"] == "Profile not published" for card in cards[:3]
    )
    assert cards[3]["status"] == "Insufficient appearances"


def _season_matches() -> pd.DataFrame:
    metrics = {}
    for row in _component_evidence().query("evidence_window == 'recent'").to_dict("records"):
        metrics.update(json.loads(row["metric_values"]))
    return pd.DataFrame([
        {
            **metrics,
            "player_id": player,
            "season": "2026-2027",
            "kickoff_utc": f"2026-08-{day:02d}T12:00:00Z",
            "snapshot_date": "2026-09-01T00:00:00Z",
            "fpl_position": "MID",
            "minutes": 90,
            "npxg": (1.0 if day <= 2 else 0.1) * scale,
        }
        for player, scale in [("selected", 1), ("peer", 2)]
        for day in range(1, 13)
    ])


def test_season_signals_use_full_season_and_ignore_previous_and_future_rows() -> None:
    matches = _season_matches()
    original = matches.copy(deep=True)
    scores, evidence = current_season_signal_profiles(matches, "2026-2027")
    old = matches.assign(season="2025-2026", npxg=9999)
    future = matches.assign(kickoff_utc="2026-09-02T12:00:00Z", npxg=9999)
    actual_scores, actual_evidence = current_season_signal_profiles(
        pd.concat([matches, old, future], ignore_index=True), "2026-2027"
    )
    pd.testing.assert_frame_equal(scores, actual_scores)
    pd.testing.assert_frame_equal(evidence, actual_evidence)
    pd.testing.assert_frame_equal(matches, original)
    cards = build_player_signal_cards(
        scores, evidence, pd.Series({"Points": 15}), _official_season_matches(),
        player_id="selected", player_name="Selected", fpl_position="MID",
        current_season_only=True,
    )
    goal = cards[0]
    assert goal["evidence_minutes"] == 1080
    assert goal["eligible_appearances"] == 12
    assert goal["evidence_window"] == "current_season"
    assert goal["trend"] is None
    assert goal["components"][0]["raw_value"] == pytest.approx(0.25)
    assert goal["components"][0]["percentile"] == 50
    assert cards[3]["eligible_appearances"] == 12


def test_season_signals_have_no_previous_season_fallback() -> None:
    scores, evidence = current_season_signal_profiles(_season_matches(), "2027-2028")
    assert scores.empty and evidence.empty
    cards = build_player_signal_cards(
        scores, evidence, pd.Series({"Points": 0}), pd.DataFrame(),
        player_id="selected", player_name="Selected", fpl_position="MID",
        current_season_only=True,
    )
    assert all(card["headline_value"] is None for card in cards[:3])
    assert all(card["status"] == "Current-season evidence unavailable" for card in cards[:3])


def test_season_signals_preserve_eligibility_and_missing_data_rules() -> None:
    matches = _season_matches()
    matches.loc[0, "minutes"] = 29
    matches.loc[1, "red_cards"] = 1
    matches.loc[2, "npxg"] = float("nan")
    scores, evidence = current_season_signal_profiles(matches, "2026-2027")
    goal = scores.loc[
        scores["player_id"].eq("selected") & scores["archetype_id"].eq("GOAL_THREAT")
    ].iloc[0]
    assert goal["eligible_appearances"] == 10
    assert goal["evidence_minutes"] == 900
    assert pd.isna(goal["score_0_100"])
    assert "npxg" in goal["missing_data_flags"]


def _official_season_matches() -> pd.DataFrame:
    matches = _season_matches()
    return pd.DataFrame({
        "player_id": matches["player_id"],
        "fixture": list(range(1, 13)) * 2,
        "fpl_pos": matches["fpl_position"],
        "kickoff_time": matches["kickoff_utc"],
        "minutes": matches["minutes"],
        "total_points": 2,
        "red_cards": 0,
    })


def test_shared_fpl_eligibility_includes_red_cards_and_reports_missing_matches() -> None:
    official = _official_season_matches()
    official.loc[0, "red_cards"] = 1
    official.loc[1, "minutes"] = 29
    provider = _season_matches().assign(fpl_fixture_id=list(range(1, 13)) * 2)
    provider.loc[0, "red_cards"] = 1
    provider.loc[0, "minutes"] = 10  # Official FPL minutes own eligibility.
    provider = provider.drop(index=2)
    scores, evidence = current_season_signal_profiles(provider, "2026-2027", official)
    cards = build_player_signal_cards(
        scores, evidence, pd.Series({"Points": 24}), official,
        player_id="selected", player_name="Selected", fpl_position="MID",
        current_season_only=True,
    )
    assert all(card["eligible_appearances"] == 11 for card in cards)
    assert all(card["evidence_minutes"] == 990 for card in cards)
    assert cards[0]["covered_appearances"] == 10
    assert cards[0]["headline_value"] is None
    assert cards[0]["components"][0]["raw_value"] is None
    assert pd.isna(cards[0]["confidence_band"])
    complete, _ = current_season_signal_profiles(
        _season_matches().assign(fpl_fixture_id=list(range(1, 13)) * 2, red_cards=1),
        "2026-2027", official,
    )
    goal = complete.query("player_id == 'selected' and archetype_id == 'GOAL_THREAT'").iloc[0]
    assert goal["covered_appearances"] == 11
    assert pd.notna(goal["score_0_100"])


def test_missing_provider_season_keeps_fpl_sample_and_no_zero_statistics() -> None:
    official = _official_season_matches()
    keeper = official.iloc[[0]].assign(player_id="keeper", fpl_pos="GKP")
    official = pd.concat([official, keeper], ignore_index=True)
    provider = _season_matches().assign(season="2025-2026", fpl_fixture_id=list(range(1, 13)) * 2)
    scores, evidence = current_season_signal_profiles(provider, "2026-2027", official)
    cards = build_player_signal_cards(
        scores, evidence, pd.Series({"Points": 24}), official,
        player_id="selected", player_name="Selected", fpl_position="MID",
        current_season_only=True,
    )
    for card in cards[:3]:
        assert card["eligible_appearances"] == 12
        assert card["covered_appearances"] == 0
        assert card["headline_value"] is None
        assert all(component["raw_value"] is None for component in card["components"])
        assert all(component["percentile"] is None for component in card["components"])


def test_fixture_join_never_uses_same_gameweek_or_ambiguous_provider_rows() -> None:
    official = _official_season_matches().assign(round=1)
    provider = _season_matches().assign(fpl_fixture_id=list(range(1, 13)) * 2)
    provider = pd.concat([provider, provider.iloc[[0]]], ignore_index=True)
    scores, _ = current_season_signal_profiles(provider, "2026-2027", official)
    goal = scores.query("player_id == 'selected' and archetype_id == 'GOAL_THREAT'").iloc[0]
    assert goal["eligible_appearances"] == 12
    assert goal["covered_appearances"] == 11
