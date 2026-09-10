from __future__ import annotations

import json

import pandas as pd

from fpl_assistant.apps.viewmodels.player_signal_profile import (
    build_player_signal_cards,
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
