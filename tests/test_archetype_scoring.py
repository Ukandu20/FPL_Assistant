from __future__ import annotations

import pandas as pd

from fpl_assistant.archetypes.scoring import (
    score_base_components,
    score_base_components_with_evidence,
)


def _observations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for player_index, player_id in enumerate(("low", "mid", "high"), start=1):
        for appearance in range(12):
            scale = float(player_index)
            rows.append(
                {
                    "match_id": f"m-{player_id}-{appearance}",
                    "player_id": player_id,
                    "team_id": f"t{player_index}",
                    "season": "2025-2026",
                    "kickoff_utc": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=7 * appearance),
                    "fpl_position": "MID",
                    "minutes": 90,
                    "npxg": 0.10 * scale,
                    "shots_in_box": 1.0 * scale,
                    "shots_on_target": 0.5 * scale,
                    "non_penalty_goals": 0.05 * scale,
                    "xa": 0.08 * scale,
                    "key_passes": 0.5 * scale,
                    "big_chances_created": 0.1 * scale,
                    "shot_creating_actions": 1.0 * scale,
                    "tackles_won": 1.0 * scale,
                    "interceptions": 0.5 * scale,
                    "clearances": 0.25 * scale,
                    "blocks": 0.2 * scale,
                    "recoveries": 1.5 * scale,
                }
            )
    return pd.DataFrame(rows)


def test_base_scoring_is_position_relative_bounded_and_evidence_aware() -> None:
    scores = score_base_components(
        _observations(),
        as_of="2026-05-01T00:00:00Z",
        current_season="2025-2026",
    )
    assert set(scores["archetype_id"]) == {
        "GOAL_THREAT", "CREATOR", "DEFENSIVE_ENGINE"
    }
    assert scores["score_0_100"].dropna().between(0, 100).all()
    goal = scores[scores["archetype_id"].eq("GOAL_THREAT")].set_index("player_id")
    assert goal.at["high", "score_0_100"] > goal.at["mid", "score_0_100"] > goal.at["low", "score_0_100"]
    assert goal.at["high", "evidence_minutes"] == 1080
    assert goal.at["high", "eligible_appearances"] == 12


def test_base_scoring_retains_complete_calculation_ledger() -> None:
    result = score_base_components_with_evidence(
        _observations(),
        as_of="2026-05-01T00:00:00Z",
        current_season="2025-2026",
    )

    row = result.evidence.loc[
        result.evidence["player_id"].eq("high")
        & result.evidence["archetype_id"].eq("GOAL_THREAT")
        & result.evidence["evidence_window"].eq("recent")
    ].iloc[0]
    assert "npxg" in row["metric_values"]
    assert "shots_in_box" in row["transformed_values"]
    assert "shots_on_target" in row["winsorized_values"]
    assert "non_penalty_goals" in row["position_z_scores"]
    assert row["window_minutes"] == 900
    assert row["final_score_0_100"] == 100


def test_future_rows_do_not_change_historical_snapshot() -> None:
    observations = _observations()
    before = score_base_components(
        observations,
        as_of="2026-02-20T00:00:00Z",
        current_season="2025-2026",
    )
    changed = observations.copy()
    future = changed["kickoff_utc"].ge(pd.Timestamp("2026-02-20", tz="UTC"))
    changed.loc[future, ["npxg", "xa", "shots_in_box"]] = 9999
    after = score_base_components(
        changed,
        as_of="2026-02-20T00:00:00Z",
        current_season="2025-2026",
    )
    pd.testing.assert_frame_equal(before, after)


def test_missing_core_field_returns_no_score_with_reason() -> None:
    observations = _observations()
    observations.loc[observations["player_id"].eq("high"), "npxg"] = pd.NA
    scores = score_base_components(
        observations,
        as_of="2026-05-01T00:00:00Z",
        current_season="2025-2026",
    )
    row = scores[
        scores["player_id"].eq("high") & scores["archetype_id"].eq("GOAL_THREAT")
    ].iloc[0]
    assert pd.isna(row["score_0_100"])
    assert row["active_label"] == False
    assert "npxg" in row["missing_data_flags"]


def test_missing_secondary_field_reweights_only_when_seventy_percent_remains() -> None:
    observations = _observations()
    observations.loc[observations["player_id"].eq("high"), "recoveries"] = pd.NA
    scores = score_base_components(
        observations, as_of="2026-05-01T00:00:00Z", current_season="2025-2026"
    )
    row = scores[
        scores["player_id"].eq("high")
        & scores["archetype_id"].eq("DEFENSIVE_ENGINE")
    ].iloc[0]
    assert pd.notna(row["score_0_100"])
    assert row["confidence_0_1"] < 1


def test_red_card_and_short_appearances_are_not_scoring_evidence() -> None:
    observations = _observations()
    observations["red_card"] = False
    observations.loc[0, "red_card"] = True
    observations.loc[1, "minutes"] = 29
    scores = score_base_components(
        observations,
        as_of="2026-05-01T00:00:00Z",
        current_season="2025-2026",
    )
    row = scores[
        scores["player_id"].eq("low") & scores["archetype_id"].eq("GOAL_THREAT")
    ].iloc[0]
    assert row["eligible_appearances"] == 10
    assert row["evidence_minutes"] == 900


def test_valid_pre_red_card_event_slice_is_retained() -> None:
    observations = _observations()
    observations["red_cards"] = 0
    observations["pre_red_card_evidence_valid"] = False
    observations.loc[0, "red_cards"] = 1
    observations.loc[0, "pre_red_card_evidence_valid"] = True
    observations.loc[0, "pre_red_card_minutes"] = 45
    for field in (
        "npxg", "shots_in_box", "shots_on_target", "non_penalty_goals",
        "xa", "key_passes", "big_chances_created", "shot_creating_actions",
        "tackles_won", "interceptions", "clearances", "blocks", "recoveries",
    ):
        observations.loc[0, f"pre_red_card_{field}"] = observations.loc[0, field] / 2
    scores = score_base_components(
        observations, as_of="2026-05-01T00:00:00Z", current_season="2025-2026"
    )
    row = scores[
        scores["player_id"].eq("low") & scores["archetype_id"].eq("GOAL_THREAT")
    ].iloc[0]
    assert row["eligible_appearances"] == 12
    assert row["evidence_minutes"] == 1035
