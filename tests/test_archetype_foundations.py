from __future__ import annotations

import math

import pandas as pd
import pytest

from fpl_assistant.archetypes.composites import resolve_production_composite
from fpl_assistant.archetypes.config import load_config
from fpl_assistant.archetypes.evidence import (
    confidence_band,
    confidence_score,
    hysteresis_state,
)
from fpl_assistant.archetypes.team_ratings import (
    TeamRating,
    build_pre_match_ratings,
    regress_for_new_season,
)
from fpl_assistant.archetypes.temporal import (
    assign_evidence_windows,
    combine_window_scores,
    shrink_toward_mean,
)
from fpl_assistant.archetypes.usage import (
    USAGE_STATES, classify_usage, estimate_usage, stabilize_usage_state,
)


def test_released_configuration_matches_catalogue_scope() -> None:
    config = load_config()
    assert config.model_version == "1.0.0"
    assert config.values["temporal_weights"] == {
        "previous_season": 0.25,
        "earlier_current": 0.30,
        "recent": 0.45,
    }
    assert set(config.values["deferred_components"]) == {"SWEEPER", "DISTRIBUTOR"}
    assert set(config.values["components"]) == {
        "GOAL_THREAT", "CREATOR", "DEFENSIVE_ENGINE", "SAVES_MACHINE"
    }
    assert config.values["provider_fields"]["non_penalty_goals"] == "understat"
    assert config.values["provider_fields"]["shots_on_target_faced"] == "whoscored"


def test_missing_temporal_window_is_proportionally_redistributed() -> None:
    score, weights = combine_window_scores(
        {"previous_season": 40.0, "earlier_current": None, "recent": 80.0},
        {"previous_season": 0.25, "earlier_current": 0.30, "recent": 0.45},
    )
    assert weights == pytest.approx({
        "previous_season": 0.25 / 0.70,
        "earlier_current": 0.0,
        "recent": 0.45 / 0.70,
    })
    assert score == pytest.approx(40 * 0.25 / 0.70 + 80 * 0.45 / 0.70)


def test_temporal_windows_use_only_eligible_chronological_appearances() -> None:
    observations = pd.DataFrame(
        {
            "season": ["2024-2025", *(["2025-2026"] * 12)],
            "kickoff_utc": pd.date_range("2025-05-01", periods=13, freq="7D", tz="UTC"),
            "minutes": [90, 10, *([60] * 11)],
        }
    )
    windows = assign_evidence_windows(observations, current_season="2025-2026")
    assert windows.iloc[0] == "previous_season"
    assert pd.isna(windows.iloc[1])
    assert windows.eq("recent").sum() == 10
    assert windows.eq("earlier_current").sum() == 1


def test_recent_window_is_selected_independently_for_each_player() -> None:
    observations = pd.DataFrame(
        {
            "player_id": ["a"] * 12 + ["b"] * 12,
            "season": ["2025-2026"] * 24,
            "kickoff_utc": list(
                pd.date_range("2025-08-01", periods=12, freq="7D", tz="UTC")
            ) * 2,
            "minutes": [90] * 24,
        }
    )
    windows = assign_evidence_windows(observations, current_season="2025-2026")
    observations["window"] = windows

    assert observations.groupby("player_id")["window"].apply(
        lambda values: values.eq("recent").sum()
    ).to_dict() == {"a": 10, "b": 10}


def test_shrinkage_is_separate_from_temporal_weighting() -> None:
    assert shrink_toward_mean(
        90.0, evidence_minutes=450, position_mean=50.0, prior_minutes=900
    ) == pytest.approx(63.3333333333)


@pytest.mark.parametrize(
    ("start", "minutes", "cameo", "expected"),
    [
        (0.90, 75, 0.0, "Nailed"),
        (0.89, 75, 0.0, "Regular Starter"),
        (0.70, 55, 0.0, "Regular Starter"),
        (0.35, 0, 0.0, "Rotation Risk"),
        (0.0, 30, 0.0, "Rotation Risk"),
        (0.34, 29.9, 0.35, "Impact Sub"),
        (0.34, 29.9, 0.34, "Fringe"),
    ],
)
def test_usage_state_boundaries_have_no_gaps(
    start: float, minutes: float, cameo: float, expected: str
) -> None:
    assert classify_usage(start, minutes, cameo) == expected
    assert expected in USAGE_STATES


def test_usage_uses_latest_six_team_matches_and_half_weights_reason_absences() -> None:
    history = pd.DataFrame(
        {
            "kickoff_utc": pd.date_range("2026-01-01", periods=8, freq="7D", tz="UTC"),
            "started": [False, False, True, True, True, True, False, False],
            "minutes": [0, 0, 90, 90, 90, 90, 10, 0],
            "availability_status": ["available"] * 8,
            "availability_reason": [None] * 7 + ["injury"],
        }
    )
    result = estimate_usage(history, prior_observations=0)
    assert result["observations"] == 6
    assert 0 <= result["start_probability"] <= 1
    assert 0 <= result["expected_minutes"] <= 90


def test_preseason_usage_uses_entire_previous_season_with_equal_weights() -> None:
    history = pd.DataFrame(
        {
            "kickoff_utc": pd.date_range("2024-08-01", periods=44, freq="7D", tz="UTC"),
            "season": ["2024-2025"] * 6 + ["2025-2026"] * 38,
            "started": [False] * 6 + [True] * 30 + [False] * 8,
            "minutes": [0] * 6 + [90] * 30 + [0] * 8,
            "availability_status": ["available"] * 44,
        }
    )

    result = estimate_usage(
        history,
        current_season="2026-2027",
        prior_observations=0,
    )

    assert result["observations"] == 38
    assert result["start_probability"] == pytest.approx(30 / 38)
    assert result["expected_minutes"] == pytest.approx(2700 / 38, abs=0.001)

    in_season_result = estimate_usage(
        history,
        current_season="2026-2027",
        preseason=False,
        prior_observations=0,
    )
    assert in_season_result["observations"] == 6


def test_usage_state_change_requires_two_updates_except_immediate_reset() -> None:
    first = stabilize_usage_state(
        "Rotation Risk", previous_state="Regular Starter"
    )
    assert first == ("Regular Starter", "Rotation Risk", 1)
    second = stabilize_usage_state(
        "Rotation Risk", previous_state=first[0],
        pending_state=first[1], pending_updates=first[2],
    )
    assert second == ("Rotation Risk", None, 0)
    reset = stabilize_usage_state(
        "Fringe", previous_state="Nailed", immediate_reset=True
    )
    assert reset == ("Fringe", None, 0)


def test_usage_hysteresis_does_not_advance_without_new_evidence() -> None:
    unchanged = stabilize_usage_state(
        "Rotation Risk",
        previous_state="Regular Starter",
        pending_state="Rotation Risk",
        pending_updates=1,
        evidence_changed=False,
    )
    assert unchanged == ("Regular Starter", "Rotation Risk", 1)


def test_usage_fingerprint_changes_only_when_usage_evidence_changes() -> None:
    history = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "kickoff_utc": pd.to_datetime(["2026-08-01", "2026-08-08"], utc=True),
            "season": ["2026-2027", "2026-2027"],
            "started": [True, True],
            "minutes": [90, 80],
            "availability_status": ["available", "available"],
        }
    )
    first = estimate_usage(history, current_season="2026-2027")
    repeated = estimate_usage(history.copy(), current_season="2026-2027")
    corrected = history.copy()
    corrected.loc[1, "minutes"] = 70
    changed = estimate_usage(corrected, current_season="2026-2027")

    assert first["evidence_fingerprint"] == repeated["evidence_fingerprint"]
    assert first["evidence_fingerprint"] != changed["evidence_fingerprint"]


def test_usage_keeps_double_gameweek_matches_separate_and_excludes_injury() -> None:
    history = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4"],
            "gameweek": [20, 20, 21, 22],
            "kickoff_utc": [
                "2026-01-01T15:00:00Z", "2026-01-04T15:00:00Z",
                "2026-01-11T15:00:00Z", "2026-01-18T15:00:00Z",
            ],
            "started": [True, False, False, False],
            "minutes": [90, 20, 0, 0],
            "availability_status": ["available", "available", "injured", "available"],
        }
    )
    result = estimate_usage(history, prior_observations=0)
    assert result["observations"] == 3
    assert result["start_probability"] == pytest.approx(1 / 3, rel=0.25)


def test_confidence_and_hysteresis_boundaries() -> None:
    confidence = confidence_score(
        eligible_minutes=900,
        required_full_minutes=900,
        eligible_appearances=10,
        required_appearances=10,
        context_coverage=1,
        interval_width=0,
        data_quality=1,
    )
    assert confidence == 1
    assert confidence_band(0.349999) == "Insufficient"
    assert confidence_band(0.35) == "Low"
    assert confidence_band(0.55) == "Medium"
    assert confidence_band(0.75) == "High"

    active, count = hysteresis_state(74.9, was_active=True)
    assert active and count == 1
    active, count = hysteresis_state(74.9, was_active=active, below_exit_updates=count)
    assert not active and count == 2


def test_composite_uses_most_specific_active_vector_and_minimums() -> None:
    components = {
        "GOAL_THREAT": {"active": True, "score": 91, "confidence": 0.82},
        "CREATOR": {"active": True, "score": 84, "confidence": 0.75},
        "DEFENSIVE_ENGINE": {"active": False, "score": 60, "confidence": 0.90},
    }
    result = resolve_production_composite("FWD", components)
    assert result == {
        "id": "PRODSTYLE_FWD_G_C",
        "display_name": "Complete Forward",
        "score": 84.0,
        "confidence": 0.75,
        "components": ["GOAL_THREAT", "CREATOR"],
    }


def test_goalkeeper_composite_never_uses_deferred_components() -> None:
    result = resolve_production_composite(
        "GKP",
        {
            "SAVES_MACHINE": {"active": True, "score": 88, "confidence": 0.8},
            "SWEEPER": {"active": True, "score": 99, "confidence": 1.0},
        },
    )
    assert result is not None
    assert result["display_name"] == "Shot Stopper"
    assert result["components"] == ["SAVES_MACHINE"]


def test_team_ratings_are_immutable_pre_match_and_identifiable() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_id": "m1", "kickoff_utc": "2026-08-01T15:00:00Z",
                "home_team_id": "a", "away_team_id": "b",
                "home_goals": 3, "away_goals": 0, "home_xg": 2.5, "away_xg": 0.4,
            },
            {
                "match_id": "m2", "kickoff_utc": "2026-08-08T15:00:00Z",
                "home_team_id": "b", "away_team_id": "a",
                "home_goals": 0, "away_goals": 1, "home_xg": 0.6, "away_xg": 1.5,
            },
        ]
    )
    snapshots = build_pre_match_ratings(matches)
    first = snapshots[snapshots["match_id"].eq("m1")]
    assert first["overall_elo_pre_match"].tolist() == [1500.0, 1500.0]
    second_a = snapshots[(snapshots["match_id"].eq("m2")) & snapshots["team_id"].eq("a")].iloc[0]
    assert second_a["overall_elo_pre_match"] > 1500
    assert second_a["attack_rating_pre_match"] > 0
    assert math.isfinite(second_a["defence_rating_pre_match"])


def test_team_league_baselines_do_not_use_future_matches() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_id": "m1", "kickoff_utc": "2026-08-01T15:00:00Z",
                "home_team_id": "a", "away_team_id": "b",
                "home_goals": 1, "away_goals": 0, "home_xg": 1.0, "away_xg": 0.5,
            },
            {
                "match_id": "m2", "kickoff_utc": "2026-08-08T15:00:00Z",
                "home_team_id": "b", "away_team_id": "a",
                "home_goals": 9, "away_goals": 9, "home_xg": 9.0, "away_xg": 9.0,
            },
        ]
    )
    first = build_pre_match_ratings(matches)
    changed = matches.copy()
    changed.loc[changed["match_id"].eq("m2"), ["home_goals", "away_goals", "home_xg", "away_xg"]] = 99
    second = build_pre_match_ratings(changed)
    columns = ["match_id", "team_id", "expected_xg_pre_match", "league_xg_baseline_pre_match"]
    pd.testing.assert_frame_equal(
        first[first["match_id"].eq("m1")][columns].reset_index(drop=True),
        second[second["match_id"].eq("m1")][columns].reset_index(drop=True),
    )


def test_promoted_team_initialization_and_season_regression() -> None:
    result = regress_for_new_season(
        {
            "established": TeamRating(elo=1600, attack=0.4, defence=0.2),
            "promoted": TeamRating(elo=1700, attack=1.0, defence=1.0),
        },
        promoted_teams=["promoted"],
    )
    assert result["established"].elo == pytest.approx(1600)
    assert result["established"].attack == pytest.approx(0.28)
    assert result["promoted"] == TeamRating()


def test_build_applies_season_regression_before_new_season_snapshot() -> None:
    matches = pd.DataFrame(
        [
            {"match_id": "m1", "season": "2024-2025", "kickoff_utc": "2025-01-01T15:00:00Z", "home_team_id": "a", "away_team_id": "b", "home_goals": 3, "away_goals": 0, "home_xg": 3.0, "away_xg": 0.2},
            {"match_id": "m2", "season": "2025-2026", "kickoff_utc": "2025-08-01T15:00:00Z", "home_team_id": "a", "away_team_id": "c", "home_goals": 1, "away_goals": 0, "home_xg": 1.2, "away_xg": 0.5},
        ]
    )
    snapshots = build_pre_match_ratings(matches)
    promoted = snapshots[
        snapshots["match_id"].eq("m2") & snapshots["team_id"].eq("c")
    ].iloc[0]
    assert promoted["attack_rating_pre_match"] == pytest.approx(0)
    assert promoted["defence_rating_pre_match"] == pytest.approx(0)
