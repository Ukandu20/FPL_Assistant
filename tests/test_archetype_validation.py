from __future__ import annotations

from dataclasses import asdict
import json

import pandas as pd

from fpl_assistant.archetypes.team_ratings import TeamRatingParameters
from fpl_assistant.archetypes.validation import (
    build_team_validation_predictions,
    select_team_rating_parameters,
    walk_forward_validation_report,
    write_validation_artifacts,
)


def test_walk_forward_release_bar_and_breakdowns() -> None:
    predictions = pd.DataFrame(
        {
            "gameweek": [1, 1, 2, 2, 3, 3, 4, 4],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "candidate": [0.05, 0.95, 0.1, 0.9, 0.05, 0.95, 0.1, 0.9],
            "simple": [0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6],
            "fpl_position": ["MID", "FWD"] * 4,
            "season": ["2024-2025"] * 4 + ["2025-2026"] * 4,
            "confidence_band": ["High"] * 8,
            "evidence_level": ["Full"] * 8,
        }
    )
    report = walk_forward_validation_report(
        predictions,
        target="target", candidate="candidate", baselines=["simple"],
        fold_column="gameweek", metric="brier", calibration_column="simple",
    )
    assert report["passed"]
    assert report["fold_win_rate"] == 1
    assert report["improvement"] > 0.02
    assert set(report["breakdowns"]) == {
        "fpl_position", "season", "confidence_band", "evidence_level"
    }


def test_failed_validation_is_reported_without_rule_mutation() -> None:
    predictions = pd.DataFrame(
        {"fold": [1, 1, 2, 2], "target": [0, 1, 0, 1], "candidate": [0.5] * 4, "baseline": [0.1, 0.9, 0.1, 0.9]}
    )
    report = walk_forward_validation_report(
        predictions, target="target", candidate="candidate", baselines=["baseline"],
        fold_column="fold", metric="brier",
    )
    assert not report["passed"]
    assert report["improvement"] < 0


def test_team_parameter_selection_is_deterministic_and_writable(tmp_path) -> None:
    matches = pd.DataFrame(
        [
            {"match_id": "m1", "kickoff_utc": "2026-01-01T15:00:00Z", "home_team_id": "a", "away_team_id": "b", "home_xg": 2.0, "away_xg": 0.5, "home_goals": 2, "away_goals": 0},
            {"match_id": "m2", "kickoff_utc": "2026-01-08T15:00:00Z", "home_team_id": "b", "away_team_id": "a", "home_xg": 0.7, "away_xg": 1.8, "home_goals": 1, "away_goals": 2},
            {"match_id": "m3", "kickoff_utc": "2026-01-15T15:00:00Z", "home_team_id": "a", "away_team_id": "b", "home_xg": 2.2, "away_xg": 0.6, "home_goals": 3, "away_goals": 0},
        ]
    )
    candidates = [
        TeamRatingParameters(prior_equivalent_matches=4),
        TeamRatingParameters(prior_equivalent_matches=16),
    ]
    selected, results = select_team_rating_parameters(matches, candidates)
    assert len(results) == 2
    assert selected in candidates
    assert results["normalized_loss"].is_monotonic_increasing

    report_path, parameters_path = write_validation_artifacts(
        {"passed": False, "reason": "synthetic smoke test"},
        asdict(selected), output_directory=tmp_path,
    )
    assert json.loads(report_path.read_text())["passed"] is False
    assert "xg_weight" in json.loads(parameters_path.read_text())

    predictions = build_team_validation_predictions(matches, parameters=selected)
    assert len(predictions) == 6
    assert predictions["expected_xg_pre_match"].gt(0).all()
    assert predictions["recent_xg_xga"].notna().all()
