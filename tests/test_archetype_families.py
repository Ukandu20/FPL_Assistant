from __future__ import annotations

import pandas as pd
import pytest

from fpl_assistant.archetypes.adapters import (
    adapt_player_matches,
    understat_team_rows_to_matches,
    validate_provider_ownership,
)
from fpl_assistant.archetypes.families import (
    estimate_fixture_effect,
    estimate_venue_effect,
    explosive_posterior,
    fixture_labels,
    points_hazard_score,
    return_shape_scores,
    temporal_context_effect,
    value_state_hysteresis,
    value_states,
    venue_label,
)


def test_provider_adapter_rejects_cross_provider_definition() -> None:
    frame = pd.DataFrame({"fixture": [1], "element": [2], "understat_npxg": [0.4]})
    with pytest.raises(ValueError, match="owned by another provider"):
        adapt_player_matches(
            frame,
            provider="fpl",
            field_map={"fixture": "match_id", "element": "player_id", "understat_npxg": "npxg"},
        )


def test_provider_adapter_records_field_level_provenance() -> None:
    frame = pd.DataFrame({"fixture": [1], "player": [2], "npxG": [0.4]})
    adapted = adapt_player_matches(
        frame,
        provider="understat",
        field_map={"fixture": "match_id", "player": "player_id", "npxG": "npxg"},
    )
    assert adapted.records.loc[0, "npxg"] == pytest.approx(0.4)
    assert adapted.provenance.loc[0, "source_field"] == "npxG"
    assert validate_provider_ownership(adapted.records, {"npxg": "understat"}) == []


def test_understat_team_rows_become_one_canonical_match() -> None:
    rows = pd.DataFrame(
        [
            {"game_id": 1, "game_date": "2026-01-01", "game_time": "15:00:00", "season": "2025-2026", "round": 1, "team_id": "a", "venue": "H", "goals": 2, "xg": 1.8},
            {"game_id": 1, "game_date": "2026-01-01", "game_time": "15:00:00", "season": "2025-2026", "round": 1, "team_id": "b", "venue": "A", "goals": 1, "xg": 0.7},
        ]
    )
    matches = understat_team_rows_to_matches(rows)
    assert matches.loc[0, "home_team_id"] == "a"
    assert matches.loc[0, "away_xg"] == pytest.approx(0.7)


def _context_history(effect: float, *, fixtures: int = 24) -> pd.DataFrame:
    difficulty = pd.Series([index / (fixtures - 1) for index in range(fixtures)])
    return pd.DataFrame(
        {
            "production_response": 2.0 - effect * difficulty,
            "matchup_difficulty": difficulty,
            "is_home": [index % 2 == 0 for index in range(fixtures)],
            "minutes": [90] * fixtures,
        }
    )


def test_fixture_effect_requires_balanced_context_and_detects_decline() -> None:
    result = estimate_fixture_effect(_context_history(1.0), prior_strength=0)
    assert result.effect_sd > 0.5
    assert result.context_coverage == 1
    labels = fixture_labels(
        pd.DataFrame(
            {
                "player_id": [f"p{i}" for i in range(5)],
                "fpl_position": ["MID"] * 5,
                "effect_sd": [0.0, 0.1, 0.2, 0.4, result.effect_sd],
                "confidence": [0.9] * 5,
                "context_coverage": [1.0] * 5,
                "evidence_minutes": [1800] * 5,
            }
        )
    )
    assert labels.iloc[-1]["fixture_label"] == "Fodder Hunter"


def test_fixture_effect_uses_catalogue_temporal_windows() -> None:
    previous = _context_history(0.2, fixtures=12)
    previous["season"] = "2024-2025"
    previous["kickoff_utc"] = pd.date_range("2024-08-01", periods=12, freq="7D", tz="UTC")
    current = _context_history(1.2, fixtures=12)
    current["season"] = "2025-2026"
    current["kickoff_utc"] = pd.date_range("2025-08-01", periods=12, freq="7D", tz="UTC")
    effect = temporal_context_effect(
        pd.concat([previous, current], ignore_index=True),
        estimator=lambda frame: estimate_fixture_effect(frame, prior_strength=0),
        current_season="2025-2026", as_of="2026-01-01T00:00:00Z",
    )
    assert effect.effect_sd > 0.2


def test_venue_effect_and_label_boundaries() -> None:
    history = _context_history(0.0, fixtures=16)
    history["production_response"] = history["is_home"].astype(float)
    effect = estimate_venue_effect(history, prior_strength=0)
    assert effect.effect_sd > 0.35
    assert effect.context_coverage == 1
    assert venue_label(0.35, 80, 0.70, context_coverage=1) == "Home Favorite"
    assert venue_label(-0.35, 80, 0.70, context_coverage=1) == "Road Warrior"
    assert venue_label(0.20, 80, 0.70, context_coverage=1) == "Anywhere Threat"
    assert venue_label(0.21, 100, 1, context_coverage=1) is None


def test_explosive_posterior_is_shrunk_and_validated() -> None:
    probability = explosive_posterior(2, 15, position_rate=0.08)
    assert 0.08 < probability < 2 / 15
    with pytest.raises(ValueError):
        explosive_posterior(3, 2, position_rate=0.08)


def test_return_shape_labels_can_overlap() -> None:
    rows: list[dict[str, object]] = []
    for player_index in range(5):
        for appearance in range(25):
            high = player_index == 4
            points = 10 if high and appearance % 5 == 0 else 6 if high else player_index
            rows.append(
                {
                    "player_id": f"p{player_index}", "fpl_position": "MID",
                    "minutes": 90, "fpl_points": points,
                    "return_event": points >= 5,
                }
            )
    scores = return_shape_scores(pd.DataFrame(rows)).set_index("player_id")
    assert scores.at["p4", "explosive_active"]
    assert scores.at["p4", "steady_active"]


def test_value_state_matrix_is_exclusive() -> None:
    players = pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(20)],
            "fpl_position": ["MID"] * 20,
            "price": list(range(20)),
            "value": list(reversed(range(20))),
        }
    )
    states = value_states(players, value_column="value").set_index("player_id")
    assert states.at["p0", "value_state"] == "Hidden Gem"
    assert states.at["p19", "value_state"] == "High Maintenance"


def test_value_state_entry_and_exit_boundaries() -> None:
    assert value_state_hysteresis(
        "Hidden Gem", price_percentile=25, value_percentile=80
    ) == (True, 0)
    assert value_state_hysteresis(
        "Hidden Gem", price_percentile=30, value_percentile=75,
        was_active=True,
    ) == (True, 0)
    first = value_state_hysteresis(
        "Hidden Gem", price_percentile=30.1, value_percentile=75,
        was_active=True,
    )
    assert first == (True, 1)
    assert value_state_hysteresis(
        "Hidden Gem", price_percentile=30.1, value_percentile=75,
        was_active=first[0], exit_updates=first[1],
    ) == (False, 2)


def test_points_hazard_uses_top_fifteen_percent_and_full_evidence() -> None:
    history = pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(20)],
            "fpl_position": ["DEF"] * 20,
            "minutes": [900] * 20,
            "yellow_cards": list(range(20)),
            "red_cards": [0] * 20,
        }
    )
    result = points_hazard_score(history).set_index("player_id")
    assert result.at["p19", "active"]
    assert not result.at["p0", "active"]
