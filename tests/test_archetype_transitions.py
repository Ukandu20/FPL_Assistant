from __future__ import annotations

import pandas as pd

from fpl_assistant.archetypes.transitions import apply_context_and_transitions


def _snapshot(score: float = 82, *, active: bool = True, family: str = "Fixture Behaviour") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "p1", "archetype_id": "FODDER_HUNTER",
                "family": family, "score_0_100": score,
                "active_label": active, "confidence_0_1": 0.8,
                "confidence_band": "High", "status": "Established", "trend": None,
            }
        ]
    )


def test_transfer_shrinks_context_score_and_confidence_without_erasing_it() -> None:
    context = pd.DataFrame(
        [{"player_id": "p1", "transferred": True, "appearances_since_transfer": 2}]
    )
    result = apply_context_and_transitions(_snapshot(), player_context=context).iloc[0]
    assert result["score_0_100"] == 66
    assert result["confidence_0_1"] == 0.6
    assert result["status"] == "Provisional"


def test_position_change_reduces_confidence_until_five_appearances_or_450_minutes() -> None:
    context = pd.DataFrame(
        [{
            "player_id": "p1", "position_changed": True,
            "appearances_since_position_change": 3, "minutes_since_position_change": 300,
        }]
    )
    result = apply_context_and_transitions(
        _snapshot(family="Production Style"), player_context=context
    ).iloc[0]
    assert result["confidence_0_1"] == 0.64
    assert result["status"] == "Provisional"


def test_injury_keeps_score_but_lowers_confidence() -> None:
    context = pd.DataFrame(
        [{"player_id": "p1", "availability_status": "injured", "days_since_eligible_appearance": 40}]
    )
    result = apply_context_and_transitions(_snapshot(), player_context=context).iloc[0]
    assert result["score_0_100"] == 82
    assert result["confidence_0_1"] == 0.6


def test_exit_requires_two_consecutive_updates() -> None:
    previous = _snapshot(82)
    previous["below_exit_updates"] = 0
    current = _snapshot(74, active=False)
    first = apply_context_and_transitions(current, previous=previous)
    assert first.iloc[0]["active_label"]
    assert first.iloc[0]["below_exit_updates"] == 1
    second = apply_context_and_transitions(current, previous=first)
    assert not second.iloc[0]["active_label"]
    assert second.iloc[0]["below_exit_updates"] == 2


def test_missing_score_fails_closed_even_when_previous_label_was_active() -> None:
    previous = _snapshot(82)
    current = _snapshot(float("nan"), active=False)
    result = apply_context_and_transitions(current, previous=previous).iloc[0]
    assert not result["active_label"]
