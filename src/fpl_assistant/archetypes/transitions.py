from __future__ import annotations

import pandas as pd

from .evidence import confidence_band, hysteresis_state, trend_label


CONTEXT_SENSITIVE_FAMILIES = {
    "Fixture Behaviour", "Venue Behaviour", "Value Historical", "Value Forward", "Production",
}
POSITION_RELATIVE_FAMILIES = {
    "Production Style", "Production Composite", "Production", "Fixture Behaviour",
    "Venue Behaviour", "Return Shape", "Value Historical", "Value Forward", "Risk Badge",
}


def _integer(value: object, default: int = 0) -> int:
    return default if value is None or pd.isna(value) else int(value)


def apply_context_and_transitions(
    current: pd.DataFrame,
    *,
    player_context: pd.DataFrame | None = None,
    previous: pd.DataFrame | None = None,
    entry_score: float = 80.0,
    exit_score: float = 75.0,
) -> pd.DataFrame:
    """Apply transfer/position/absence confidence rules and generic hysteresis."""
    if current.empty:
        return current.copy()
    result = current.copy()
    context_lookup: dict[str, dict[str, object]] = {}
    if player_context is not None and not player_context.empty:
        context_lookup = {
            str(row["player_id"]): row for row in player_context.to_dict("records")
        }
    previous_lookup: dict[tuple[str, str], dict[str, object]] = {}
    if previous is not None and not previous.empty:
        previous_lookup = {
            (str(row["player_id"]), str(row["archetype_id"])): row
            for row in previous.to_dict("records")
        }

    for index, row in result.iterrows():
        player_id = str(row["player_id"])
        family = str(row["family"])
        context = context_lookup.get(player_id, {})
        score = pd.to_numeric(row.get("score_0_100"), errors="coerce")
        confidence = float(row.get("confidence_0_1", 0.0) or 0.0)
        status_override: str | None = None
        score_adjusted = False

        transfer_appearances = int(context.get("appearances_since_transfer", 99) or 0)
        if bool(context.get("transferred", False)) and transfer_appearances < 5:
            if pd.notna(score) and (
                family in CONTEXT_SENSITIVE_FAMILIES
                or family in {"Production Style", "Production Composite"}
            ):
                retention = 0.50 if family in CONTEXT_SENSITIVE_FAMILIES else 0.80
                score = 50.0 + retention * (float(score) - 50.0)
                score_adjusted = True
            confidence *= 0.75
            status_override = "Provisional"

        position_appearances = int(context.get("appearances_since_position_change", 99) or 0)
        position_minutes = float(context.get("minutes_since_position_change", 9999) or 0)
        if (
            bool(context.get("position_changed", False))
            and family in POSITION_RELATIVE_FAMILIES
            and position_appearances < 5
            and position_minutes < 450
        ):
            confidence *= 0.80
            status_override = "Provisional"

        days_absent = float(context.get("days_since_eligible_appearance", 0) or 0)
        unavailable = str(context.get("availability_status", "")).lower() in {
            "injured", "suspended", "unavailable", "illness",
        }
        if days_absent > 30 or unavailable:
            confidence *= 0.75
            if bool(row.get("active_label", False)):
                status_override = "Provisional"

        confidence = max(0.0, min(1.0, confidence))
        result.at[index, "score_0_100"] = score
        result.at[index, "confidence_0_1"] = round(confidence, 6)
        result.at[index, "confidence_band"] = confidence_band(confidence)

        prior = previous_lookup.get((player_id, str(row["archetype_id"])), {})
        was_active = bool(prior.get("active_label", False))
        below = _integer(prior.get("below_exit_updates", 0))
        current_qualifies = bool(row.get("active_label", False))
        if score_adjusted:
            current_qualifies = bool(current_qualifies and float(score) >= entry_score)
        if bool(row.get("transition_applied", False)):
            below_count = _integer(row.get("below_exit_updates", 0))
            active = bool(confidence >= 0.35 and pd.notna(score) and current_qualifies)
        else:
            candidate_active, below_count = hysteresis_state(
                None if pd.isna(score) else float(score),
                was_active=was_active, below_exit_updates=below,
                entry=entry_score, exit=exit_score, required_exit_updates=2,
            )
            active = bool(
                confidence >= 0.35 and pd.notna(score)
                and (current_qualifies or (was_active and candidate_active))
            )
        result.at[index, "active_label"] = active
        result.at[index, "below_exit_updates"] = below_count

        previous_score = pd.to_numeric(prior.get("score_0_100"), errors="coerce")
        delta = float(score - previous_score) if pd.notna(score) and pd.notna(previous_score) else None
        direction = (
            "rising" if delta is not None and delta >= 10 else
            "declining" if delta is not None and delta <= -10 else "stable"
        )
        persistence = _integer(prior.get("trend_persistence", 0)) + 1 if prior.get("trend_direction") == direction else 1
        result.at[index, "trend"] = trend_label(delta, persistent_updates=persistence)
        result.at[index, "trend_direction"] = direction
        result.at[index, "trend_persistence"] = persistence
        if status_override:
            result.at[index, "status"] = status_override
        elif not active and confidence < 0.35:
            result.at[index, "status"] = "Insufficient Evidence"
        elif active and confidence >= 0.70:
            result.at[index, "status"] = "Established"
        elif active:
            result.at[index, "status"] = "Provisional"
    return result


__all__ = ["apply_context_and_transitions"]
