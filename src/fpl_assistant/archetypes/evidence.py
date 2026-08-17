from __future__ import annotations

from collections.abc import Mapping


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def confidence_score(
    *,
    eligible_minutes: float,
    required_full_minutes: float,
    eligible_appearances: int,
    required_appearances: int,
    context_coverage: float = 1.0,
    interval_width: float = 0.0,
    maximum_useful_width: float = 40.0,
    data_quality: float = 1.0,
    weights: Mapping[str, float] | None = None,
) -> float:
    selected = weights or {
        "minutes": 0.30,
        "appearances": 0.15,
        "context": 0.20,
        "precision": 0.20,
        "data_quality": 0.15,
    }
    minutes = clamp(eligible_minutes / required_full_minutes) if required_full_minutes else 1.0
    appearances = clamp(eligible_appearances / required_appearances) if required_appearances else 1.0
    precision = clamp(1 - interval_width / maximum_useful_width) if maximum_useful_width else 1.0
    value = (
        selected["minutes"] * minutes
        + selected["appearances"] * appearances
        + selected["context"] * clamp(context_coverage)
        + selected["precision"] * precision
        + selected["data_quality"] * clamp(data_quality)
    )
    return round(clamp(value), 6)


def confidence_band(confidence: float) -> str:
    value = clamp(confidence)
    if value < 0.35:
        return "Insufficient"
    if value < 0.55:
        return "Low"
    if value < 0.75:
        return "Medium"
    return "High"


def trend_label(current_delta: float | None, *, persistent_updates: int) -> str | None:
    if current_delta is None or persistent_updates < 2:
        return None
    if current_delta >= 10:
        return "Rising"
    if current_delta <= -10:
        return "Declining"
    if abs(current_delta) <= 5:
        return "Stable"
    return "Emerging"


def status_label(
    *,
    qualifies: bool,
    confidence: float,
    trend: str | None,
    historical_qualified: bool = False,
) -> str:
    if confidence < 0.35:
        return "Insufficient Evidence"
    if qualifies and confidence < 0.70:
        return "Provisional"
    if qualifies and trend == "Declining":
        return "Declining"
    if qualifies and not historical_qualified and trend == "Rising":
        return "Emerging"
    if qualifies and historical_qualified:
        return "Established"
    return "Stable" if trend == "Stable" else "Provisional"


def hysteresis_state(
    score: float | None,
    *,
    was_active: bool,
    below_exit_updates: int = 0,
    entry: float = 80.0,
    exit: float = 75.0,
    required_exit_updates: int = 2,
) -> tuple[bool, int]:
    if score is None:
        return False, 0
    if not was_active:
        return float(score) >= entry, 0
    if float(score) < exit:
        count = below_exit_updates + 1
        return count < required_exit_updates, count
    return True, 0


__all__ = [
    "confidence_band",
    "confidence_score",
    "hysteresis_state",
    "status_label",
    "trend_label",
]
