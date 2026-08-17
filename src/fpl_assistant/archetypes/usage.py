from __future__ import annotations

import math

import numpy as np
import pandas as pd


USAGE_STATES = ("Nailed", "Regular Starter", "Rotation Risk", "Impact Sub", "Fringe")


def classify_usage(
    start_probability: float,
    expected_minutes: float,
    cameo_probability: float,
) -> str:
    start = max(0.0, min(1.0, float(start_probability)))
    minutes = max(0.0, min(90.0, float(expected_minutes)))
    cameo = max(0.0, min(1.0, float(cameo_probability)))
    if start >= 0.90 and minutes >= 75:
        return "Nailed"
    if start >= 0.70 and minutes >= 55:
        return "Regular Starter"
    if start >= 0.35 or minutes >= 30:
        return "Rotation Risk"
    if cameo >= 0.35:
        return "Impact Sub"
    return "Fringe"


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid].astype(float), weights=weights.loc[valid]))


def estimate_usage(
    history: pd.DataFrame,
    *,
    recent_team_matches: int = 6,
    half_life_matches: float = 3.0,
    prior_start_probability: float = 0.50,
    prior_cameo_probability: float = 0.30,
    prior_expected_minutes: float = 45.0,
    prior_observations: float = 2.0,
) -> dict[str, float | int | str]:
    required = {"kickoff_utc", "started", "minutes", "availability_status"}
    missing = required - set(history)
    if missing:
        raise KeyError(f"Usage history missing: {sorted(missing)}")
    work = history.copy()
    work["kickoff_utc"] = pd.to_datetime(work["kickoff_utc"], utc=True, errors="raise")
    work = work.sort_values(["kickoff_utc"], kind="stable").tail(recent_team_matches)
    status = work["availability_status"].astype("string").str.lower()
    confirmed_unavailable = status.isin({"injured", "suspended", "unavailable"})
    available = ~confirmed_unavailable
    usable = work.loc[available].copy()
    if usable.empty:
        return {
            "state": "Fringe",
            "start_probability": prior_start_probability,
            "cameo_probability": prior_cameo_probability,
            "expected_minutes": prior_expected_minutes,
            "observations": 0,
        }

    ages = np.arange(len(usable) - 1, -1, -1, dtype=float)
    weights = pd.Series(np.power(0.5, ages / half_life_matches), index=usable.index)
    reasons = usable.get("availability_reason", pd.Series(pd.NA, index=usable.index)).astype("string").str.lower()
    reason_coded = reasons.isin({"injury", "suspension", "illness"})
    weights.loc[reason_coded] *= 0.5
    starts = usable["started"].astype("boolean").astype("Float64")
    minutes = pd.to_numeric(usable["minutes"], errors="coerce").clip(0, 90)
    not_starting = starts.eq(0)
    cameo = minutes.gt(0).where(not_starting)

    observed_weight = float(weights.sum())
    start_observed = _weighted_mean(starts, weights)
    minute_observed = _weighted_mean(minutes, weights)
    cameo_observed = _weighted_mean(cameo.astype("Float64"), weights)
    reliability = observed_weight / (observed_weight + prior_observations)

    def shrunk(observed: float, prior: float) -> float:
        return prior if math.isnan(observed) else reliability * observed + (1 - reliability) * prior

    start_probability = shrunk(start_observed, prior_start_probability)
    expected_minutes = shrunk(minute_observed, prior_expected_minutes)
    cameo_probability = shrunk(cameo_observed, prior_cameo_probability)
    return {
        "state": classify_usage(start_probability, expected_minutes, cameo_probability),
        "start_probability": round(start_probability, 6),
        "cameo_probability": round(cameo_probability, 6),
        "expected_minutes": round(expected_minutes, 3),
        "observations": int(len(usable)),
    }


def stabilize_usage_state(
    proposed_state: str,
    *,
    previous_state: str | None,
    pending_state: str | None = None,
    pending_updates: int = 0,
    immediate_reset: bool = False,
) -> tuple[str, str | None, int]:
    if proposed_state not in USAGE_STATES:
        raise ValueError(f"Unknown usage state: {proposed_state}")
    if previous_state is None or immediate_reset or proposed_state == previous_state:
        return proposed_state, None, 0
    count = pending_updates + 1 if pending_state == proposed_state else 1
    if count >= 2:
        return proposed_state, None, 0
    return previous_state, proposed_state, count


__all__ = ["USAGE_STATES", "classify_usage", "estimate_usage", "stabilize_usage_state"]
