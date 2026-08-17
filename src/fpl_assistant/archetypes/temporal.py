from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


WINDOWS = ("previous_season", "earlier_current", "recent")


def renormalized_window_weights(
    available: Mapping[str, bool],
    base_weights: Mapping[str, float],
) -> dict[str, float]:
    unknown = set(available) - set(WINDOWS)
    if unknown:
        raise ValueError(f"Unknown evidence windows: {sorted(unknown)}")
    selected = {
        window: float(base_weights[window])
        for window in WINDOWS
        if bool(available.get(window, False))
    }
    total = sum(selected.values())
    if total <= 0:
        return {window: 0.0 for window in WINDOWS}
    return {
        window: selected.get(window, 0.0) / total
        for window in WINDOWS
    }


def combine_window_scores(
    scores: Mapping[str, float | None],
    base_weights: Mapping[str, float],
) -> tuple[float | None, dict[str, float]]:
    available = {
        window: scores.get(window) is not None
        and np.isfinite(float(scores[window]))
        for window in WINDOWS
    }
    weights = renormalized_window_weights(available, base_weights)
    if not any(available.values()):
        return None, weights
    return (
        float(sum(float(scores[window]) * weights[window] for window in WINDOWS if available[window])),
        weights,
    )


def assign_evidence_windows(
    observations: pd.DataFrame,
    *,
    current_season: str,
    recent_appearances: int = 10,
    eligible_minutes: int = 30,
) -> pd.Series:
    required = {"season", "kickoff_utc", "minutes"}
    missing = required - set(observations)
    if missing:
        raise KeyError(f"Evidence observations missing: {sorted(missing)}")
    work = observations.copy()
    kickoff = pd.to_datetime(work["kickoff_utc"], utc=True, errors="raise")
    eligible = pd.to_numeric(work["minutes"], errors="coerce").ge(eligible_minutes)
    labels = pd.Series(pd.NA, index=work.index, dtype="string")
    prior_seasons = sorted(
        set(work.loc[eligible, "season"].dropna().astype(str)) - {str(current_season)}
    )
    if prior_seasons:
        labels.loc[eligible & work["season"].astype(str).eq(prior_seasons[-1])] = "previous_season"
    current_mask = eligible & work["season"].astype(str).eq(str(current_season))
    labels.loc[current_mask] = "earlier_current"
    if "player_id" in work:
        current_groups = work.loc[current_mask].groupby("player_id", sort=False)
        recent_indices = [
            index
            for _, group in current_groups
            for index in kickoff.loc[group.index]
            .sort_values(kind="stable")
            .index[-recent_appearances:]
        ]
    else:
        ordered = list(kickoff.loc[current_mask].sort_values(kind="stable").index)
        recent_indices = ordered[-recent_appearances:]
    labels.loc[recent_indices] = "recent"
    return labels


def shrink_toward_mean(
    value: float,
    *,
    evidence_minutes: float,
    position_mean: float,
    prior_minutes: float,
) -> float:
    if evidence_minutes < 0 or prior_minutes < 0:
        raise ValueError("Minutes used for shrinkage cannot be negative")
    denominator = float(evidence_minutes) + float(prior_minutes)
    if denominator == 0:
        return float(position_mean)
    reliability = float(evidence_minutes) / denominator
    return reliability * float(value) + (1 - reliability) * float(position_mean)


__all__ = [
    "WINDOWS",
    "assign_evidence_windows",
    "combine_window_scores",
    "renormalized_window_weights",
    "shrink_toward_mean",
]
