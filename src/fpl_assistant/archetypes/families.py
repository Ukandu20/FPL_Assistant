from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd

from .evidence import confidence_band, confidence_score
from .preprocessing import normalize_position, percentile_within_position, zscore_within_group
from .temporal import assign_evidence_windows, combine_window_scores


@dataclass(frozen=True)
class ContextEffect:
    effect_sd: float
    interval_width: float
    confidence: float
    eligible_appearances: int
    evidence_minutes: float
    context_coverage: float


def temporal_context_effect(
    history: pd.DataFrame,
    *,
    estimator,
    current_season: str,
    as_of: str | pd.Timestamp,
    base_weights: dict[str, float] | None = None,
) -> ContextEffect:
    """Apply TIME-01 to a fixture or venue effect, then keep overall confidence."""
    snapshot = pd.Timestamp(as_of)
    if snapshot.tzinfo is None:
        snapshot = snapshot.tz_localize("UTC")
    work = history.copy()
    work["kickoff_utc"] = pd.to_datetime(work["kickoff_utc"], utc=True, errors="raise")
    work = work[work["kickoff_utc"].lt(snapshot)].copy()
    work["window"] = assign_evidence_windows(
        work, current_season=current_season, recent_appearances=10, eligible_minutes=30
    )
    effects: dict[str, float | None] = {}
    widths: dict[str, float | None] = {}
    for window in ("previous_season", "earlier_current", "recent"):
        sample = work[work["window"].eq(window)]
        if len(sample) < 3:
            effects[window] = None
            widths[window] = None
            continue
        estimate = estimator(sample)
        effects[window] = estimate.effect_sd if math.isfinite(estimate.effect_sd) else None
        widths[window] = estimate.interval_width if math.isfinite(estimate.interval_width) else None
    weights = base_weights or {"previous_season": 0.25, "earlier_current": 0.30, "recent": 0.45}
    effect, applied = combine_window_scores(effects, weights)
    width = sum(
        float(widths[window]) * applied[window]
        for window in applied if widths.get(window) is not None
    )
    overall = estimator(work[work["window"].notna()])
    return ContextEffect(
        float("nan") if effect is None else effect,
        width if effect is not None else float("inf"),
        overall.confidence,
        overall.eligible_appearances,
        overall.evidence_minutes,
        overall.context_coverage,
    )


def _ridge_effect(
    frame: pd.DataFrame,
    *,
    response: str,
    exposure: str,
    controls: Iterable[str] = (),
    prior_strength: float = 8.0,
) -> tuple[float, float]:
    columns = [exposure, *controls]
    clean = frame[[response, *columns]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean) < 3:
        return float("nan"), float("inf")
    y = clean[response].to_numpy(float)
    y_scale = float(np.std(y, ddof=0))
    if y_scale == 0:
        return 0.0, float("inf")
    x_values = clean[columns].to_numpy(float)
    means = x_values.mean(axis=0)
    scales = x_values.std(axis=0)
    scales[scales == 0] = 1.0
    x = np.column_stack([np.ones(len(clean)), (x_values - means) / scales])
    penalty = np.eye(x.shape[1]) * prior_strength
    penalty[0, 0] = 0
    inverse = np.linalg.pinv(x.T @ x + penalty)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    sigma2 = float(np.sum(residual**2) / max(1, len(y) - x.shape[1]))
    standard_error = math.sqrt(max(0.0, sigma2 * inverse[1, 1]))
    return float(beta[1] / y_scale), float(2 * 1.96 * standard_error / y_scale)


def estimate_fixture_effect(
    history: pd.DataFrame,
    *,
    response: str = "production_response",
    difficulty: str = "matchup_difficulty",
    prior_strength: float = 8.0,
) -> ContextEffect:
    required = {response, difficulty, "minutes"}
    missing = required - set(history)
    if missing:
        raise KeyError(f"Fixture history missing: {sorted(missing)}")
    eligible = history[pd.to_numeric(history["minutes"], errors="coerce").ge(30)].copy()
    difficulty_values = pd.to_numeric(eligible[difficulty], errors="coerce")
    q1, q2 = difficulty_values.quantile([1 / 3, 2 / 3]) if len(eligible) else (np.nan, np.nan)
    counts = [int((difficulty_values <= q1).sum()), int(((difficulty_values > q1) & (difficulty_values <= q2)).sum()), int((difficulty_values > q2).sum())]
    coverage = min(1.0, min(counts, default=0) / 8.0)
    span = float(difficulty_values.max() - difficulty_values.min()) if difficulty_values.notna().any() else 0.0
    if span < 0.60:
        coverage *= span / 0.60
    controls = [field for field in ("team_attack", "opponent_defence", "is_home", "expected_minutes") if field in eligible]
    slope, width = _ridge_effect(
        eligible, response=response, exposure=difficulty, controls=controls,
        prior_strength=prior_strength,
    )
    effect = -slope * span if math.isfinite(slope) else float("nan")
    confidence = confidence_score(
        eligible_minutes=float(pd.to_numeric(eligible["minutes"], errors="coerce").sum()),
        required_full_minutes=900,
        eligible_appearances=len(eligible), required_appearances=18,
        context_coverage=coverage, interval_width=width, maximum_useful_width=1.0,
    )
    return ContextEffect(effect, width, confidence, len(eligible), float(eligible["minutes"].sum()), coverage)


def fixture_labels(effects: pd.DataFrame) -> pd.DataFrame:
    required = {"player_id", "fpl_position", "effect_sd", "confidence", "context_coverage"}
    missing = required - set(effects)
    if missing:
        raise KeyError(f"Fixture effects missing: {sorted(missing)}")
    result = effects.copy()
    result["fpl_position"] = normalize_position(result["fpl_position"])
    result["fodder_score"] = percentile_within_position(
        result["effect_sd"], result["fpl_position"], result.get("evidence_minutes", pd.Series(0, index=result.index))
    )
    result["matchup_proof_score"] = percentile_within_position(
        -result["effect_sd"].abs(), result["fpl_position"], result.get("evidence_minutes", pd.Series(0, index=result.index))
    )
    enough = result["confidence"].ge(0.70) & result["context_coverage"].ge(0.75)
    result["fixture_label"] = pd.NA
    fodder = enough & result["effect_sd"].ge(0.50) & result["fodder_score"].ge(80)
    proof = enough & result["effect_sd"].abs().le(0.20) & result["matchup_proof_score"].ge(80)
    result.loc[fodder, "fixture_label"] = "Fodder Hunter"
    result.loc[proof, "fixture_label"] = "Matchup Proof"
    return result


def estimate_venue_effect(
    history: pd.DataFrame,
    *,
    response: str = "production_response",
    prior_strength: float = 8.0,
) -> ContextEffect:
    required = {response, "is_home", "minutes"}
    missing = required - set(history)
    if missing:
        raise KeyError(f"Venue history missing: {sorted(missing)}")
    eligible = history[pd.to_numeric(history["minutes"], errors="coerce").ge(30)].copy()
    home_count = int(eligible["is_home"].fillna(False).astype(bool).sum())
    away_count = len(eligible) - home_count
    coverage = min(1.0, min(home_count, away_count) / 8.0)
    controls = [field for field in ("team_attack", "opponent_defence", "matchup_difficulty", "expected_minutes") if field in eligible]
    effect, width = _ridge_effect(
        eligible, response=response, exposure="is_home", controls=controls,
        prior_strength=prior_strength,
    )
    confidence = confidence_score(
        eligible_minutes=float(pd.to_numeric(eligible["minutes"], errors="coerce").sum()),
        required_full_minutes=900, eligible_appearances=len(eligible), required_appearances=16,
        context_coverage=coverage, interval_width=width, maximum_useful_width=1.0,
    )
    return ContextEffect(effect, width, confidence, len(eligible), float(eligible["minutes"].sum()), coverage)


def venue_label(effect_sd: float, score: float, confidence: float, *, context_coverage: float) -> str | None:
    if confidence < 0.70 or context_coverage < 1.0 or score < 80:
        return None
    if abs(effect_sd) <= 0.20:
        return "Anywhere Threat"
    if effect_sd >= 0.35:
        return "Home Favorite"
    if effect_sd <= -0.35:
        return "Road Warrior"
    return None


def explosive_posterior(
    haul_count: int,
    appearances: int,
    *,
    position_rate: float,
    prior_appearances: float = 10.0,
) -> float:
    if haul_count < 0 or appearances < haul_count:
        raise ValueError("Invalid haul evidence")
    alpha = position_rate * prior_appearances
    beta = (1 - position_rate) * prior_appearances
    return float((haul_count + alpha) / (appearances + alpha + beta))


def return_shape_scores(history: pd.DataFrame) -> pd.DataFrame:
    required = {"player_id", "fpl_position", "minutes", "fpl_points", "return_event"}
    missing = required - set(history)
    if missing:
        raise KeyError(f"Return history missing: {sorted(missing)}")
    work = history[pd.to_numeric(history["minutes"], errors="coerce").ge(30)].copy()
    work["fpl_position"] = normalize_position(work["fpl_position"])
    work["haul"] = pd.to_numeric(work["fpl_points"], errors="coerce").ge(10)
    work["blank"] = ~work["return_event"].fillna(False).astype(bool)
    rows: list[dict[str, object]] = []
    position_rates = work.groupby("fpl_position")["haul"].mean()
    for player_id, group in work.groupby("player_id", sort=False):
        position = str(group["fpl_position"].iloc[-1])
        points = pd.to_numeric(group["fpl_points"], errors="coerce")
        return_rate = float(group["return_event"].fillna(False).astype(bool).mean())
        blank_rate = float(group["blank"].mean())
        downside = float(points[points < points.median()].std(ddof=0)) if len(points) else float("nan")
        rows.append(
            {
                "player_id": player_id, "fpl_position": position,
                "appearances": len(group), "minutes": float(group["minutes"].sum()),
                "hauls": int(group["haul"].sum()),
                "haul_probability": explosive_posterior(
                    int(group["haul"].sum()), len(group), position_rate=float(position_rates[position])
                ),
                "return_rate": return_rate, "blank_rate": blank_rate,
                "downside_variability": downside,
            }
        )
    result = pd.DataFrame(rows)
    groups = result["fpl_position"]
    result["steady_raw"] = (
        0.50 * zscore_within_group(result["return_rate"], groups)
        - 0.30 * zscore_within_group(result["blank_rate"], groups)
        - 0.20 * zscore_within_group(result["downside_variability"], groups)
    )
    result["explosive_score"] = percentile_within_position(
        result["haul_probability"], groups, result["minutes"]
    )
    result["steady_score"] = percentile_within_position(result["steady_raw"], groups, result["minutes"])
    result["explosive_active"] = (
        result["appearances"].ge(15) & result["hauls"].ge(2)
        & result["haul_probability"].ge(0.10) & result["explosive_score"].ge(80)
    )
    production_percentile = percentile_within_position(result["return_rate"], groups, result["minutes"])
    result["steady_active"] = (
        result["appearances"].ge(10) & result["steady_score"].ge(80)
        & production_percentile.ge(60)
    )
    return result


VALUE_LABELS = {
    ("cheap", "high"): "Hidden Gem",
    ("premium", "high"): "Premium Pick",
    ("premium", "low"): "High Maintenance",
    ("cheap", "low"): "Bench Auto-fill",
}


def value_states(players: pd.DataFrame, *, value_column: str) -> pd.DataFrame:
    required = {"player_id", "fpl_position", "price", value_column}
    missing = required - set(players)
    if missing:
        raise KeyError(f"Value inputs missing: {sorted(missing)}")
    result = players.copy()
    result["fpl_position"] = normalize_position(result["fpl_position"])
    result["price_percentile"] = result.groupby("fpl_position")["price"].rank(pct=True, method="average") * 100
    result["value_percentile"] = result.groupby("fpl_position")[value_column].rank(pct=True, method="average") * 100

    def label(row: pd.Series) -> str | None:
        price_band = "cheap" if row["price_percentile"] <= 25 else "premium" if row["price_percentile"] >= 75 else None
        value_band = "high" if row["value_percentile"] >= 80 else "low" if row["value_percentile"] <= 20 else None
        return VALUE_LABELS.get((price_band, value_band))

    result["value_state"] = result.apply(label, axis=1)
    return result


def value_state_hysteresis(
    display_name: str,
    *,
    price_percentile: float,
    value_percentile: float,
    was_active: bool = False,
    exit_updates: int = 0,
) -> tuple[bool, int]:
    if display_name not in VALUE_LABELS.values():
        raise ValueError(f"Unknown value state: {display_name}")
    cheap = display_name in {"Hidden Gem", "Bench Auto-fill"}
    high_value = display_name in {"Hidden Gem", "Premium Pick"}
    price_entry = price_percentile <= 25 if cheap else price_percentile >= 75
    value_entry = value_percentile >= 80 if high_value else value_percentile <= 20
    if not was_active:
        return bool(price_entry and value_entry), 0
    price_exit = price_percentile > 30 if cheap else price_percentile < 70
    value_exit = value_percentile < 75 if high_value else value_percentile > 25
    if price_exit or value_exit:
        count = exit_updates + 1
        return count < 2, count
    return True, 0


def points_hazard_score(history: pd.DataFrame) -> pd.DataFrame:
    required = {"player_id", "fpl_position", "minutes", "yellow_cards", "red_cards"}
    missing = required - set(history)
    if missing:
        raise KeyError(f"Risk inputs missing: {sorted(missing)}")
    work = history.copy()
    for field in ("second_yellow_cards", "expected_suspension_minutes"):
        if field not in work:
            work[field] = 0.0
    rows: list[dict[str, object]] = []
    for player_id, group in work.groupby("player_id", sort=False):
        minutes = float(pd.to_numeric(group["minutes"], errors="coerce").sum())
        deduction = (
            pd.to_numeric(group["yellow_cards"], errors="coerce").sum()
            + 3 * pd.to_numeric(group["second_yellow_cards"], errors="coerce").sum()
            + 3 * pd.to_numeric(group["red_cards"], errors="coerce").sum()
        )
        suspension = pd.to_numeric(group["expected_suspension_minutes"], errors="coerce").sum()
        raw = (float(deduction) + float(suspension) / 90.0) / minutes * 90 if minutes else np.nan
        rows.append({"player_id": player_id, "fpl_position": group["fpl_position"].iloc[-1], "minutes": minutes, "raw": raw})
    result = pd.DataFrame(rows)
    result["score"] = percentile_within_position(result["raw"], result["fpl_position"], result["minutes"])
    result["active"] = result["minutes"].ge(900) & result["score"].ge(85)
    return result


__all__ = [
    "ContextEffect", "VALUE_LABELS", "estimate_fixture_effect", "estimate_venue_effect",
    "explosive_posterior", "fixture_labels", "points_hazard_score",
    "return_shape_scores", "temporal_context_effect", "value_state_hysteresis",
    "value_states", "venue_label",
]
