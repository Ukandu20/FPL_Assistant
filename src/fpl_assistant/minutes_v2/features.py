from __future__ import annotations

import numpy as np
import pandas as pd


START_FEATURES = [
    "min_lag1", "min_ewm_hl2", "start_lag1", "start_rate_hl3",
    "start_streak", "bench_streak", "days_feat", "history_matches", "pos",
]
START_MIN_FEATURES = [
    "min_lag1", "min_ewm_hl2", "start_rate_hl3", "days_feat",
    "history_matches", "pos",
]
CAMEO_FEATURES = [
    "min_lag1", "min_ewm_hl2", "start_rate_hl3", "bench_streak",
    "days_feat", "history_matches", "pos",
]
CAMEO_MIN_FEATURES = [
    "min_lag1", "min_ewm_hl2", "bench_streak", "days_feat",
    "history_matches", "pos",
]
P60_FEATURES = START_FEATURES.copy()
ALL_FEATURES = list(dict.fromkeys(
    START_FEATURES + START_MIN_FEATURES + CAMEO_FEATURES + CAMEO_MIN_FEATURES
    + ["season_history_matches", "cold_start"]
))
FORBIDDEN_CANONICAL_FEATURES = {
    "played_last", "long_gap14", "fdr", "team_rot3", "season_prior",
    "availability", "congestion", "manager_effect", "player_rotation",
}

POSITION_MAP = {
    "GK": "GK", "GKP": "GK", "GOALKEEPER": "GK",
    "DF": "DEF", "DEF": "DEF", "DEFENDER": "DEF",
    "MF": "MID", "MID": "MID", "MIDFIELDER": "MID",
    "FW": "FWD", "FWD": "FWD", "FORWARD": "FWD",
}


def normalize_position(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.upper().str.strip().map(POSITION_MAP)
    if normalized.isna().any():
        unknown = sorted(values[normalized.isna()].dropna().astype(str).unique())
        raise ValueError(f"Unknown positions; refusing ordinal/fallback encoding: {unknown}")
    return pd.Categorical(normalized, categories=["GK", "DEF", "MID", "FWD"])


def _prior_streak(values: pd.Series, observed: pd.Series, wanted: int) -> pd.Series:
    out = np.zeros(len(values), dtype=float)
    run = 0
    for i, (value, is_observed) in enumerate(zip(values, observed, strict=True)):
        out[i] = run
        if is_observed:
            run = run + 1 if value == wanted else 0
    return pd.Series(out, index=values.index)


def build_features(rows: pd.DataFrame, prediction_cutoff: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Build strictly lagged player features.

    Rows at or after ``prediction_cutoff`` may receive features but never
    contribute labels/history. Missing evidence remains NaN; only evidence
    counts and ``cold_start`` describe its absence.
    """
    df = rows.copy()
    required = {"season", "player_id", "date_sched", "minutes", "is_starter", "pos"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Feature input missing required columns: {sorted(missing)}")
    df["date_sched"] = pd.to_datetime(df["date_sched"], utc=True, errors="coerce")
    if df["date_sched"].isna().any():
        raise ValueError("date_sched must be known for every feature row")
    df["pos"] = normalize_position(df["pos"])
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["is_starter"] = pd.to_numeric(df["is_starter"], errors="coerce")
    df = df.sort_values(["date_sched", "player_id", "season"], kind="mergesort").reset_index(drop=True)
    cutoff = pd.to_datetime(prediction_cutoff, utc=True) if prediction_cutoff is not None else None

    feature_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("player_id", sort=False, observed=True):
        g = group.sort_values("date_sched", kind="mergesort").copy()
        observed = g["minutes"].notna() & g["is_starter"].notna()
        if cutoff is not None:
            observed &= g["date_sched"] < cutoff
        hist_minutes = g["minutes"].where(observed)
        hist_starts = g["is_starter"].where(observed)
        g["min_lag1"] = hist_minutes.ffill().shift(1)
        # EWM is calculated on observed history only and shifted before merge.
        g["min_ewm_hl2"] = hist_minutes.shift(1).ewm(halflife=2, adjust=False).mean()
        g["start_lag1"] = hist_starts.ffill().shift(1)
        g["start_rate_hl3"] = hist_starts.shift(1).ewm(halflife=3, adjust=False).mean()
        start_seq = hist_starts.ffill().fillna(-1).astype(int)
        g["start_streak"] = _prior_streak(start_seq, observed, 1)
        g["bench_streak"] = _prior_streak(start_seq, observed, 0)
        previous_date = g["date_sched"].where(observed).ffill().shift(1)
        g["days_feat"] = (g["date_sched"] - previous_date).dt.total_seconds() / 86400.0
        g["history_matches"] = observed.astype(int).cumsum().shift(1, fill_value=0)
        g["season_history_matches"] = (
            observed.groupby(g["season"], sort=False).cumsum() - observed.astype(int)
        )
        g["cold_start"] = (g["season_history_matches"] == 0).astype(int)
        feature_parts.append(g)
    result = pd.concat(feature_parts).sort_index()
    result["pos"] = normalize_position(result["pos"].astype("string"))
    return result


def assert_canonical_features(feature_names: list[str]) -> None:
    forbidden = sorted(set(feature_names) & FORBIDDEN_CANONICAL_FEATURES)
    if forbidden:
        raise ValueError(f"V2.1+ or ablation-only features in canonical model: {forbidden}")
