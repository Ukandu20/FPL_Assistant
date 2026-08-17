from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def normalize_position(series: pd.Series) -> pd.Series:
    positions = series.astype("string").str.strip().str.upper()
    return positions.replace({"GK": "GKP"})


def aggregate_rate(
    values: pd.Series,
    minutes: pd.Series,
    *,
    per: float = 90.0,
) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_minutes = pd.to_numeric(minutes, errors="coerce")
    valid = numeric_values.notna() & numeric_minutes.notna() & numeric_minutes.gt(0)
    if not valid.any():
        return float("nan")
    denominator = float(numeric_minutes.loc[valid].sum())
    return float(numeric_values.loc[valid].sum()) / denominator * per


def winsorize_within_group(
    values: pd.Series,
    groups: Iterable[object] | pd.Series,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    group_series = pd.Series(groups, index=values.index)
    result = numeric.copy()
    for _, index in group_series.groupby(group_series, dropna=False).groups.items():
        group = numeric.loc[index]
        if group.notna().any():
            result.loc[index] = group.clip(group.quantile(lower), group.quantile(upper))
    return result


def zscore_within_group(values: pd.Series, groups: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")

    def standardize(group: pd.Series) -> pd.Series:
        deviation = group.std(ddof=0)
        if pd.isna(deviation) or deviation == 0:
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / deviation

    return numeric.groupby(groups, dropna=False).transform(standardize)


def percentile_within_position(
    values: pd.Series,
    positions: pd.Series,
    evidence_minutes: pd.Series,
) -> pd.Series:
    frame = pd.DataFrame(
        {"value": pd.to_numeric(values, errors="coerce"),
         "position": normalize_position(positions),
         "minutes": pd.to_numeric(evidence_minutes, errors="coerce").fillna(0)},
        index=values.index,
    )
    # Stable secondary ordering gives more minutes the higher percentile on ties.
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, index in frame.groupby("position", dropna=False).groups.items():
        group = frame.loc[index].dropna(subset=["value"])
        if group.empty:
            continue
        ordered = group.sort_values(["value", "minutes"], kind="stable")
        if len(ordered) == 1:
            result.loc[ordered.index] = 50.0
        else:
            result.loc[ordered.index] = np.linspace(0.0, 100.0, len(ordered))
    return result


__all__ = [
    "aggregate_rate", "normalize_position", "percentile_within_position",
    "winsorize_within_group", "zscore_within_group",
]
