"""Testable presentation helpers for the Streamlit Player Card."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


FORECAST_WINDOW = re.compile(r"^GW(?P<start>\d+)_(?P<end>\d+)\.(?P<ext>parquet|csv)$")


def latest_forecast_path(root: Path, season: str) -> Path | None:
    """Return the newest GW-windowed forecast for exactly ``season``.

    Parquet is preferred when both formats exist. No cross-season fallback is
    permitted because stale projections are worse than an explicit empty state.
    """
    season_root = root / season
    if not season_root.is_dir():
        return None

    candidates: list[tuple[int, int, int, Path]] = []
    for path in season_root.iterdir():
        match = FORECAST_WINDOW.match(path.name)
        if not match:
            continue
        start = int(match.group("start"))
        end = int(match.group("end"))
        format_preference = 1 if match.group("ext") == "parquet" else 0
        candidates.append((end, start, format_preference, path))
    return max(candidates, default=(0, 0, 0, None))[-1]


def prepare_player_forecast(
    forecasts: pd.DataFrame,
    player_id: str,
    *,
    season: str,
) -> pd.DataFrame:
    """Return one deterministic upcoming-fixture row per player/game/fixture."""
    if forecasts.empty or "player_id" not in forecasts:
        return pd.DataFrame()
    data = forecasts.copy()
    if "season" in data:
        data = data.loc[data["season"].astype("string").eq(str(season))]
    data["player_id"] = data["player_id"].astype("string")
    data = data.loc[data["player_id"].eq(str(player_id))].copy()
    if data.empty:
        return data

    for column in [
        "gw_orig",
        "pred_minutes",
        "p_goal",
        "p_assist",
        "xg_mean",
        "xa_mean",
        "fdr",
        "xPts",
    ]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    identity = [column for column in ["player_id", "gw_orig", "game_id"] if column in data]
    if not identity:
        identity = ["player_id"]
    sort_columns = [column for column in ["gw_orig", "date_sched", "game_id"] if column in data]
    if sort_columns:
        data = data.sort_values(sort_columns, kind="mergesort")
    return data.drop_duplicates(identity, keep="last").reset_index(drop=True)


def profile_dimensions(profile: pd.Series) -> list[dict[str, float | str]]:
    """Return the three display dimensions for an outfielder or goalkeeper."""
    position = str(profile.get("fpl_pos", "")).upper()
    if position in {"GK", "GKP"}:
        dimensions = [
            ("Shot stopping", "shot_stopping_percentile"),
            ("Sweeping", "sweeping_percentile"),
            ("Distribution", "distribution_percentile"),
        ]
    else:
        dimensions = [
            ("Goal threat", "goal_threat_percentile"),
            ("Creativity", "creativity_percentile"),
            ("Defensive activity", "defensive_threat_percentile"),
        ]
    output = []
    for label, column in dimensions:
        value = pd.to_numeric(profile.get(column), errors="coerce")
        if pd.notna(value):
            output.append({"Dimension": label, "Percentile": float(value)})
    return output


def forecast_summary(forecast: pd.DataFrame) -> dict[str, float | int | str]:
    """Aggregate the selected upcoming window for Overview cards."""
    if forecast.empty:
        return {}
    xp = pd.to_numeric(
        forecast.get("xPts", pd.Series(index=forecast.index, dtype="float64")),
        errors="coerce",
    )
    minutes = pd.to_numeric(
        forecast.get(
            "pred_minutes", pd.Series(index=forecast.index, dtype="float64")
        ),
        errors="coerce",
    )
    first = forecast.iloc[0]
    opponent = str(first.get("opponent", "—"))
    home_value = first.get("is_home", False)
    is_home = str(home_value).strip().lower() in {"1", "1.0", "true", "yes"}
    venue = "H" if is_home else "A"
    return {
        "fixtures": int(len(forecast)),
        "expected_points": float(xp.sum(min_count=1)),
        "predicted_minutes": float(minutes.sum(min_count=1)),
        "next_fixture": f"{opponent} ({venue})",
    }


__all__ = [
    "forecast_summary",
    "latest_forecast_path",
    "prepare_player_forecast",
    "profile_dimensions",
]
