"""Discovery and presentation of published minutes-v2 forecasts."""

from pathlib import Path
import re

import pandas as pd
import streamlit as st

from apps.fpl.app_data import read_csv
from apps.fpl.catalog import PREDICTIONS_ROOT, file_version


MINUTES_ROOT = PREDICTIONS_ROOT / "minutes" / "v2"


def forecast_files(season: str, root: Path = MINUTES_ROOT) -> list[Path]:
    """Newest gameweek first; exclude audit files and other seasons."""
    return sorted(
        (p for p in (root / season).glob("GW*.csv")
         if re.fullmatch(r"GW\d+\.csv", p.name)),
        key=lambda p: int(p.stem[2:]), reverse=True,
    )


def load_minutes(season: str, path: Path | None = None) -> tuple[pd.DataFrame, Path | None]:
    paths = forecast_files(season) if path is None else [path]
    if not paths:
        return pd.DataFrame(), None
    path = paths[0]
    data = read_csv(str(path), file_version(path)).copy()
    if "season" in data:
        data = data.loc[data["season"].astype(str).eq(season)].copy()
    data["pred_minutes"] = pd.Series(float("nan"), index=data.index)
    for column in ("pred_exp_minutes_final", "pred_exp_minutes", "pred_exp_minutes_raw"):
        if column in data:
            data["pred_minutes"] = data["pred_minutes"].fillna(
                pd.to_numeric(data[column], errors="coerce")
            )
    return data, path


def minutes_table(data: pd.DataFrame) -> pd.DataFrame:
    """Keep fixture rows distinct and express unconditional probabilities as percentages."""
    labels = {
        "player": "Player", "team": "Team", "pos": "Position",
        "gw_orig": "GW", "date_sched": "Kickoff", "opponent_id": "Opponent ID",
        "was_home": "Venue",
        "pred_minutes": "Expected minutes", "p_start_state": "Start %",
        "p_cameo_state": "Cameo %", "p_dnp_state": "DNP %", "p60_cal": "60+ %",
        "pred_minutes_if_start": "Minutes if starting",
        "pred_minutes_if_cameo": "Minutes if cameo",
        "override_applied": "Override", "override_reason": "Override reason",
    }
    work = data.copy()
    if "was_home" in work:
        work["was_home"] = work["was_home"].astype(str).str.lower().map(
            {"1": "Home", "1.0": "Home", "true": "Home",
             "0": "Away", "0.0": "Away", "false": "Away"}
        )
    if {"team_id", "team"}.issubset(work):
        teams = work.dropna(subset=["team_id", "team"]).drop_duplicates("team_id")
        names = teams.set_index("team_id")["team"]
        if "opponent_id" in work:
            work["opponent_id"] = work["opponent_id"].map(names).fillna(work["opponent_id"])
            labels["opponent_id"] = "Opponent"
    for column in ("p_start_state", "p_cameo_state", "p_dnp_state", "p60_cal"):
        if column in work:
            work[column] = pd.to_numeric(work[column], errors="coerce") * 100
    return work[[c for c in labels if c in work]].rename(columns=labels).round(1)


def add_fixture_minutes(fixtures: pd.DataFrame, data: pd.DataFrame, player_id: str) -> pd.DataFrame:
    """Match by fixture ID, preserving double gameweeks and missing forecasts."""
    if fixtures.empty or not {"player_id", "fpl_id", "pred_minutes"}.issubset(data) or "fpl_id" not in fixtures:
        return fixtures
    selected = data.loc[data["player_id"].astype(str).eq(str(player_id))].copy()
    selected["fixture_key"] = pd.to_numeric(selected["fpl_id"], errors="coerce")
    values = selected.dropna(subset=["fixture_key"]).drop_duplicates("fixture_key").set_index("fixture_key")["pred_minutes"]
    result = fixtures.copy()
    predicted = pd.to_numeric(result["fpl_id"], errors="coerce").map(values)
    result["pred_minutes"] = predicted.combine_first(result.get("pred_minutes", pd.Series(float("nan"), index=result.index)))
    return result


def render_minutes(data: pd.DataFrame, path: Path | None, *, player_id: str | None = None) -> None:
    st.subheader("Minutes forecast")
    table = minutes_table(data)
    if player_id is not None:
        table = table.loc[data["player_id"].astype(str).eq(str(player_id))] if "player_id" in data else table.iloc[:0]
    if table.empty:
        st.info("No published minutes forecast is available for this selection.")
        return
    st.caption(
        f"{path.parent.name} · {path.stem} · Minutes v2 · One row per fixture. "
        "Expected minutes include overrides; probabilities are model estimates."
    )
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        label: st.column_config.NumberColumn(format="%.1f%%")
        for label in ("Start %", "Cameo %", "DNP %", "60+ %")
    })
    if "prediction_cutoff" in data:
        st.caption("Prediction cutoff: " + ", ".join(data["prediction_cutoff"].dropna().astype(str).unique()))
