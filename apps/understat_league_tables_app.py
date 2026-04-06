from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Understat League Tables", layout="wide")

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
UNDERSTAT_ROOT = PROJECT_ROOT / "data" / "processed" / "understat"
EXCLUDED_DIRS = {"_audit"}
EPL_LEAGUE_NAME = "ENG-Premier League"
DEFAULT_FIXTURE_TEAMS = ["MUN", "MCI", "ARS"]
RESULT_COLORS = {
    "W": "background-color: #2ecc71; color: #0b1f12;",
    "D": "background-color: #f1c40f; color: #1f1400;",
    "L": "background-color: #e74c3c; color: #2a0b08;",
    "N": "background-color: #f3f4f6; color: #111827;",
    "F": "background-color: #111827; color: #ffffff; font-weight: 700;",
}


@st.cache_data(show_spinner=False)
def discover_leagues() -> list[str]:
    if not UNDERSTAT_ROOT.exists():
        return []
    leagues = [
        d.name
        for d in UNDERSTAT_ROOT.iterdir()
        if d.is_dir() and d.name not in EXCLUDED_DIRS
    ]
    return sorted(leagues)


@st.cache_data(show_spinner=False)
def discover_seasons(league: str) -> list[str]:
    league_dir = UNDERSTAT_ROOT / league
    if not league_dir.exists():
        return []
    seasons = [
        d.name
        for d in league_dir.iterdir()
        if d.is_dir() and (d / "team_season.csv").exists()
    ]
    return sorted(seasons)


@st.cache_data(show_spinner=False)
def load_team_season(league: str, season: str) -> pd.DataFrame:
    path = UNDERSTAT_ROOT / league / season / "team_season.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_schedule(league: str, season: str) -> pd.DataFrame:
    path = UNDERSTAT_ROOT / league / season / "schedule.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_league_table(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "team",
        "matches",
        "wins",
        "draws",
        "losses",
        "goals_for",
        "goals_against",
        "goal_difference",
        "points",
        "expected_points",
        "xg",
        "xga",
        "xg_difference",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return pd.DataFrame()

    table = df.copy()
    numeric_cols = [c for c in required_cols if c != "team"]
    for col in numeric_cols:
        table[col] = pd.to_numeric(table[col], errors="coerce")

    table = table.dropna(subset=["team", "points"]).copy()
    table["points_minus_xpts"] = table["points"] - table["expected_points"]

    table = table.sort_values(
        by=["points", "goal_difference", "goals_for", "xg_difference"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    table["position"] = table.index + 1
    return table


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    norm = series.astype(str).str.strip().str.lower()
    return norm.isin({"true", "1", "yes", "y", "t"})


def build_remaining_fixtures(schedule_df: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    base_cols = ["Team", "GW", "Date", "Time", "Fixture", "Opponent", "Venue"]
    if schedule_df.empty or not teams:
        return pd.DataFrame(columns=base_cols)

    fixtures = schedule_df.copy()

    if "is_result" in fixtures.columns:
        played = _coerce_bool(fixtures["is_result"])
        remaining = fixtures[~played].copy()
    else:
        result_text = fixtures.get("result", pd.Series(index=fixtures.index, dtype=object))
        remaining = fixtures[result_text.isna() | result_text.astype(str).str.strip().eq("")].copy()

    remaining["team"] = remaining["team"].astype(str)
    remaining = remaining[remaining["team"].isin(teams)].copy()
    if remaining.empty:
        return pd.DataFrame(columns=base_cols)

    remaining["GW"] = pd.to_numeric(remaining["round"], errors="coerce").astype("Int64")

    date_dt = pd.to_datetime(remaining.get("game_date"), errors="coerce")
    remaining["Date"] = date_dt.dt.strftime("%Y-%m-%d").fillna("")

    if "game_time" in remaining.columns:
        remaining["Time"] = remaining["game_time"].fillna("").astype(str)
    else:
        remaining["Time"] = ""

    remaining["Opponent"] = remaining["opp"].fillna("-").astype(str)
    remaining["Venue"] = remaining["venue"].fillna("").astype(str).str.strip().str.upper().str[:1]
    remaining["Venue"] = remaining["Venue"].where(remaining["Venue"].isin(["H", "A"]), "?")
    remaining["Fixture"] = remaining["Opponent"] + " (" + remaining["Venue"] + ")"

    remaining["__sort_date"] = date_dt
    remaining["__sort_time"] = pd.to_datetime(
        remaining["Time"], format="%H:%M:%S", errors="coerce"
    )

    remaining = remaining.sort_values(
        by=["team", "GW", "__sort_date", "__sort_time", "Opponent"],
        kind="mergesort",
    )

    out = remaining.rename(columns={"team": "Team"})[base_cols]
    return out.reset_index(drop=True)


def build_remaining_fixtures_wide(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    if fixtures_df.empty:
        return pd.DataFrame()

    wide_source = fixtures_df.copy()
    grouped = (
        wide_source.groupby(["GW", "Team"], dropna=False)["Fixture"]
        .apply(lambda s: " / ".join(s.astype(str)))
        .reset_index()
    )

    wide = grouped.pivot(index="GW", columns="Team", values="Fixture")

    def _gw_sort_key(value: object) -> tuple[int, int]:
        if pd.notna(value):
            try:
                return (0, int(value))
            except Exception:
                pass
        return (1, 999)

    ordered_rows = sorted(list(wide.index), key=_gw_sort_key)
    wide = wide.reindex(index=ordered_rows)
    wide.index = wide.index.map(lambda x: f"GW {int(x)}" if pd.notna(x) else "GW ?")
    wide = wide.fillna("-").reset_index().rename(columns={"index": "GW"})
    return wide


def _extract_remaining_matches(schedule_df: pd.DataFrame) -> list[dict[str, object]]:
    if schedule_df.empty:
        return []
    required = ["game_id", "team", "opp", "venue"]
    if any(col not in schedule_df.columns for col in required):
        return []

    fixtures = schedule_df.copy()
    if "is_result" in fixtures.columns:
        played = _coerce_bool(fixtures["is_result"])
        fixtures = fixtures.loc[~played].copy()
    elif "result" in fixtures.columns:
        result_text = fixtures["result"]
        fixtures = fixtures.loc[result_text.isna() | result_text.astype(str).str.strip().eq("")].copy()

    if fixtures.empty:
        return []

    fixtures["venue"] = fixtures["venue"].fillna("").astype(str).str.upper().str.strip().str[:1]
    fixtures["__date"] = pd.to_datetime(fixtures.get("game_date"), errors="coerce")
    fixtures["__time"] = pd.to_datetime(fixtures.get("game_time"), format="%H:%M:%S", errors="coerce")
    fixtures["__gw"] = pd.to_numeric(fixtures.get("round"), errors="coerce")

    records: list[dict[str, object]] = []
    for game_id, game_rows in fixtures.groupby("game_id", sort=False):
        home_rows = game_rows.loc[game_rows["venue"] == "H"]
        if not home_rows.empty:
            row = home_rows.iloc[0]
            home = str(row["team"])
            away = str(row["opp"])
            p_home = row.get("forecast_win")
            p_draw = row.get("forecast_draw")
            p_away = row.get("forecast_loss")
        else:
            row = game_rows.iloc[0]
            if str(row.get("venue", "")).upper() == "A":
                home = str(row["opp"])
                away = str(row["team"])
                p_home = row.get("forecast_loss")
                p_draw = row.get("forecast_draw")
                p_away = row.get("forecast_win")
            else:
                home = str(row["team"])
                away = str(row["opp"])
                p_home = row.get("forecast_win")
                p_draw = row.get("forecast_draw")
                p_away = row.get("forecast_loss")

        probs = np.array([p_home, p_draw, p_away], dtype=float)
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            probs = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        else:
            probs = probs / probs.sum()

        gw_value = row.get("__gw")
        gw = int(gw_value) if pd.notna(gw_value) else None

        records.append(
            {
                "game_id": int(game_id),
                "gw": gw,
                "date": row.get("__date"),
                "time": row.get("__time"),
                "home": home,
                "away": away,
                "p_home": float(probs[0]),
                "p_draw": float(probs[1]),
                "p_away": float(probs[2]),
            }
        )

    records.sort(
        key=lambda r: (
            999 if r["gw"] is None else int(r["gw"]),
            pd.Timestamp.max if pd.isna(r["date"]) else r["date"],
            pd.Timestamp.max if pd.isna(r["time"]) else r["time"],
            int(r["game_id"]),
        )
    )
    return records


def _build_points_path_table(
    matches: list[dict[str, object]],
    chosen_outcomes: np.ndarray,
    base_points: dict[str, int],
    tracked_teams: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tracked = [team for team in tracked_teams if team in base_points]
    if not tracked:
        return pd.DataFrame(), pd.DataFrame()

    matches_by_gw: dict[object, list[tuple[int, dict[str, object]]]] = {}
    for idx, match in enumerate(matches):
        matches_by_gw.setdefault(match["gw"], []).append((idx, match))

    def _gw_sort_key(gw: object) -> tuple[int, int]:
        if gw is None:
            return (1, 999)
        return (0, int(gw))

    ordered_gws = sorted(matches_by_gw.keys(), key=_gw_sort_key)
    running_points = {team: int(base_points[team]) for team in tracked}

    rows: list[dict[str, object]] = []
    codes: list[dict[str, str]] = []

    for gw in ordered_gws:
        gw_label = f"GW {int(gw)}" if gw is not None else "GW ?"
        row: dict[str, object] = {"GW": gw_label}
        code_row: dict[str, str] = {"GW": ""}
        gains = {team: 0 for team in tracked}
        fixture_labels = {team: [] for team in tracked}

        for match_idx, match in matches_by_gw.get(gw, []):
            outcome = int(chosen_outcomes[match_idx])  # 0=home win, 1=draw, 2=away win
            home = str(match["home"])
            away = str(match["away"])

            if home in gains:
                if outcome == 0:
                    pts = 3
                elif outcome == 1:
                    pts = 1
                else:
                    pts = 0
                gains[home] += pts
                fixture_labels[home].append(f"{away} (H)")

            if away in gains:
                if outcome == 2:
                    pts = 3
                elif outcome == 1:
                    pts = 1
                else:
                    pts = 0
                gains[away] += pts
                fixture_labels[away].append(f"{home} (A)")

        for team in tracked:
            running_points[team] += gains[team]
            if fixture_labels[team]:
                row[team] = f"{' / '.join(fixture_labels[team])} | {running_points[team]}"
                if gains[team] >= 3:
                    code_row[team] = "W"
                elif gains[team] >= 1:
                    code_row[team] = "D"
                else:
                    code_row[team] = "L"
            else:
                row[team] = f"- | {running_points[team]}"
                code_row[team] = "N"

        rows.append(row)
        codes.append(code_row)

    final_row: dict[str, object] = {"GW": "Final Total"}
    final_codes: dict[str, str] = {"GW": "F"}
    for team in tracked:
        final_row[team] = f"{running_points[team]} pts"
        final_codes[team] = "F"
    rows.append(final_row)
    codes.append(final_codes)

    return pd.DataFrame(rows), pd.DataFrame(codes)


@st.cache_data(show_spinner=False)
def run_title_simulation(
    league: str,
    season: str,
    target_team: str,
    tracked_teams: tuple[str, ...],
    simulations: int,
    seed: int,
    min_wins: int = 0,
    max_wins: int = 38,
    min_draws: int = 0,
    max_draws: int = 38,
    min_losses: int = 0,
    max_losses: int = 38,
    min_points_gain: int = 0,
    max_points_gain: int = 114,
) -> dict[str, object]:
    table_df = load_team_season(league, season)
    schedule_df = load_schedule(league, season)
    if table_df.empty or schedule_df.empty:
        return {"error": f"Missing team_season or schedule data for {league} ({season})."}

    if "team" not in table_df.columns or "points" not in table_df.columns:
        return {"error": "team_season.csv is missing required columns (team, points)."}

    base = table_df[["team", "points"]].copy()
    base["team"] = base["team"].astype(str)
    base["points"] = pd.to_numeric(base["points"], errors="coerce")
    base = base.dropna(subset=["team", "points"]).drop_duplicates(subset=["team"], keep="first")
    if base.empty:
        return {"error": "No valid team/points rows found in team_season.csv."}

    teams = base["team"].tolist()
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    base_points_arr = base["points"].astype(np.int16).to_numpy()
    base_points_map = dict(zip(base["team"], base["points"].astype(int)))

    if target_team not in team_to_idx:
        return {"error": f"Target team '{target_team}' is not in the current table."}

    matches = _extract_remaining_matches(schedule_df)
    matches = [m for m in matches if m["home"] in team_to_idx and m["away"] in team_to_idx]
    if not matches:
        return {"error": "No remaining fixtures available to simulate."}

    sims = int(max(1000, simulations))
    rng = np.random.default_rng(int(seed))
    points = np.repeat(base_points_arr[None, :], sims, axis=0).astype(np.int16)
    outcomes = np.empty((len(matches), sims), dtype=np.int8)

    for idx, match in enumerate(matches):
        p_home = float(match["p_home"])
        p_draw = float(match["p_draw"])
        p_away = float(match["p_away"])
        r = rng.random(sims)
        outcome = np.where(r < p_home, 0, np.where(r < p_home + p_draw, 1, 2)).astype(np.int8)
        outcomes[idx] = outcome

        home_idx = team_to_idx[str(match["home"])]
        away_idx = team_to_idx[str(match["away"])]
        points[:, home_idx] += np.where(outcome == 0, 3, np.where(outcome == 1, 1, 0)).astype(np.int16)
        points[:, away_idx] += np.where(outcome == 2, 3, np.where(outcome == 1, 1, 0)).astype(np.int16)

    target_idx = team_to_idx[target_team]
    target_match_meta: list[tuple[int, bool]] = []
    for idx, match in enumerate(matches):
        if str(match["home"]) == target_team:
            target_match_meta.append((idx, True))
        elif str(match["away"]) == target_team:
            target_match_meta.append((idx, False))

    target_remaining_matches = len(target_match_meta)
    cap_wins = max(0, target_remaining_matches)
    cap_draws = max(0, target_remaining_matches)
    cap_losses = max(0, target_remaining_matches)
    cap_points_gain = max(0, target_remaining_matches * 3)

    min_wins = int(max(0, min(min_wins, cap_wins)))
    max_wins = int(max(min_wins, min(max_wins, cap_wins)))
    min_draws = int(max(0, min(min_draws, cap_draws)))
    max_draws = int(max(min_draws, min(max_draws, cap_draws)))
    min_losses = int(max(0, min(min_losses, cap_losses)))
    max_losses = int(max(min_losses, min(max_losses, cap_losses)))
    min_points_gain = int(max(0, min(min_points_gain, cap_points_gain)))
    max_points_gain = int(max(min_points_gain, min(max_points_gain, cap_points_gain)))

    target_wins = np.zeros(sims, dtype=np.int16)
    target_draws = np.zeros(sims, dtype=np.int16)
    target_losses = np.zeros(sims, dtype=np.int16)
    for match_idx, is_home in target_match_meta:
        o = outcomes[match_idx]
        if is_home:
            target_wins += (o == 0).astype(np.int16)
            target_draws += (o == 1).astype(np.int16)
            target_losses += (o == 2).astype(np.int16)
        else:
            target_wins += (o == 2).astype(np.int16)
            target_draws += (o == 1).astype(np.int16)
            target_losses += (o == 0).astype(np.int16)

    target_points_gain = (points[:, target_idx] - int(base_points_map[target_team])).astype(np.int16)
    eligible_mask = (
        (target_wins >= min_wins)
        & (target_wins <= max_wins)
        & (target_draws >= min_draws)
        & (target_draws <= max_draws)
        & (target_losses >= min_losses)
        & (target_losses <= max_losses)
        & (target_points_gain >= min_points_gain)
        & (target_points_gain <= max_points_gain)
    )

    eligible_count = int(eligible_mask.sum())
    if points.shape[1] > 1:
        others_max = np.max(np.delete(points, target_idx, axis=1), axis=1)
    else:
        others_max = np.full(sims, -1, dtype=np.int16)

    outright_first = points[:, target_idx] > others_max
    tied_first = points[:, target_idx] == points.max(axis=1)
    winning_indices = np.flatnonzero(outright_first & eligible_mask)

    result: dict[str, object] = {
        "title_prob_outright_unfiltered": float(outright_first.mean()),
        "title_prob_tied_unfiltered": float(tied_first.mean()),
        "title_prob_outright": float((outright_first & eligible_mask).sum() / eligible_count)
        if eligible_count > 0
        else 0.0,
        "title_prob_tied": float((tied_first & eligible_mask).sum() / eligible_count)
        if eligible_count > 0
        else 0.0,
        "winning_simulations": int(winning_indices.size),
        "simulations": sims,
        "eligible_simulations": eligible_count,
        "eligible_share": float(eligible_count / sims) if sims > 0 else 0.0,
        "base_points": int(base_points_map[target_team]),
        "target_remaining_matches": target_remaining_matches,
        "filters": {
            "wins": [min_wins, max_wins],
            "draws": [min_draws, max_draws],
            "losses": [min_losses, max_losses],
            "points_gain": [min_points_gain, max_points_gain],
        },
        "median_title_points": None,
        "median_title_gain": None,
        "points_path_table": pd.DataFrame(),
        "points_path_codes": pd.DataFrame(),
    }

    if eligible_count == 0 or winning_indices.size == 0:
        return result

    winner_target_points = points[winning_indices, target_idx]
    median_points = float(np.median(winner_target_points))
    rep_pos = int(np.argmin(np.abs(winner_target_points - median_points)))
    representative_idx = int(winning_indices[rep_pos])

    tracked = [team for team in tracked_teams if team in base_points_map]
    if target_team not in tracked:
        tracked = [target_team] + tracked
    dedup_tracked: list[str] = []
    for team in tracked:
        if team not in dedup_tracked:
            dedup_tracked.append(team)

    path_table, path_codes = _build_points_path_table(
        matches,
        outcomes[:, representative_idx],
        base_points_map,
        dedup_tracked,
    )

    result["median_title_points"] = int(round(median_points))
    result["median_title_gain"] = int(round(median_points - base_points_map[target_team]))
    result["points_path_table"] = path_table
    result["points_path_codes"] = path_codes
    return result


def _style_points_path_table(table_df: pd.DataFrame, code_df: pd.DataFrame):
    styles = pd.DataFrame("", index=table_df.index, columns=table_df.columns)
    for row_idx in table_df.index:
        is_final = str(table_df.at[row_idx, "GW"]) == "Final Total"
        if is_final:
            for col in table_df.columns:
                styles.at[row_idx, col] = RESULT_COLORS["F"]
            continue

        styles.at[row_idx, "GW"] = "font-weight: 600;"
        for col in table_df.columns:
            if col == "GW":
                continue
            code = str(code_df.at[row_idx, col]) if col in code_df.columns else ""
            styles.at[row_idx, col] = RESULT_COLORS.get(code, "")

    return table_df.style.apply(lambda _x: styles, axis=None)


def render_table(table: pd.DataFrame) -> None:
    display = pd.DataFrame(
        {
            "Pos": table["position"],
            "Team": table["team"],
            "MP": table["matches"],
            "W": table["wins"],
            "D": table["draws"],
            "L": table["losses"],
            "GF": table["goals_for"],
            "GA": table["goals_against"],
            "GD": table["goal_difference"],
            "Pts": table["points"],
            "xPts": table["expected_points"],
            "Pts-xPts": table["points_minus_xpts"],
            "xG": table["xg"],
            "xGA": table["xga"],
            "xGD": table["xg_difference"],
        }
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "xPts": st.column_config.NumberColumn(format="%.2f"),
            "Pts-xPts": st.column_config.NumberColumn(format="%.2f"),
            "xG": st.column_config.NumberColumn(format="%.2f"),
            "xGA": st.column_config.NumberColumn(format="%.2f"),
            "xGD": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def render_charts(table: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)

    with col1:
        points_chart = table.sort_values("points", ascending=True)
        fig_points = px.bar(
            points_chart,
            x="points",
            y="team",
            orientation="h",
            color="points_minus_xpts",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            labels={"points": "Points", "team": "Team", "points_minus_xpts": "Pts - xPts"},
            title="Points by Team",
        )
        fig_points.update_layout(
            height=max(380, len(points_chart) * 24),
            margin=dict(t=50, b=20, l=20, r=20),
            yaxis_title=None,
        )
        st.plotly_chart(fig_points, use_container_width=True)

    with col2:
        fig_perf = px.scatter(
            table,
            x="expected_points",
            y="points",
            text="team",
            size="goals_for",
            color="goal_difference",
            color_continuous_scale="Blues",
            labels={"expected_points": "Expected Points (xPts)", "points": "Points"},
            title="Points vs Expected Points",
        )
        x_min = float(table["expected_points"].min())
        x_max = float(table["expected_points"].max())
        fig_perf.add_shape(
            type="line",
            x0=x_min,
            y0=x_min,
            x1=x_max,
            y1=x_max,
            line=dict(color="#6b7280", width=1, dash="dash"),
        )
        fig_perf.update_traces(textposition="top center")
        fig_perf.update_layout(
            height=max(380, len(table) * 22),
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_perf, use_container_width=True)


def render_remaining_fixtures_section(league: str, season: str, selected_teams: list[str]) -> None:
    if league != EPL_LEAGUE_NAME:
        return

    st.markdown("### Remaining Fixtures")

    schedule_df = load_schedule(league, season)
    if schedule_df.empty:
        st.info(f"No schedule.csv data found for {league} ({season}).")
        return

    required_base = ["team", "opp", "venue", "round"]
    missing_base = [col for col in required_base if col not in schedule_df.columns]
    has_result_source = "is_result" in schedule_df.columns or "result" in schedule_df.columns

    if missing_base or not has_result_source:
        missing = missing_base.copy()
        if not has_result_source:
            missing.append("is_result/result")
        st.warning(
            "Cannot build remaining fixtures due to missing schedule columns: "
            + ", ".join(missing)
        )
        return

    available_teams = sorted(schedule_df["team"].dropna().astype(str).unique())
    valid_selected_teams = [team for team in selected_teams if team in available_teams]
    dropped = [team for team in selected_teams if team not in available_teams]

    if dropped:
        st.warning("Unavailable team(s) were dropped: " + ", ".join(dropped))

    if not valid_selected_teams:
        st.info("No valid teams selected for remaining fixtures.")
        return

    long_fixtures = build_remaining_fixtures(schedule_df, valid_selected_teams)
    if long_fixtures.empty:
        st.info("No remaining fixtures found for selected teams.")
        return

    wide_fixtures = build_remaining_fixtures_wide(long_fixtures)

    st.markdown("#### Wide by Gameweek")
    st.dataframe(wide_fixtures, use_container_width=True, hide_index=True)

    st.markdown("#### Fixture List")
    st.dataframe(
        long_fixtures[["Team", "GW", "Date", "Time", "Fixture", "Opponent", "Venue"]],
        use_container_width=True,
        hide_index=True,
    )


def render_title_simulation_section(
    league: str,
    season: str,
    target_team: str,
    tracked_teams: list[str],
    simulations: int,
    seed: int,
    min_wins: int,
    max_wins: int,
    min_draws: int,
    max_draws: int,
    min_losses: int,
    max_losses: int,
    min_points_gain: int,
    max_points_gain: int,
) -> None:
    if league != EPL_LEAGUE_NAME:
        return

    st.markdown(f"### Title Simulation: {target_team} to Finish 1st")
    st.caption(
        "Cell format: OPP (H/A) | cumulative points. "
        "Green=win-like GW return (>=3), yellow=draw-like (1-2), red=loss-like (0)."
    )

    with st.spinner("Running title simulation..."):
        sim_result = run_title_simulation(
            league=league,
            season=season,
            target_team=target_team,
            tracked_teams=tuple(tracked_teams),
            simulations=int(simulations),
            seed=int(seed),
            min_wins=int(min_wins),
            max_wins=int(max_wins),
            min_draws=int(min_draws),
            max_draws=int(max_draws),
            min_losses=int(min_losses),
            max_losses=int(max_losses),
            min_points_gain=int(min_points_gain),
            max_points_gain=int(max_points_gain),
        )

    if "error" in sim_result:
        st.warning(str(sim_result["error"]))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Outright 1st Prob", f"{sim_result['title_prob_outright']:.2%}")
    c2.metric("1st or Tied Prob", f"{sim_result['title_prob_tied']:.2%}")
    c3.metric(
        "Eligible Sims",
        f"{sim_result['eligible_simulations']:,}/{sim_result['simulations']:,}",
        f"{sim_result['eligible_share']:.2%}",
    )
    c4.metric(
        "Unfiltered Outright",
        f"{sim_result['title_prob_outright_unfiltered']:.2%}",
    )

    c5, c6 = st.columns(2)
    c5.caption(
        "Filters: "
        f"W {min_wins}-{max_wins}, D {min_draws}-{max_draws}, "
        f"L {min_losses}-{max_losses}, PtsGain {min_points_gain}-{max_points_gain}"
    )
    c6.metric("Winning Sims (Filtered)", f"{sim_result['winning_simulations']:,}")
    if sim_result["median_title_points"] is None:
        st.metric("Median Title Pts", "-")
    else:
        st.metric(
            "Median Title Pts",
            f"{sim_result['median_title_points']}",
            f"+{sim_result['median_title_gain']} from now",
        )

    points_path_table = sim_result["points_path_table"]
    points_path_codes = sim_result["points_path_codes"]
    if sim_result["eligible_simulations"] == 0:
        st.warning("No simulations satisfy the selected filters. Widen the filter ranges.")
        return

    if points_path_table.empty:
        st.info(
            f"No outright-title simulation was found for {target_team} under the current model "
            f"after applying filters ({sim_result['eligible_simulations']:,} eligible runs)."
        )
        return

    st.markdown("#### Simulation Path Table")
    styled_table = _style_points_path_table(points_path_table, points_path_codes)
    st.dataframe(styled_table, use_container_width=True, hide_index=True)


def render_league_tab(league: str, season: str) -> None:
    df = load_team_season(league, season)
    if df.empty:
        st.warning(f"No team_season data found for {league} ({season}).")
        return

    table = build_league_table(df)
    if table.empty:
        st.warning(f"Required table columns are missing for {league} ({season}).")
        return

    leader = table.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Leader", str(leader["team"]))
    c2.metric("Top Points", int(leader["points"]))
    c3.metric("Most Goals", int(table["goals_for"].max()))
    c4.metric("Best xGD", f"{table['xg_difference'].max():.2f}")

    render_table(table)
    render_charts(table)


def main() -> None:
    st.title("Understat League Table Dashboard")
    st.caption("League standings across all processed Understat leagues")

    leagues = discover_leagues()
    if not leagues:
        st.error(f"No leagues found under: {UNDERSTAT_ROOT}")
        return

    st.sidebar.header("Filters")
    selected_leagues = st.sidebar.multiselect("Leagues", options=leagues, default=leagues)
    if not selected_leagues:
        st.info("Select at least one league from the sidebar.")
        return

    season_mode = st.sidebar.radio("Season", options=["Latest per league", "Choose one season"], index=0)

    global_season = None
    if season_mode == "Choose one season":
        all_seasons = sorted({s for lg in selected_leagues for s in discover_seasons(lg)})
        if not all_seasons:
            st.error("No seasons with team_season.csv were found for the selected leagues.")
            return
        global_season = st.sidebar.selectbox("Season to use", options=all_seasons, index=len(all_seasons) - 1)

    selected_fixture_teams: list[str] = []
    simulation_target_team = "MUN"
    simulation_tracked_teams: list[str] = []
    simulation_runs = 50000
    simulation_seed = 42
    sim_min_wins = 0
    sim_max_wins = 38
    sim_min_draws = 0
    sim_max_draws = 38
    sim_min_losses = 0
    sim_max_losses = 38
    sim_min_points_gain = 0
    sim_max_points_gain = 114
    if EPL_LEAGUE_NAME in selected_leagues:
        epl_seasons = discover_seasons(EPL_LEAGUE_NAME)
        epl_selected_season = None

        if epl_seasons:
            if season_mode == "Latest per league":
                epl_selected_season = epl_seasons[-1]
            elif global_season in epl_seasons:
                epl_selected_season = global_season

        if epl_selected_season:
            epl_schedule = load_schedule(EPL_LEAGUE_NAME, epl_selected_season)
            if not epl_schedule.empty and "team" in epl_schedule.columns:
                team_options = sorted(epl_schedule["team"].dropna().astype(str).unique())
                default_teams = [t for t in DEFAULT_FIXTURE_TEAMS if t in team_options]
                missing_defaults = [t for t in DEFAULT_FIXTURE_TEAMS if t not in team_options]

                if missing_defaults:
                    st.sidebar.warning(
                        "Default fixture teams unavailable for this EPL data: "
                        + ", ".join(missing_defaults)
                    )

                if not default_teams and team_options:
                    default_teams = team_options[:3]

                selected_fixture_teams = st.sidebar.multiselect(
                    "Teams for remaining fixtures",
                    options=team_options,
                    default=default_teams,
                )

                st.sidebar.markdown("### Title Simulation")
                target_default = "MUN" if "MUN" in team_options else team_options[0]
                simulation_target_team = st.sidebar.selectbox(
                    "Target team",
                    options=team_options,
                    index=team_options.index(target_default),
                )

                sim_default_teams = [t for t in DEFAULT_FIXTURE_TEAMS if t in team_options]
                if simulation_target_team not in sim_default_teams:
                    sim_default_teams = [simulation_target_team] + sim_default_teams
                if not sim_default_teams:
                    sim_default_teams = [simulation_target_team]

                simulation_tracked_teams = st.sidebar.multiselect(
                    "Teams in simulation table",
                    options=team_options,
                    default=sim_default_teams,
                )
                if simulation_target_team not in simulation_tracked_teams:
                    simulation_tracked_teams = [simulation_target_team] + simulation_tracked_teams

                simulation_runs = st.sidebar.slider(
                    "Simulation runs",
                    min_value=5000,
                    max_value=200000,
                    value=50000,
                    step=5000,
                )
                simulation_seed = int(
                    st.sidebar.number_input(
                        "Simulation seed",
                        min_value=0,
                        max_value=999999,
                        value=42,
                        step=1,
                    )
                )

                remaining_matches = _extract_remaining_matches(epl_schedule)
                target_remaining = sum(
                    1
                    for m in remaining_matches
                    if str(m["home"]) == simulation_target_team or str(m["away"]) == simulation_target_team
                )
                max_gain = target_remaining * 3

                st.sidebar.markdown("#### Simulation Filters")
                sim_min_wins, sim_max_wins = st.sidebar.slider(
                    "Target wins range",
                    min_value=0,
                    max_value=max(0, target_remaining),
                    value=(0, max(0, target_remaining)),
                    step=1,
                )
                sim_min_draws, sim_max_draws = st.sidebar.slider(
                    "Target draws range",
                    min_value=0,
                    max_value=max(0, target_remaining),
                    value=(0, max(0, target_remaining)),
                    step=1,
                )
                sim_min_losses, sim_max_losses = st.sidebar.slider(
                    "Target losses range",
                    min_value=0,
                    max_value=max(0, target_remaining),
                    value=(0, max(0, target_remaining)),
                    step=1,
                )
                sim_min_points_gain, sim_max_points_gain = st.sidebar.slider(
                    "Target points gain range",
                    min_value=0,
                    max_value=max(0, max_gain),
                    value=(0, max(0, max_gain)),
                    step=1,
                )
            else:
                st.sidebar.info("Could not load EPL team options from schedule.csv.")

    tabs = st.tabs(selected_leagues)
    for tab, league in zip(tabs, selected_leagues):
        with tab:
            seasons = discover_seasons(league)
            if not seasons:
                st.warning(f"No seasons found for {league}.")
                continue

            season = seasons[-1] if season_mode == "Latest per league" else global_season
            if season not in seasons:
                st.info(f"{league} has no data for {season}. Available seasons: {', '.join(seasons)}")
                continue

            st.subheader(f"{league} - {season}")
            render_league_tab(league, season)

            if league == EPL_LEAGUE_NAME:
                render_remaining_fixtures_section(league, season, selected_fixture_teams)
                render_title_simulation_section(
                    league=league,
                    season=season,
                    target_team=simulation_target_team,
                    tracked_teams=simulation_tracked_teams,
                    simulations=simulation_runs,
                    seed=simulation_seed,
                    min_wins=sim_min_wins,
                    max_wins=sim_max_wins,
                    min_draws=sim_min_draws,
                    max_draws=sim_max_draws,
                    min_losses=sim_min_losses,
                    max_losses=sim_max_losses,
                    min_points_gain=sim_min_points_gain,
                    max_points_gain=sim_max_points_gain,
                )


if __name__ == "__main__":
    main()
