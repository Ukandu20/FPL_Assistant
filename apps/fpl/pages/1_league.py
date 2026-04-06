from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="League Tables", layout="wide")

UNDERSTAT_ROOT = Path("data/processed/understat")
EXCLUDED_DIRS = {"_audit"}


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
    st.title("League Table Dashboard")
    st.caption("Understat team-season standings across available leagues")

    leagues = discover_leagues()
    if not leagues:
        st.error("No processed Understat league folders were found in data/processed/understat.")
        return

    st.sidebar.header("Filters")
    selected_leagues = st.sidebar.multiselect(
        "Leagues",
        options=leagues,
        default=leagues,
    )
    if not selected_leagues:
        st.info("Select at least one league from the sidebar.")
        return

    season_mode = st.sidebar.radio(
        "Season",
        options=["Latest per league", "Choose one season"],
        index=0,
    )

    global_season = None
    if season_mode == "Choose one season":
        all_seasons = sorted({s for lg in selected_leagues for s in discover_seasons(lg)})
        if not all_seasons:
            st.error("No seasons with team_season.csv were found for the selected leagues.")
            return
        global_season = st.sidebar.selectbox("Season to use", options=all_seasons, index=len(all_seasons) - 1)

    tabs = st.tabs(selected_leagues)
    for tab, league in zip(tabs, selected_leagues):
        with tab:
            seasons = discover_seasons(league)
            if not seasons:
                st.warning(f"No seasons found for {league}.")
                continue

            season = seasons[-1] if season_mode == "Latest per league" else global_season
            if season not in seasons:
                st.info(f"{league} does not have data for {season}. Available: {', '.join(seasons)}")
                continue

            st.subheader(f"{league} - {season}")
            render_league_tab(league, season)


if __name__ == "__main__":
    main()
