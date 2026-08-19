import pandas as pd
import plotly.express as px
import streamlit as st

from apps.fpl.state import query_value, switch_page, update_query
from apps.fpl.ui import apply_chart_style, empty_state, file_freshness, inject_global_styles, page_header
from apps.fpl.catalog import (
    FPL_ROOT,
    UNDERSTAT_ROOT,
    discover_leagues as catalog_discover_leagues,
    discover_seasons as catalog_discover_seasons,
    file_version,
    understat_team_season_path,
)


st.set_page_config(page_title="FPL League Insights", page_icon="🏆", layout="wide")


@st.cache_data(show_spinner=False)
def discover_leagues() -> list[str]:
    return [
        league
        for league in catalog_discover_leagues(UNDERSTAT_ROOT)
        if league != "_audit"
    ]


@st.cache_data(show_spinner=False)
def discover_seasons(league: str) -> list[str]:
    return catalog_discover_seasons(
        league, root=UNDERSTAT_ROOT, required_path="team_season.csv"
    )


@st.cache_data(show_spinner=False)
def load_team_season(
    league: str,
    season: str,
    data_version: tuple[int, int] | None = None,
) -> pd.DataFrame:
    del data_version
    path = understat_team_season_path(league, season)
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


def league_table_display(table: pd.DataFrame) -> pd.DataFrame:
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

    return display


def render_table(table: pd.DataFrame, *, key: str = "league_table") -> object:
    return st.dataframe(
        league_table_display(table), width="stretch", hide_index=True,
        on_select="rerun", selection_mode="single-row", key=key,
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
        fig_perf = px.scatter(
            table,
            x="expected_points",
            y="points",
            text="team",
            size="goals_for",
            color="goal_difference",
            color_continuous_scale="Purples",
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
        apply_chart_style(fig_perf, height=max(380, len(table) * 22))
        st.plotly_chart(fig_perf, width="stretch")

    with col2:
        fig_context = px.scatter(
            table,
            x="xg",
            y="xga",
            text="team",
            size="points",
            color="points_minus_xpts",
            color_continuous_scale="BrBG",
            color_continuous_midpoint=0,
            labels={"xg": "Expected goals", "xga": "Expected goals against"},
            title="Attack and defence context",
        )
        fig_context.update_traces(textposition="top center")
        fig_context.update_yaxes(autorange="reversed")
        apply_chart_style(fig_context, height=max(380, len(table) * 22))
        st.plotly_chart(fig_context, width="stretch")


def render_league_tab(league: str, season: str) -> None:
    path = understat_team_season_path(league, season)
    df = load_team_season(league, season, file_version(path))
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
    inject_global_styles()
    fpl_leagues = catalog_discover_leagues(FPL_ROOT)
    leagues = [league for league in fpl_leagues if league in discover_leagues()]
    if not leagues:
        page_header("League", "FPL-focused league context and team performance.")
        empty_state("No league evidence", "No league has both an FPL roster and team-performance data.", icon="⚠️")
        return

    requested_league = query_value("league")
    league_index = leagues.index(requested_league) if requested_league in leagues else 0
    league = st.sidebar.selectbox("League", leagues, index=league_index, key="league_selected")
    seasons = discover_seasons(league)
    if not seasons:
        empty_state("No seasons", f"No team-performance seasons are available for {league}.")
        return
    requested_season = query_value("season")
    season_index = seasons.index(requested_season) if requested_season in seasons else 0
    season = st.sidebar.selectbox("Season", seasons, index=season_index, key="league_season")
    update_query(league=league, season=season)

    path = understat_team_season_path(league, season)
    frame = load_team_season(league, season, file_version(path))
    table = build_league_table(frame)
    page_header(
        "League",
        "Standings and underlying team context connected directly to FPL team decisions.",
        eyebrow=f"{league} · {season}",
        freshness=file_freshness(path),
    )
    if table.empty:
        empty_state("No standings", "The selected season does not contain the required team metrics.")
        return

    leader = table.iloc[0]
    runner_up = table.iloc[1] if len(table) > 1 else None
    best_attack = table.loc[table["xg"].idxmax()]
    best_defence = table.loc[table["xga"].idxmin()]
    overperformer = table.loc[table["points_minus_xpts"].idxmax()]
    metrics = st.columns(4)
    gap = "—" if runner_up is None else f"+{leader['points'] - runner_up['points']:.0f} pts"
    metrics[0].metric("Leader", str(leader["team"]), gap)
    metrics[1].metric("Best attack", str(best_attack["team"]), f"{best_attack['xg']:.1f} xG")
    metrics[2].metric("Best defence", str(best_defence["team"]), f"{best_defence['xga']:.1f} xGA")
    metrics[3].metric(
        "Biggest overperformance", str(overperformer["team"]),
        f"{overperformer['points_minus_xpts']:+.1f} vs xPts",
    )

    view = st.segmented_control(
        "League view", ["Standings", "Underlying performance"], default="Standings",
        label_visibility="collapsed",
    ) or "Standings"
    if view == "Standings":
        st.caption("Select a team row to open its FPL squad and fixture analysis.")
        event = render_table(table, key=f"league_table_{league}_{season}")
        if event.selection.rows:
            selected_team = str(table.iloc[event.selection.rows[0]]["team"])
            switch_page("2_teams.py", league=league, season=season, team=selected_team)
    else:
        render_charts(table)
        st.caption(
            "Lower xGA is better. Chart labels and axes carry the meaning so colour is not the sole signal."
        )


if __name__ == "__main__":
    main()
