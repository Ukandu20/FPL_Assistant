"""Team-level FPL and underlying-performance analysis."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from apps.fpl.catalog import (
    FPL_ROOT,
    discover_leagues,
    discover_seasons,
    file_version,
    fpl_gameweeks_path,
    fpl_season_path,
    understat_team_season_path,
)


st.set_page_config(page_title="FPL Team Analysis", layout="wide")


@st.cache_data(show_spinner=False)
def load_csv(path: str, version: tuple[int, int] | None) -> pd.DataFrame:
    """Load a version-keyed CSV without retaining stale Streamlit cache data."""
    del version
    csv_path = path
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(
        frame.get(column, pd.Series(index=frame.index, dtype="float64")),
        errors="coerce",
    )


def build_team_player_table(
    players: pd.DataFrame,
    gameweeks: pd.DataFrame,
    team: str,
    *,
    recent_rounds: int = 5,
) -> pd.DataFrame:
    """Build one decision-oriented row per player for a selected club."""
    if players.empty or "team" not in players.columns:
        return pd.DataFrame()

    team_players = players.loc[players["team"].astype("string").eq(team)].copy()
    if team_players.empty:
        return pd.DataFrame()

    team_players["player_id"] = team_players["player_id"].astype("string")
    output = pd.DataFrame(
        {
            "player_id": team_players["player_id"],
            "Player": team_players.get("name", ""),
            "Position": team_players.get("fpl_pos", ""),
            "Status": team_players.get("status", ""),
            "Price (£m)": numeric_series(team_players, "now_cost").div(10),
            "Season points": numeric_series(team_players, "total_points"),
            "Minutes": numeric_series(team_players, "minutes"),
            "Goals": numeric_series(team_players, "goals_scored"),
            "Assists": numeric_series(team_players, "assists"),
            "xG": numeric_series(team_players, "xg"),
            "xA": numeric_series(team_players, "xa"),
            "Def Con": numeric_series(team_players, "defcon"),
            "Ownership %": numeric_series(team_players, "selected_by_percent"),
        }
    )

    if not gameweeks.empty and {"team", "player_id", "round"}.issubset(gameweeks):
        recent = gameweeks.loc[
            gameweeks["team"].astype("string").eq(team)
        ].copy()
        recent["round"] = numeric_series(recent, "round")
        latest_round = recent["round"].max()
        if pd.notna(latest_round):
            recent = recent.loc[recent["round"].ge(latest_round - recent_rounds + 1)]
            recent["player_id"] = recent["player_id"].astype("string")
            recent["_points"] = numeric_series(recent, "total_points").fillna(0)
            recent["_minutes"] = numeric_series(recent, "minutes").fillna(0)
            recent_summary = (
                recent.groupby("player_id", as_index=False)
                .agg(
                    **{
                        f"Last {recent_rounds} points": ("_points", "sum"),
                        f"Last {recent_rounds} minutes": ("_minutes", "sum"),
                    }
                )
            )
            output = output.merge(recent_summary, on="player_id", how="left")

    recent_points_column = f"Last {recent_rounds} points"
    recent_minutes_column = f"Last {recent_rounds} minutes"
    for column in (recent_points_column, recent_minutes_column):
        if column not in output:
            output[column] = pd.NA

    output["Points/£m"] = output["Season points"].div(
        output["Price (£m)"].where(output["Price (£m)"].gt(0))
    )
    return (
        output.drop(columns="player_id")
        .sort_values(
            [recent_points_column, "Season points", "Minutes"],
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_team_performance(team_season: pd.DataFrame, team: str) -> pd.Series | None:
    """Return one Understat season record for the selected club."""
    if team_season.empty or "team" not in team_season.columns:
        return None
    records = team_season.loc[team_season["team"].astype("string").eq(team)]
    return None if records.empty else records.iloc[0]


def format_number(value: object, digits: int = 1) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(number) else f"{number:.{digits}f}"


def main() -> None:
    st.title("Team Analysis")
    st.caption(
        "Compare club performance and identify the players producing the most "
        "FPL value and recent returns."
    )

    leagues = discover_leagues(FPL_ROOT)
    if not leagues:
        st.error(f"No FPL league data found in {FPL_ROOT}.")
        return

    league = st.sidebar.selectbox("League", leagues, key="team_league")
    seasons = discover_seasons(league)
    if not seasons:
        st.warning(f"No FPL seasons are available for {league}.")
        return
    season = st.sidebar.selectbox("Season", seasons, key="team_season")

    players_path = fpl_season_path(league, season)
    gameweeks_path = fpl_gameweeks_path(league, season)
    understat_path = understat_team_season_path(league, season)
    players = load_csv(str(players_path), file_version(players_path))
    gameweeks = load_csv(str(gameweeks_path), file_version(gameweeks_path))
    team_season = load_csv(str(understat_path), file_version(understat_path))

    if players.empty or "team" not in players.columns:
        st.warning(f"No player-season data is available for {league} {season}.")
        return

    teams = sorted(players["team"].dropna().astype(str).unique())
    team = st.sidebar.selectbox("Team", teams, key="team_selected")
    recent_rounds = st.sidebar.slider("Recent form window", 3, 8, 5)

    performance = build_team_performance(team_season, team)
    st.subheader(f"{team} · {season}")
    if performance is None:
        st.info(
            "Underlying team-season performance is not available for this "
            "season yet. The player analysis below uses the current FPL roster."
        )
    else:
        metrics = st.columns(6)
        metrics[0].metric("Points", format_number(performance.get("points"), 0))
        metrics[1].metric("Expected points", format_number(performance.get("expected_points")))
        metrics[2].metric("Goals", format_number(performance.get("goals_for"), 0))
        metrics[3].metric("xG", format_number(performance.get("xg")))
        metrics[4].metric("xGA", format_number(performance.get("xga")))
        metrics[5].metric("xGD", format_number(performance.get("xg_difference")))

    player_table = build_team_player_table(
        players, gameweeks, team, recent_rounds=recent_rounds
    )
    if player_table.empty:
        st.warning("No players were found for this team.")
        return

    has_played_performance = bool(
        numeric_series(player_table, "Minutes").fillna(0).gt(0).any()
        or numeric_series(player_table, f"Last {recent_rounds} minutes")
        .fillna(0)
        .gt(0)
        .any()
    )
    if not has_played_performance:
        st.info(
            "This is a preseason roster: no played minutes or FPL returns are "
            "available yet. Select the previous season to compare performance."
        )
        st.subheader("Current roster")
        st.dataframe(
            player_table[["Player", "Position", "Status", "Price (£m)", "Ownership %"]],
            hide_index=True,
            width="stretch",
            column_config={
                "Price (£m)": st.column_config.NumberColumn(format="£%.1fm"),
                "Ownership %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        return

    recent_points = f"Last {recent_rounds} points"
    leader_columns = st.columns(4)
    leaders = (
        ("Season points leader", "Season points"),
        ("Recent form leader", recent_points),
        ("Best value", "Points/£m"),
        ("Minutes leader", "Minutes"),
    )
    for container, (label, column) in zip(leader_columns, leaders):
        eligible = player_table.dropna(subset=[column])
        if eligible.empty:
            container.metric(label, "—")
        else:
            leader = eligible.loc[eligible[column].idxmax()]
            container.metric(label, str(leader["Player"]), format_number(leader[column]))

    chart_data = player_table.nlargest(12, "Season points").sort_values("Season points")
    figure = px.bar(
        chart_data,
        x="Season points",
        y="Player",
        orientation="h",
        color="Position",
        hover_data=["Price (£m)", recent_points, "Minutes", "Points/£m"],
        title=f"Top FPL performers for {team}",
    )
    figure.update_layout(height=430, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(figure, width="stretch")

    st.subheader("Player performance")
    st.dataframe(
        player_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Price (£m)": st.column_config.NumberColumn(format="£%.1fm"),
            "xG": st.column_config.NumberColumn(format="%.2f"),
            "xA": st.column_config.NumberColumn(format="%.2f"),
            "Points/£m": st.column_config.NumberColumn(format="%.2f"),
            "Ownership %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )


if __name__ == "__main__":
    main()
