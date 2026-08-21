"""Team-level FPL and underlying-performance analysis."""

from __future__ import annotations

import importlib

import pandas as pd
import plotly.express as px
import streamlit as st

from apps.fpl.app_data import (
    archetypes,
    raw_fixtures,
    raw_teams,
    season_players,
    set_piece_roles,
)
from apps.fpl.state import (
    comparison,
    query_value,
    set_comparison,
    shortlist,
    switch_page,
    toggle_shortlist,
    update_query,
)
from apps.fpl import ui as fpl_ui
from apps.fpl.catalog import (
    FPL_ROOT,
    discover_leagues,
    discover_seasons,
    file_version,
    fpl_gameweeks_path,
    fpl_season_path,
    understat_team_season_path,
)
from fpl_assistant.apps.viewmodels.dashboard import enrich_current_players, fixture_rows
from fpl_assistant.apps.viewmodels.player_card import team_badge_url


fpl_ui = importlib.reload(fpl_ui)
apply_chart_style = fpl_ui.apply_chart_style
empty_state = fpl_ui.empty_state
file_freshness = fpl_ui.file_freshness
inject_global_styles = fpl_ui.inject_global_styles
page_header = fpl_ui.page_header
style_availability_table = fpl_ui.style_availability_table


st.set_page_config(page_title="FPL Teams", page_icon="🛡️", layout="wide")


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
    inject_global_styles()
    leagues = discover_leagues(FPL_ROOT)
    if not leagues:
        empty_state("No FPL data", f"No processed league data was found in {FPL_ROOT}.", icon="⚠️")
        return

    requested_league = query_value("league")
    league_index = leagues.index(requested_league) if requested_league in leagues else 0
    league = st.sidebar.selectbox("League", leagues, index=league_index, key="team_league")
    seasons = discover_seasons(league)
    if not seasons:
        empty_state("No seasons", f"No FPL seasons are available for {league}.", icon="⚠️")
        return
    requested_season = query_value("season")
    season_index = seasons.index(requested_season) if requested_season in seasons else 0
    season = st.sidebar.selectbox("Season", seasons, index=season_index, key="team_season")

    players_path = fpl_season_path(league, season)
    gameweeks_path = fpl_gameweeks_path(league, season)
    players = season_players(league, season)
    gameweeks = load_csv(str(gameweeks_path), file_version(gameweeks_path))
    if players.empty or "team" not in players.columns:
        empty_state("No roster", f"No player-season data is available for {league} {season}.")
        return

    teams = sorted(players["team"].dropna().astype(str).unique())
    requested_team = query_value("team")
    team_index = teams.index(requested_team) if requested_team in teams else 0
    team = st.sidebar.selectbox("Team", teams, index=team_index, key="team_selected")
    recent_rounds = st.sidebar.slider("Recent form window", 3, 8, 5)
    update_query(league=league, season=season, team=team)

    previous_index = seasons.index(season) + 1
    previous_season = seasons[previous_index] if previous_index < len(seasons) else None
    previous_players = season_players(league, previous_season) if previous_season else pd.DataFrame()
    archetype_data, _ = archetypes(season)
    enriched = enrich_current_players(players, previous_players, archetype_data)
    enriched_team = enriched.loc[enriched["team"].astype(str).eq(team)].copy()

    raw_team_data = raw_teams(league, season)
    team_row = raw_team_data.loc[raw_team_data.get("short_name", pd.Series(dtype="object")).astype(str).eq(team)]
    badge = None
    if not team_row.empty:
        badge = team_badge_url(team_row.iloc[0].get("code"), asset_version="25")
    header_columns = st.columns([1, 8], vertical_alignment="center")
    if badge:
        header_columns[0].image(badge, width=72)
    with header_columns[1]:
        page_header(
            team,
            "Team context, fixture outlook, squad roles and player-level FPL decisions.",
            eyebrow=f"{season} team analysis",
            freshness=file_freshness(players_path),
        )

    understat_path = understat_team_season_path(league, season)
    team_season = load_csv(str(understat_path), file_version(understat_path))
    performance = build_team_performance(team_season, team)
    performance_label = season
    if performance is None and previous_season:
        baseline_path = understat_team_season_path(league, previous_season)
        baseline = load_csv(str(baseline_path), file_version(baseline_path))
        performance = build_team_performance(baseline, team)
        performance_label = f"{previous_season} baseline"
    if performance is None:
        st.info("Team-performance context is not available; squad and fixture analysis remains current.")
    else:
        st.caption(f"Team-performance evidence: {performance_label}")
        first_metrics = st.columns(4)
        second_metrics = st.columns(2)
        values = [
            ("Points", performance.get("points"), 0),
            ("Expected points", performance.get("expected_points"), 1),
            ("Goals", performance.get("goals_for"), 0),
            ("xG", performance.get("xg"), 1),
            ("xGA", performance.get("xga"), 1),
            ("xGD", performance.get("xg_difference"), 1),
        ]
        for container, (label, value, digits) in zip([*first_metrics, *second_metrics], values):
            container.metric(label, format_number(value, digits))

    all_fixture_rows = fixture_rows(raw_fixtures(league, season), raw_team_data)
    team_numeric = pd.to_numeric(enriched_team.get("fpl_team_numeric_id"), errors="coerce").dropna()
    team_fixtures = (
        all_fixture_rows.loc[all_fixture_rows["team_numeric_id"].eq(team_numeric.iloc[0])].head(5)
        if not all_fixture_rows.empty and not team_numeric.empty else pd.DataFrame()
    )
    st.markdown("### Fixture run")
    if team_fixtures.empty:
        empty_state("No upcoming fixtures", "The current fixture feed has no scheduled matches for this team.")
    else:
        fixture_columns = st.columns(min(5, len(team_fixtures)))
        for container, fixture in zip(fixture_columns, team_fixtures.to_dict("records")):
            with container:
                st.metric(
                    f"GW{int(fixture['GW'])} · {fixture['Venue']}",
                    str(fixture["Opponent"]),
                    f"FDR {fixture['FDR']:.0f}",
                )

    st.markdown("### Squad structure")
    structure_columns = st.columns(3)
    for container, (label, column) in zip(
        structure_columns,
        [("Production profiles", "Production profile"), ("Usage states", "Usage"), ("Risk flags", "Risk")],
    ):
        with container:
            st.markdown(f"#### {label}")
            if column not in enriched_team or enriched_team[column].dropna().empty:
                st.caption("No active classifications")
            else:
                counts = enriched_team[column].dropna().value_counts().rename_axis("Profile").reset_index(name="Players")
                st.dataframe(counts, hide_index=True, width="stretch")

    st.markdown("### Set-piece hierarchy")
    roles_season = season
    roles = set_piece_roles(league, season)
    if roles.empty and previous_season:
        roles = set_piece_roles(league, previous_season)
        roles_season = previous_season
    team_roles = roles.loc[roles.get("team", pd.Series(dtype="object")).astype(str).eq(team)].copy()
    if team_roles.empty:
        st.caption("No observed set-piece hierarchy is available for this team.")
    else:
        st.caption(f"Observed taker evidence from {roles_season}; current availability may differ.")
        role_display = team_roles[[
            column for column in [
                "player", "role", "side", "role_rank", "share", "confidence_label", "last_taken"
            ] if column in team_roles
        ]].rename(
            columns={
                "player": "Player", "role": "Role", "side": "Side",
                "role_rank": "Hierarchy", "share": "Share",
                "confidence_label": "Confidence", "last_taken": "Last taken",
            }
        )
        if "Share" in role_display:
            role_display["Share"] = pd.to_numeric(role_display["Share"], errors="coerce").mul(100)
        st.dataframe(
            role_display.head(20), hide_index=True, width="stretch",
            column_config={"Share": st.column_config.NumberColumn(format="%.0f%%")},
        )

    player_table = build_team_player_table(players, gameweeks, team, recent_rounds=recent_rounds)
    if player_table.empty:
        empty_state("No players", "No players were found for this team.")
        return
    live_minutes = numeric_series(player_table, "Minutes").fillna(0)
    has_live = live_minutes.gt(0).any()
    identity = enriched_team[["player_id", "name", *[
        column for column in ["Production profile", "Usage", "Risk", "baseline_total_points", "baseline_minutes"]
        if column in enriched_team
    ]]].rename(columns={"name": "Player"})
    display = player_table.merge(identity, on="Player", how="left", validate="one_to_one")
    if not has_live:
        st.info(
            "This is a preseason roster. Performance columns use the previous-season baseline "
            "where the player has matching historical evidence."
        )
        display["Baseline points"] = pd.to_numeric(display.get("baseline_total_points"), errors="coerce")
        display["Baseline minutes"] = pd.to_numeric(display.get("baseline_minutes"), errors="coerce")
    else:
        recent_points = f"Last {recent_rounds} points"
        leaders = st.columns(4)
        for container, (label, column) in zip(
            leaders,
            [
                ("Season points leader", "Season points"),
                ("Recent form leader", recent_points),
                ("Best value", "Points/£m"),
                ("Minutes leader", "Minutes"),
            ],
        ):
            eligible = display.dropna(subset=[column])
            if eligible.empty:
                container.metric(label, "—")
            else:
                leader = eligible.loc[eligible[column].idxmax()]
                container.metric(label, str(leader["Player"]), format_number(leader[column]))
        chart_data = display.nlargest(12, "Season points").sort_values("Season points")
        figure = px.bar(
            chart_data, x="Season points", y="Player", orientation="h", color="Position",
            hover_data=["Price (£m)", recent_points, "Minutes", "Points/£m"],
            title=f"Top FPL performers for {team}",
        )
        apply_chart_style(figure, height=430)
        st.plotly_chart(figure, width="stretch")

    st.markdown("### Players")
    visible = display.drop(columns=[
        column for column in ["player_id", "baseline_total_points", "baseline_minutes"] if column in display
    ])
    selection = st.dataframe(
        style_availability_table(visible),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=f"team_players_{season}_{team}",
        column_config={
            "Price (£m)": st.column_config.NumberColumn(format="£%.1fm"),
            "xG": st.column_config.NumberColumn(format="%.2f"),
            "xA": st.column_config.NumberColumn(format="%.2f"),
            "Points/£m": st.column_config.NumberColumn(format="%.2f"),
            "Ownership %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    if selection.selection.rows:
        chosen = display.iloc[selection.selection.rows[0]]
        st.caption(f"Selected: {chosen['Player']}")
        actions = st.columns(3)
        if actions[0].button("Open Player Card", type="primary", width="stretch"):
            switch_page("0_main.py", season=season, player=chosen["player_id"], view="Overview")
        if actions[1].button("Add to Compare", width="stretch"):
            set_comparison([*comparison(), chosen["player_id"]])
            st.toast(f"Added {chosen['Player']} to Compare")
        is_shortlisted = str(chosen["player_id"]) in shortlist()
        if actions[2].button(
            "Remove shortlist" if is_shortlisted else "Add shortlist", width="stretch"
        ):
            toggle_shortlist(chosen["player_id"])
            st.rerun()


if __name__ == "__main__":
    main()
