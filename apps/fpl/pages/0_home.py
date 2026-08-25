"""Gameweek-first landing page for the FPL Assistant."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib

import pandas as pd
import streamlit as st
import streamlit_shadcn_ui as ui

from apps.fpl.catalog import FPL_ROOT, discover_leagues, discover_seasons, fpl_season_path
from apps.fpl.app_data import (
    archetypes,
    forecast,
    gameweeks as season_gameweeks,
    previous_archetypes,
    raw_events,
    raw_fixtures,
    raw_teams,
    season_players,
)
from apps.fpl.state import query_value, shortlist, switch_page, update_query
from apps.fpl import ui as fpl_ui
from fpl_assistant.apps.viewmodels import dashboard as dashboard_viewmodels


# Streamlit can retain imported dependencies while rerunning this page module.
# Reload small presentation dependencies so newly added helpers are available
# without requiring the user to restart the development server.
fpl_ui = importlib.reload(fpl_ui)
empty_state = fpl_ui.empty_state
file_freshness = fpl_ui.file_freshness
inject_global_styles = fpl_ui.inject_global_styles
page_header = fpl_ui.page_header
style_availability_table = fpl_ui.style_availability_table

dashboard_viewmodels = importlib.reload(dashboard_viewmodels)
archetype_changes = dashboard_viewmodels.archetype_changes
enrich_current_players = dashboard_viewmodels.enrich_current_players
forecast_watchlist = dashboard_viewmodels.forecast_watchlist
fixture_rows = dashboard_viewmodels.fixture_rows
gameweek_deadline = dashboard_viewmodels.gameweek_deadline
player_watchlist = dashboard_viewmodels.player_watchlist
stat_leaders = dashboard_viewmodels.stat_leaders
team_fixture_outlook = dashboard_viewmodels.team_fixture_outlook
upcoming_fixture_rows = dashboard_viewmodels.upcoming_fixture_rows


st.set_page_config(page_title="FPL Gameweek Hub", page_icon="⚽", layout="wide")


HUB_ROW_OPTIONS = [5, 10, 20, "All"]


def _saved_option(saved: dict[str, object], name: str, options: list, default: object) -> object:
    """Return a valid persisted control value without trusting stale state."""
    value = saved.get(name, default)
    return value if value in options else default


def _table_rows(frame: pd.DataFrame, row_limit: int | None) -> pd.DataFrame:
    """Apply the hub-wide table limit; ``None`` means show every row."""
    return frame if row_limit is None else frame.head(row_limit)


def _previous_season(seasons: list[str], season: str) -> str | None:
    try:
        index = seasons.index(season)
    except ValueError:
        return None
    return seasons[index + 1] if index + 1 < len(seasons) else None


def _open_selected_player(event: object, source: pd.DataFrame, season: str) -> None:
    rows = getattr(getattr(event, "selection", None), "rows", [])
    if not rows:
        return
    player_id = str(source.iloc[rows[0]]["player_id"])
    switch_page("0_main.py", season=season, player=player_id, view="Overview")


def _deadline_display(deadline: pd.Timestamp) -> tuple[str, str]:
    """Format a UTC deadline and a compact live countdown."""
    display = deadline.strftime("%d %b · %H:%M UTC")
    remaining = deadline.to_pydatetime() - datetime.now(timezone.utc)
    seconds = int(remaining.total_seconds())
    if seconds <= 0:
        return display, "Deadline passed"
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes = remainder // 60
    if days:
        return display, f"Closes in {days}d {hours}h"
    return display, f"Closes in {hours}h {minutes}m"


def main() -> None:
    inject_global_styles()
    leagues = discover_leagues(FPL_ROOT)
    if not leagues:
        page_header("Gameweek Hub", "FPL decision signals in one place.")
        empty_state("No FPL data", f"No processed league folders were found in {FPL_ROOT}.", icon="⚠️")
        return

    requested_league = query_value("league")
    league_index = leagues.index(requested_league) if requested_league in leagues else 0
    league = st.sidebar.selectbox("League", leagues, index=league_index, key="hub_league")
    seasons = discover_seasons(league)
    if not seasons:
        empty_state("No seasons", f"No processed FPL seasons are available for {league}.", icon="⚠️")
        return
    requested_season = query_value("season")
    season_index = seasons.index(requested_season) if requested_season in seasons else 0
    season = st.sidebar.selectbox("Season", seasons, index=season_index, key="hub_season")
    update_query(league=league, season=season)

    saved_filters = st.session_state.get("hub_filter_state", {})
    reset_version = st.session_state.get("hub_filter_reset_version", 0)
    saved_rows = _saved_option(saved_filters, "rows", HUB_ROW_OPTIONS, 10)
    row_choice = st.sidebar.selectbox(
        "Rows per table",
        HUB_ROW_OPTIONS,
        index=HUB_ROW_OPTIONS.index(saved_rows),
        key=f"hub_rows_{reset_version}",
    )
    row_limit = None if row_choice == "All" else int(row_choice)
    if st.sidebar.button("Reset hub filters", key=f"hub_reset_{reset_version}"):
        st.session_state.pop("hub_filter_state", None)
        st.session_state["hub_filter_reset_version"] = reset_version + 1
        st.rerun()

    current = season_players(league, season)
    previous_name = _previous_season(seasons, season)
    previous = season_players(league, previous_name) if previous_name else pd.DataFrame()
    archetype_data, archetype_path = archetypes(season)
    previous_archetype_data, _ = previous_archetypes(season)
    fixtures = raw_fixtures(league, season)
    events = raw_events(league, season)
    teams = raw_teams(league, season)
    gameweek_data = season_gameweeks(league, season)
    forecast_data, forecast_path = forecast(season)

    rows = fixture_rows(fixtures, teams)
    unfinished = upcoming_fixture_rows(rows)
    current_gw = pd.to_numeric(unfinished.get("GW"), errors="coerce").min() if not unfinished.empty else pd.NA
    next_kickoff = unfinished["Kickoff"].min() if not unfinished.empty else pd.NaT
    deadline, deadline_source = gameweek_deadline(events, fixtures, current_gw)
    freshness = file_freshness(fpl_season_path(league, season))
    page_header(
        "Gameweek Hub",
        "The most relevant player, fixture, availability and archetype signals for the selected season.",
        eyebrow="FPL decision centre",
        freshness=freshness,
    )

    headline = st.columns(4)
    with headline[0]:
        ui.metric_card(
            "Current gameweek",
            "—" if pd.isna(current_gw) else f"GW{int(current_gw)}",
            description="Next unfinished round",
            key=f"hub_kpi_gameweek_{season}",
        )
    deadline_label = "Official deadline" if deadline_source == "official" else "Estimated deadline"
    with headline[1]:
        if pd.isna(deadline):
            ui.metric_card(
                "Deadline", "Not scheduled", key=f"hub_kpi_deadline_{season}"
            )
        else:
            deadline_value, deadline_delta = _deadline_display(deadline)
            ui.metric_card(
                deadline_label,
                deadline_value,
                description=deadline_delta,
                key=f"hub_kpi_deadline_{season}",
            )
    with headline[2]:
        ui.metric_card(
            "Next kickoff",
            "Not scheduled"
            if pd.isna(next_kickoff)
            else pd.Timestamp(next_kickoff).strftime("%d %b · %H:%M UTC"),
            description="Scheduled kickoff",
            key=f"hub_kpi_kickoff_{season}",
        )
    with headline[3]:
        ui.metric_card(
            "Current roster",
            f"{len(current):,} players",
            description="Selected-season player pool",
            key=f"hub_kpi_roster_{season}",
        )
    covered = archetype_data["player_id"].nunique() if "player_id" in archetype_data else 0
    deadline_note = {
        "official": "Deadline supplied by the official FPL gameweek calendar.",
        "estimated": "Estimated from the first scheduled kickoff minus the official 90-minute cutoff.",
        "unavailable": "Deadline unavailable because no gameweek calendar or kickoff is published.",
    }[deadline_source]
    st.caption(f"Archetype coverage: {covered:,} players. {deadline_note}")
    if forecast_path is None:
        st.info(
            "No current-season expected-points forecast is published. Rankings below are clearly "
            "labelled as previous-season baselines and will switch to forecasts automatically."
        )

    enriched = enrich_current_players(current, previous, archetype_data)
    result_limit = len(enriched) if row_limit is None else row_limit
    watchlist = (
        forecast_watchlist(enriched, forecast_data, limit=result_limit)
        if not forecast_data.empty
        else player_watchlist(enriched, limit=result_limit)
    )
    watchlist_title = "Decision candidates" if forecast_path else "Preseason watchlist"
    st.markdown(f"### {watchlist_title}")
    if watchlist.empty:
        empty_state("No candidates", "The current roster could not be joined to usable evidence.")
    else:
        has_scaled_price = "Price" in watchlist.columns
        display = watchlist.rename(
            columns={
                "name": "Player", "team": "Team", "fpl_pos": "Position",
                "now_cost": "Price", "selected_by_percent": "Ownership", "status": "Status",
            }
        ).copy()
        display["Price"] = pd.to_numeric(display.get("Price"), errors="coerce")
        if not has_scaled_price:
            display["Price"] = display["Price"].div(10)
        table_config = {
            "Price": st.column_config.NumberColumn(format="£%.1fm"),
            "Ownership": st.column_config.NumberColumn(format="%.1f%%"),
            "Points": st.column_config.NumberColumn(format="%.0f"),
            "Value": st.column_config.NumberColumn(format="%.2f"),
            "Forecast xPts": st.column_config.NumberColumn(format="%.1f"),
            "Forecast value": st.column_config.NumberColumn(format="%.2f"),
        }
        if forecast_path:
            captain_column, transfer_column = st.columns(2)
            with captain_column:
                st.markdown("#### Captain candidates")
                captain_source = _table_rows(watchlist, row_limit).reset_index(drop=True)
                captain_display = _table_rows(display, row_limit).reset_index(drop=True)
                event = st.dataframe(
                    style_availability_table(
                        captain_display.drop(columns=["player_id"], errors="ignore")
                    ),
                    hide_index=True, width="stretch", on_select="rerun",
                    selection_mode="single-row", key=f"hub_captains_{season}",
                    column_config=table_config,
                )
                _open_selected_player(event, captain_source, season)
            with transfer_column:
                st.markdown("#### Transfer targets by forecast value")
                transfer_source = _table_rows(
                    watchlist.sort_values("Forecast value", ascending=False), row_limit
                ).reset_index(drop=True)
                transfer_display = transfer_source.rename(
                    columns={
                        "name": "Player", "team": "Team", "fpl_pos": "Position",
                        "selected_by_percent": "Ownership", "status": "Status",
                    }
                ).copy()
                event = st.dataframe(
                    style_availability_table(
                        transfer_display.drop(columns=["player_id"], errors="ignore")
                    ),
                    hide_index=True, width="stretch", on_select="rerun",
                    selection_mode="single-row", key=f"hub_transfers_{season}",
                    column_config=table_config,
                )
                _open_selected_player(event, transfer_source, season)
        else:
            event = st.dataframe(
                style_availability_table(
                    display.drop(columns=["player_id"], errors="ignore")
                ),
                hide_index=True,
                width="stretch",
                on_select="rerun",
                selection_mode="single-row",
                key=f"hub_watchlist_{season}",
                column_config=table_config,
            )
            _open_selected_player(event, watchlist, season)

    st.markdown("### Stat leaders")
    leader_controls = st.columns([1.35, 1.1, 1.15, 0.8, 1.15])
    metric_options = ["Points", "Points / £m", "Goals", "Assists", "Defensive contributions", "DefCon hit rate"]
    saved_metric = _saved_option(saved_filters, "metric", metric_options, "Points")
    leader_metric = leader_controls[0].selectbox(
        "Metric", metric_options, index=metric_options.index(saved_metric),
        key=f"hub_leader_metric_{season}_{reset_version}",
    )
    basis_options = ["Rate"] if leader_metric == "DefCon hit rate" else ["Total", "Per appearance", "Per 90"]
    saved_basis = _saved_option(saved_filters, "basis", basis_options, basis_options[0])
    leader_basis = leader_controls[1].selectbox(
        "Basis", basis_options, index=basis_options.index(saved_basis),
        key=f"hub_leader_basis_{season}_{reset_version}"
    )
    period_options = ["Season", "Current GW", "Last 5 appearances"]
    saved_period = _saved_option(saved_filters, "period", period_options, "Season")
    leader_period = leader_controls[2].selectbox(
        "Period", period_options, index=period_options.index(saved_period),
        key=f"hub_leader_period_{season}_{reset_version}"
    )
    position_options = ["All", "GKP", "DEF", "MID", "FWD"]
    saved_position = _saved_option(saved_filters, "position", position_options, "All")
    leader_position = leader_controls[3].selectbox(
        "Position", position_options, index=position_options.index(saved_position),
        key=f"hub_leader_position_{season}_{reset_version}"
    )
    sample_options = ["Played", "3 appearances", "5 appearances", "450 minutes"]
    default_sample = "Played" if leader_period == "Current GW" else "5 appearances"
    saved_sample = _saved_option(saved_filters, "sample", sample_options, default_sample)
    sample_label = leader_controls[4].selectbox(
        "Minimum sample",
        sample_options,
        index=sample_options.index(saved_sample),
        key=f"hub_leader_sample_{season}_{reset_version}",
    )
    st.session_state["hub_filter_state"] = {
        "rows": row_choice,
        "metric": leader_metric,
        "basis": leader_basis,
        "period": leader_period,
        "position": leader_position,
        "sample": sample_label,
    }
    min_apps = {"Played": 1, "3 appearances": 3, "5 appearances": 5, "450 minutes": 0}[sample_label]
    min_mins = 450 if sample_label == "450 minutes" else 0
    leaders = stat_leaders(
        current,
        gameweek_data,
        metric=leader_metric,
        basis=leader_basis,
        period=leader_period,
        position=leader_position,
        min_appearances=min_apps,
        min_minutes=min_mins,
        current_gameweek=current_gw,
        limit=result_limit,
    )
    if leaders.empty:
        empty_state(
            "No qualifying performances",
            "Leaderboard data will appear after players meet the selected match and sample criteria.",
        )
    else:
        value_label = leader_metric
        if leader_metric != "DefCon hit rate" and leader_basis != "Total":
            value_label = f"{leader_metric} · {leader_basis}"
        leader_display = leaders.copy()
        if leader_metric == "DefCon hit rate":
            leader_display["Value"] = leader_display["Value"].mul(100)
        leader_display = leader_display.rename(columns={"Value": value_label}).drop(columns="player_id")
        value_format = "%.1f%%" if leader_metric == "DefCon hit rate" else "%.2f"
        event = st.dataframe(
            leader_display,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key=f"hub_leaders_{season}_{leader_metric}_{leader_basis}_{leader_period}_{leader_position}_{sample_label}",
            column_config={
                value_label: st.column_config.NumberColumn(format=value_format),
                "Price": st.column_config.NumberColumn(format="£%.1fm"),
                "Minutes": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        _open_selected_player(event, leaders, season)
    st.caption(
        "Points/£m uses current price. Per-appearance figures count matches with minutes played; "
        "double Gameweeks count as separate matches. DefCon hit rate is the share of appearances "
        "reaching 10 contributions for defenders or 12 for midfielders/forwards; goalkeepers are excluded."
    )

    fixture_summary = team_fixture_outlook(rows)
    st.markdown("### Fixture swings")
    if fixture_summary.empty:
        empty_state("No fixture outlook", "Upcoming fixture difficulty is not available.")
    else:
        strong_column, difficult_column = st.columns(2)
        with strong_column:
            st.markdown("#### Best upcoming runs")
            st.dataframe(_table_rows(fixture_summary, row_limit), hide_index=True, width="stretch")
        with difficult_column:
            st.markdown("#### Most difficult runs")
            st.dataframe(
                _table_rows(fixture_summary.sort_values("Average FDR", ascending=False), row_limit),
                hide_index=True,
                width="stretch",
            )

    st.markdown("### Availability and usage risks")
    if enriched.empty:
        empty_state("No risk data", "Player availability and usage data are unavailable.")
    else:
        status = enriched.get("status", pd.Series("", index=enriched.index)).astype("string").str.lower()
        usage = enriched.get("Usage", pd.Series("", index=enriched.index)).astype("string")
        risk = enriched.get("Risk", pd.Series("", index=enriched.index)).astype("string")
        risk_rows = enriched.loc[
            ~status.isin(["a", "available", ""]) | usage.isin(["Rotation Risk", "Fringe", "Impact Sub"]) | risk.ne("")
        ].copy()
        risk_display = risk_rows[[
            column for column in ["name", "team", "fpl_pos", "status", "Usage", "Risk"] if column in risk_rows
        ]].rename(columns={"name": "Player", "team": "Team", "fpl_pos": "Position", "status": "Status"})
        st.dataframe(
            style_availability_table(_table_rows(risk_display, row_limit)),
            hide_index=True,
            width="stretch",
        )

    changes = archetype_changes(archetype_data, previous_archetype_data, current)
    st.markdown("### Archetype movers")
    if changes.empty:
        empty_state(
            "Awaiting another snapshot",
            "Changes will appear after a second archetype snapshot is published for this season.",
            icon="↔️",
        )
    else:
        st.dataframe(
            _table_rows(changes, row_limit)[[column for column in ["name", "team", "Change", "Archetype", "Family", "Confidence"] if column in changes]],
            hide_index=True,
            width="stretch",
        )

    st.markdown("### Your shortlist")
    ids = shortlist()
    shortlisted = enriched.loc[enriched["player_id"].astype(str).isin(ids)] if ids and not enriched.empty else pd.DataFrame()
    if shortlisted.empty:
        empty_state(
            "No shortlisted players",
            "Add players from a Player Card, then return here to monitor them together.",
            icon="☆",
        )
    else:
        shortlist_display = _table_rows(shortlisted, row_limit)[[
            column for column in ["player_id", "name", "team", "fpl_pos", "now_cost", "Production profile", "Usage", "Risk"] if column in shortlisted
        ]]
        event = st.dataframe(
            shortlist_display.drop(columns="player_id"),
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key=f"hub_shortlist_{season}",
        )
        _open_selected_player(event, shortlist_display, season)

    st.caption(
        "Data status — roster: " + freshness + " · forecast: " + file_freshness(forecast_path)
        + " · archetypes: " + file_freshness(archetype_path)
    )


if __name__ == "__main__":
    main()
