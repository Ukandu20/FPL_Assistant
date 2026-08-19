"""Gameweek-first landing page for the FPL Assistant."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.fpl.catalog import FPL_ROOT, discover_leagues, discover_seasons, fpl_season_path
from apps.fpl.app_data import (
    archetypes,
    forecast,
    previous_archetypes,
    raw_fixtures,
    raw_teams,
    season_players,
)
from apps.fpl.state import query_value, shortlist, switch_page, update_query
from apps.fpl.ui import empty_state, file_freshness, inject_global_styles, page_header
from fpl_assistant.apps.viewmodels.dashboard import (
    archetype_changes,
    enrich_current_players,
    forecast_watchlist,
    fixture_rows,
    player_watchlist,
    team_fixture_outlook,
)


st.set_page_config(page_title="FPL Gameweek Hub", page_icon="⚽", layout="wide")


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

    current = season_players(league, season)
    previous_name = _previous_season(seasons, season)
    previous = season_players(league, previous_name) if previous_name else pd.DataFrame()
    archetype_data, archetype_path = archetypes(season)
    previous_archetype_data, _ = previous_archetypes(season)
    fixtures = raw_fixtures(league, season)
    teams = raw_teams(league, season)
    forecast_data, forecast_path = forecast(season)

    rows = fixture_rows(fixtures, teams)
    unfinished = rows.loc[~rows.get("finished", False).fillna(False).astype(bool)] if not rows.empty else rows
    current_gw = pd.to_numeric(unfinished.get("GW"), errors="coerce").min() if not unfinished.empty else pd.NA
    next_kickoff = unfinished["Kickoff"].min() if not unfinished.empty else pd.NaT
    freshness = file_freshness(fpl_season_path(league, season))
    page_header(
        "Gameweek Hub",
        "The most relevant player, fixture, availability and archetype signals for the selected season.",
        eyebrow="FPL decision centre",
        freshness=freshness,
    )

    headline = st.columns(4)
    headline[0].metric("Current gameweek", "—" if pd.isna(current_gw) else f"GW{int(current_gw)}")
    headline[1].metric("Official deadline", "Not published")
    headline[2].metric(
        "Next kickoff",
        "Not scheduled" if pd.isna(next_kickoff) else pd.Timestamp(next_kickoff).strftime("%d %b · %H:%M UTC"),
    )
    headline[3].metric("Current roster", f"{len(current):,} players")
    covered = archetype_data["player_id"].nunique() if "player_id" in archetype_data else 0
    st.caption(
        f"Archetype coverage: {covered:,} players. The local FPL artifacts do not include an "
        "official deadline field, so the app does not infer one from kickoff time."
    )
    if forecast_path is None:
        st.info(
            "No current-season expected-points forecast is published. Rankings below are clearly "
            "labelled as previous-season baselines and will switch to forecasts automatically."
        )

    enriched = enrich_current_players(current, previous, archetype_data)
    watchlist = (
        forecast_watchlist(enriched, forecast_data, limit=12)
        if not forecast_data.empty
        else player_watchlist(enriched, limit=12)
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
                "now_cost": "Price", "selected_by_percent": "Ownership",
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
                captain_source = watchlist.head(6).reset_index(drop=True)
                captain_display = display.head(6).reset_index(drop=True)
                event = st.dataframe(
                    captain_display.drop(columns=["player_id"], errors="ignore"),
                    hide_index=True, width="stretch", on_select="rerun",
                    selection_mode="single-row", key=f"hub_captains_{season}",
                    column_config=table_config,
                )
                _open_selected_player(event, captain_source, season)
            with transfer_column:
                st.markdown("#### Transfer targets by forecast value")
                transfer_source = watchlist.sort_values("Forecast value", ascending=False).head(6).reset_index(drop=True)
                transfer_display = transfer_source.rename(
                    columns={
                        "name": "Player", "team": "Team", "fpl_pos": "Position",
                        "selected_by_percent": "Ownership",
                    }
                ).copy()
                event = st.dataframe(
                    transfer_display.drop(columns=["player_id"], errors="ignore"),
                    hide_index=True, width="stretch", on_select="rerun",
                    selection_mode="single-row", key=f"hub_transfers_{season}",
                    column_config=table_config,
                )
                _open_selected_player(event, transfer_source, season)
        else:
            event = st.dataframe(
                display.drop(columns=["player_id"], errors="ignore"),
                hide_index=True,
                width="stretch",
                on_select="rerun",
                selection_mode="single-row",
                key=f"hub_watchlist_{season}",
                column_config=table_config,
            )
            _open_selected_player(event, watchlist, season)

    fixture_summary = team_fixture_outlook(rows)
    st.markdown("### Fixture swings")
    if fixture_summary.empty:
        empty_state("No fixture outlook", "Upcoming fixture difficulty is not available.")
    else:
        strong_column, difficult_column = st.columns(2)
        with strong_column:
            st.markdown("#### Best upcoming runs")
            st.dataframe(fixture_summary.head(5), hide_index=True, width="stretch")
        with difficult_column:
            st.markdown("#### Most difficult runs")
            st.dataframe(
                fixture_summary.sort_values("Average FDR", ascending=False).head(5),
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
        st.dataframe(risk_display.head(20), hide_index=True, width="stretch")

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
            changes[[column for column in ["name", "team", "Change", "Archetype", "Family", "Confidence"] if column in changes]],
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
        shortlist_display = shortlisted[[
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
