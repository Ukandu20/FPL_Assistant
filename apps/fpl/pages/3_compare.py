"""Persistent, shareable comparison for two to four FPL players."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from apps.fpl.catalog import FPL_ROOT, discover_leagues, discover_seasons, fpl_season_path
from apps.fpl.app_data import archetypes, forecast, raw_fixtures, raw_teams, season_players
from apps.fpl.state import comparison, query_value, set_comparison, shortlist, switch_page, toggle_shortlist, update_query
from apps.fpl.ui import apply_chart_style, empty_state, file_freshness, inject_global_styles, page_header
from fpl_assistant.apps.viewmodels.dashboard import comparison_table, enrich_current_players, fixture_rows


st.set_page_config(page_title="Compare FPL Players", page_icon="⚖️", layout="wide")


def _previous_season(seasons: list[str], season: str) -> str | None:
    index = seasons.index(season)
    return seasons[index + 1] if index + 1 < len(seasons) else None


def _forecast_summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty or "player_id" not in forecasts:
        return pd.DataFrame(columns=["player_id"])
    work = forecasts.copy()
    work["player_id"] = work["player_id"].astype(str)
    for column in ["xPts", "pred_minutes"]:
        work[column] = pd.to_numeric(work.get(column), errors="coerce")
    return work.groupby("player_id", as_index=False).agg(
        **{"Forecast xPts": ("xPts", "sum"), "Forecast minutes": ("pred_minutes", "sum")}
    )


def main() -> None:
    inject_global_styles()
    leagues = discover_leagues(FPL_ROOT)
    if not leagues:
        empty_state("No FPL data", "Player comparison requires a processed roster.", icon="⚠️")
        return
    league = query_value("league")
    if league not in leagues:
        league = leagues[0]
    seasons = discover_seasons(league)
    requested_season = query_value("season")
    season = requested_season if requested_season in seasons else seasons[0]
    season = st.sidebar.selectbox("Season", seasons, index=seasons.index(season), key="compare_season")
    update_query(league=league, season=season)

    current = season_players(league, season)
    previous_name = _previous_season(seasons, season)
    previous = season_players(league, previous_name) if previous_name else pd.DataFrame()
    archetype_data, _ = archetypes(season)
    enriched = enrich_current_players(current, previous, archetype_data)
    forecasts, forecast_path = forecast(season)

    page_header(
        "Compare players",
        "Compare exact values, roles, fixtures and risks without forcing one universal winner.",
        eyebrow="FPL decision centre",
        freshness=file_freshness(fpl_season_path(league, season)),
    )
    if enriched.empty:
        empty_state("No players", "No current roster is available for comparison.")
        return

    enriched["player_id"] = enriched["player_id"].astype(str)
    labels = {
        str(row.player_id): f"{row.name} · {row.team} · {row.fpl_pos}"
        for row in enriched[["player_id", "name", "team", "fpl_pos"]].itertuples(index=False)
    }
    selected_ids = [value for value in comparison() if value in labels]
    same_position = st.toggle("Same position only", value=True)
    option_ids = list(labels)
    if same_position and selected_ids:
        first_position = enriched.loc[enriched["player_id"].eq(selected_ids[0]), "fpl_pos"].iloc[0]
        option_ids = enriched.loc[enriched["fpl_pos"].eq(first_position), "player_id"].astype(str).tolist()
        selected_ids = [value for value in selected_ids if value in option_ids]
    chosen = st.multiselect(
        "Players to compare",
        option_ids,
        default=selected_ids,
        max_selections=4,
        format_func=lambda value: labels[value],
        placeholder="Choose two to four players",
    )
    chosen = set_comparison(chosen)
    if len(chosen) < 2:
        empty_state(
            "Choose at least two players",
            "Add players here or use the Compare action from any Player Card.",
            icon="⚖️",
        )
        return

    selected = enriched.loc[enriched["player_id"].isin(chosen)].copy()
    selected["_order"] = selected["player_id"].map({value: index for index, value in enumerate(chosen)})
    selected = selected.sort_values("_order")
    table = comparison_table(selected)
    projections = _forecast_summary(forecasts)
    if not projections.empty:
        table = table.merge(
            selected[["player_id", "name"]].rename(columns={"name": "Player"}),
            on="Player",
            how="left",
        ).merge(projections, on="player_id", how="left").drop(columns="player_id")

    st.markdown("### Decision summary")
    cards = st.columns(min(4, len(selected)))
    for container, row in zip(cards, selected.to_dict("records")):
        player_table = table.loc[table["Player"].eq(row.get("name"))]
        comparison_row = player_table.iloc[0] if not player_table.empty else pd.Series(dtype="object")
        with container:
            st.markdown(f"#### {row.get('name')}")
            st.caption(f"{row.get('team')} · {row.get('fpl_pos')}")
            st.metric("Price", f"£{pd.to_numeric(row.get('now_cost'), errors='coerce') / 10:.1f}m")
            headline = comparison_row.get("Forecast xPts", comparison_row.get("Points"))
            label = "Forecast xPts" if "Forecast xPts" in comparison_row else "Evidence points"
            st.metric(label, "—" if pd.isna(headline) else f"{float(headline):.1f}")
            st.caption(
                f"{row.get('Production profile', 'No production profile')} · "
                f"{row.get('Usage', 'Usage unavailable')}"
            )
            if pd.notna(row.get("Risk")):
                st.error(str(row.get("Risk")), icon="⚠️")
            if st.button("Open Player Card", key=f"open_compare_{row['player_id']}", width="stretch"):
                switch_page("0_main.py", season=season, player=row["player_id"], view="Overview")
            in_shortlist = row["player_id"] in shortlist()
            if st.button(
                "Remove shortlist" if in_shortlist else "Add shortlist",
                key=f"shortlist_compare_{row['player_id']}",
                width="stretch",
            ):
                toggle_shortlist(row["player_id"])
                st.rerun()

    st.markdown("### Exact comparison")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Price": st.column_config.NumberColumn(format="£%.1fm"),
            "Ownership": st.column_config.NumberColumn(format="%.1f%%"),
            "xG": st.column_config.NumberColumn(format="%.2f"),
            "xA": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    numeric_metrics = [column for column in ["Points", "Minutes", "Goals", "Assists", "xG", "xA", "Forecast xPts"] if column in table]
    chart_metric = st.selectbox("Comparison chart metric", numeric_metrics)
    chart_data = table.sort_values(chart_metric, ascending=True)
    figure = px.bar(
        chart_data,
        x=chart_metric,
        y="Player",
        orientation="h",
        color="Position",
        text_auto=".1f",
        title=f"{chart_metric} comparison",
    )
    apply_chart_style(figure, height=max(300, len(table) * 85))
    st.plotly_chart(figure, width="stretch")

    fixtures = fixture_rows(raw_fixtures(league, season), raw_teams(league, season))
    st.markdown("### Fixture comparison")
    fixture_columns = st.columns(min(4, len(selected)))
    for container, row in zip(fixture_columns, selected.to_dict("records")):
        team_id = pd.to_numeric(row.get("fpl_team_numeric_id"), errors="coerce")
        player_fixtures = fixtures.loc[fixtures["team_numeric_id"].eq(team_id)].head(5) if not fixtures.empty else pd.DataFrame()
        with container:
            st.markdown(f"**{row.get('name')}**")
            if player_fixtures.empty:
                st.caption("No upcoming fixtures")
            else:
                for fixture in player_fixtures.to_dict("records"):
                    st.markdown(
                        f":gray-badge[GW{int(fixture['GW'])}] "
                        f"{fixture['Opponent']} ({fixture['Venue']}) · FDR {fixture['FDR']:.0f}"
                    )

    st.markdown("### Best for…")
    strengths: list[str] = []
    criteria = [
        ("Highest projected return", "Forecast xPts", True),
        ("Strongest evidence output", "Points", True),
        ("Safest historical minutes", "Minutes", True),
    ]
    if "Price" in table and "Points" in table:
        table["Points per £m"] = table["Points"].div(table["Price"].where(table["Price"].gt(0)))
        criteria.append(("Best historical value", "Points per £m", True))
    for description, column, descending in criteria:
        eligible = table.dropna(subset=[column]) if column in table else pd.DataFrame()
        if eligible.empty:
            continue
        winner = eligible.sort_values(column, ascending=not descending).iloc[0]
        strengths.append(f"**{description}:** {winner['Player']} ({float(winner[column]):.1f})")
    if strengths:
        for strength in strengths:
            st.markdown(f"- {strength}")
    else:
        st.caption("No shared evidence metrics are available for a recommendation yet.")
    st.caption(
        "These labels identify category leaders, not a universal winner. Price, role, risk and "
        "squad structure can make different players appropriate for different decisions."
    )
    st.caption("Forecast data: " + file_freshness(forecast_path))


if __name__ == "__main__":
    main()
