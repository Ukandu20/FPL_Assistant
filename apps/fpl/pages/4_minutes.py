"""Browse published minutes forecasts by season and gameweek."""

import streamlit as st

from apps.fpl.minutes import MINUTES_ROOT, forecast_files, load_minutes, minutes_table, render_minutes
from apps.fpl.ui import page_header


def main() -> None:
    page_header("Minutes Forecast", "Expected playing time, starting chances and rotation risk.")
    seasons = sorted(
        (p.name for p in MINUTES_ROOT.iterdir() if p.is_dir() and forecast_files(p.name)),
        reverse=True,
    ) if MINUTES_ROOT.is_dir() else []
    if not seasons:
        st.info("No minutes forecasts have been published yet.")
        return
    left, right = st.columns(2)
    season = left.selectbox("Season", seasons)
    path = right.selectbox("Gameweek", forecast_files(season), format_func=lambda p: p.stem)
    data, path = load_minutes(season, path)
    if data.empty:
        render_minutes(data, path)
        return
    left, middle, right = st.columns(3)
    teams = left.multiselect("Teams", sorted(data["team"].dropna().unique()))
    positions = middle.multiselect("Positions", sorted(data["pos"].dropna().unique()))
    search = right.text_input("Player name")
    filtered = data.copy()
    if teams:
        filtered = filtered.loc[filtered["team"].isin(teams)]
    if positions:
        filtered = filtered.loc[filtered["pos"].isin(positions)]
    if search.strip():
        filtered = filtered.loc[filtered["player"].str.contains(search.strip(), case=False, regex=False, na=False)]
    # Resolve opponents before team/player filters remove the opposing team's rows.
    names = data.drop_duplicates("team_id").set_index("team_id")["team"]
    filtered["opponent_id"] = filtered["opponent_id"].map(names).fillna(filtered["opponent_id"])
    filtered = filtered.sort_values("pred_minutes", ascending=False)
    render_minutes(filtered, path)
    st.download_button("Download filtered forecast", minutes_table(filtered).to_csv(index=False),
                       file_name=f"minutes_{season}_{path.stem}.csv", mime="text/csv")


if __name__ == "__main__":
    main()
