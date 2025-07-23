#!/usr/bin/env python3
"""
export_fpl_csvs_all_seasons.py
===============================
Loop through all archived FPL seasons and write DataFrames to CSV, including team-specific files.

Assumes:
  data/raw/fpl/<season>/...

Outputs to:
  data/processed/fpl/<season>/...
"""
from pathlib import Path
from fpl_json_to_df import (
    load_bootstrap,
    load_fixtures,
    load_events,
    load_gameweek_live,
    load_player_history,
    build_player_table,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "fpl"
OUT_ROOT = ROOT / "data" / "processed" / "fpl"
# ──────────────────────────────────────────────────────────────────────────────

for season_dir in sorted(RAW_ROOT.iterdir()):
    if not season_dir.is_dir():
        continue
    season = season_dir.name
    print(f"Processing season: {season}")

    # Prepare output folder
    out_dir = OUT_ROOT / season
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    players_df, teams_df, element_types_df = load_bootstrap(season_dir)
    fixtures_df = load_fixtures(season_dir)
    events_df   = load_events(season_dir)
    gw_live_df  = load_gameweek_live(season_dir)
    hist_df, past_df = load_player_history(season_dir)
    player_table_df  = build_player_table(players_df, teams_df, element_types_df)

    # Write standard CSVs
    players_df.to_csv(out_dir / "players.csv", index=False)
    teams_df.to_csv(out_dir / "teams.csv", index=False)
    element_types_df.to_csv(out_dir / "element_types.csv", index=False)
    fixtures_df.to_csv(out_dir / "fixtures.csv", index=False)
    events_df.to_csv(out_dir / "events.csv", index=False)
    gw_live_df.to_csv(out_dir / "gameweek_live.csv", index=False)
    hist_df.to_csv(out_dir / "player_history_gw.csv", index=False)
    past_df.to_csv(out_dir / "player_history_seasons.csv", index=False)
    player_table_df.to_csv(out_dir / "player_master_table.csv", index=False)

    # Additional CSV: master table sorted by team
    sorted_by_team = player_table_df.sort_values(["team_name", "web_name"])
    sorted_by_team.to_csv(
        out_dir / "player_master_table_sorted_by_team.csv", index=False
    )

    # Per-team CSVs: one file per club
    team_dir = out_dir / "players_by_team"
    team_dir.mkdir(exist_ok=True)
    for team in player_table_df["team_name"].unique():
        df_team = player_table_df[player_table_df["team_name"] == team]
        df_team = df_team.sort_values("web_name")
        filename = f"{team.replace(' ', '_').lower()}.csv"
        df_team.to_csv(team_dir / filename, index=False)

    print(f"CSV export complete for {season} -> {out_dir}\n")
