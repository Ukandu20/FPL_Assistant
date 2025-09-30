#!/usr/bin/env python3
"""
FPL data pull + fixture metadata side-cars (idempotent & season-normalized).

Changes vs your version
• Season folder is normalized to YYYY-YYYY+1 (e.g., 2025-2026), regardless of input.
• Weekly reruns overwrite: all .to_csv are atomic (temp write then os.replace).
• Optional --fresh to remove known generated CSVs before run (safe cleanup).
• Creates needed folders safely.
• cleaned_players.csv and ALL fixture CSVs live under data/raw/fpl/<SEASON>/season/

Outputs
-------
Under data/raw/fpl/<YYYY-YYYY+1>/:
  players/                       (player histories)
  gws/xP<gw>.csv, gws/* merged GW artifacts via your functions
  season/:
    cleaned_players.csv
    fixtures.csv
    teams.csv
    fixture_metadata.csv
    fixture_metadata_resolved.csv
    fixture_metadata_per_team.csv
    fixture_metadata_per_team_resolved.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import csv
from typing import Optional, List

import pandas as pd

from scripts.fpl_pipeline.utils.parse_helpers import *  # noqa: F401,F403
from scripts.fpl_pipeline.clean.cleaners import clean_players, id_players, get_player_ids
from scripts.fpl_pipeline.scrape.api_client import get_data, get_individual_player_data, get_fixtures_data
from scripts.fpl_pipeline.analysis.gw_data_collector import collect_gw, merge_gw


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _with_sep(path: str) -> str:
    return path if path.endswith(os.sep) else path + os.sep

def _write_csv_atomic(df: pd.DataFrame, out_path: str) -> None:
    _ensure_dir(os.path.dirname(out_path))
    tmp = out_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)

def _safe_overwrite_text(out_path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(out_path))
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        f.write(text)
    os.replace(tmp, out_path)

def _normalize_season_fmt(season: Optional[str]) -> str:
    if not season or season.strip().lower() in {"auto", "current"}:
        return _current_euro_season()

    s = season.strip()
    m = re.fullmatch(r"(\d{4})[-/](\d{4})", s)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y2 != y1 + 1:
            y2 = y1 + 1
        return f"{y1:04d}-{y2:04d}"

    m = re.fullmatch(r"(\d{4})[-/](\d{2})", s)
    if m:
        y1, y2_two = int(m.group(1)), int(m.group(2))
        y2 = (y1 // 100) * 100 + y2_two
        if y2 < y1:
            y2 += 100
        return f"{y1:04d}-{y2:04d}"

    m = re.fullmatch(r"(\d{4})", s)
    if m:
        y1 = int(m.group(1))
        return f"{y1:04d}-{(y1+1):04d}"

    return _current_euro_season()

def _current_euro_season(today: Optional[dt.date] = None) -> str:
    if today is None:
        today = dt.date.today()
    y = today.year
    start = y if today.month >= 7 else y - 1
    return f"{start:04d}-{start+1:04d}"

def _delete_if_exists(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def _season_root(season_norm: str) -> str:
    return os.path.join("data", "raw", "fpl", season_norm)

def _fresh_cleanup(season_dir: str) -> None:
    gws_dir = os.path.join(season_dir, "gws")
    if os.path.isdir(gws_dir):
        for name in os.listdir(gws_dir):
            if name.lower().startswith("xp") and name.lower().endswith(".csv"):
                _delete_if_exists(os.path.join(gws_dir, name))
    season_sub = os.path.join(season_dir, "season")
    for fname in [
        "cleaned_players.csv",
        "fixture_metadata.csv",
        "fixture_metadata_resolved.csv",
        "fixture_metadata_per_team.csv",
        "fixture_metadata_per_team_resolved.csv",
        "fixtures.csv",
        "teams.csv",
    ]:
        _delete_if_exists(os.path.join(season_sub, fname))

def _detect_kickoff_col(df: pd.DataFrame) -> str:
    if "kickoff_time" in df.columns:
        return "kickoff_time"
    if "kickofftime" in df.columns:
        return "kickofftime"
    raise KeyError("Neither 'kickoff_time' nor 'kickofftime' found in fixtures.csv")

def _normalize_fixtures_columns(fx: pd.DataFrame) -> pd.DataFrame:
    if "heam_a" in fx.columns and "team_a" not in fx.columns:
        fx = fx.rename(columns={"heam_a": "team_a"})
    return fx

def _teams_map(teams_path: str) -> pd.DataFrame | None:
    if not os.path.exists(teams_path):
        return None
    teams = pd.read_csv(teams_path)
    id_col = "id" if "id" in teams.columns else None
    if id_col is None:
        return None
    cols = [id_col]
    if "name" in teams.columns:
        cols.append("name")
    if "short_name" in teams.columns:
        cols.append("short_name")
    return teams[cols].rename(columns={id_col: "team_id"})

def _add_names(df: pd.DataFrame, tmap: Optional[pd.DataFrame], id_field: str, prefix: str) -> pd.DataFrame:
    if tmap is None:
        return df
    out = df.merge(tmap, left_on=id_field, right_on="team_id", how="left")
    if "name" in tmap.columns:
        out = out.rename(columns={"name": f"{prefix}_name"})
    if "short_name" in tmap.columns:
        out = out.rename(columns={"short_name": f"{prefix}_short"})
    return out.drop(columns=["team_id"])

def create_fixture_metadata(season_dir_season: str) -> None:
    fixtures_path = os.path.join(season_dir_season, "fixtures.csv")
    teams_path    = os.path.join(season_dir_season, "teams.csv")
    out_basic     = os.path.join(season_dir_season, "fixture_metadata.csv")
    out_resolved  = os.path.join(season_dir_season, "fixture_metadata_resolved.csv")

    if not os.path.exists(fixtures_path):
        print(f"WARNING: {fixtures_path} not found; skipping fixture_metadata.csv")
        return

    fx = pd.read_csv(fixtures_path)
    fx = _normalize_fixtures_columns(fx)
    kickoff_col = _detect_kickoff_col(fx)

    needed = {"id", "team_a", "team_h", kickoff_col}
    missing = needed - set(fx.columns)
    if missing:
        raise KeyError(f"fixtures.csv missing required columns: {missing}")

    date_sched = pd.to_datetime(fx[kickoff_col], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")

    meta = pd.DataFrame({
        "fpl_id":     fx["id"].astype("Int64"),
        "team":       fx["team_a"].astype("Int64"),
        "home":       fx["team_h"].astype("Int64"),
        "away":       fx["team_a"].astype("Int64"),
        "date_sched": date_sched.astype("string"),
    })

    meta = meta.sort_values(["fpl_id"]).drop_duplicates(["fpl_id"], keep="first")
    _write_csv_atomic(meta, out_basic)
    print(f"✅ wrote {out_basic} with {len(meta):,} rows")

    tmap = _teams_map(teams_path)
    if tmap is not None:
        resolved = meta.copy()
        resolved = _add_names(resolved, tmap, "team", "team")
        resolved = _add_names(resolved, tmap, "home", "home")
        resolved = _add_names(resolved, tmap, "away", "away")
        _write_csv_atomic(resolved, out_resolved)
        print(f"✅ wrote {out_resolved} with {len(resolved):,} rows")

def create_fixture_metadata_per_team(season_dir_season: str) -> None:
    fixtures_path = os.path.join(season_dir_season, "fixtures.csv")
    teams_path    = os.path.join(season_dir_season, "teams.csv")
    out_basic     = os.path.join(season_dir_season, "fixture_metadata_per_team.csv")
    out_resolved  = os.path.join(season_dir_season, "fixture_metadata_per_team_resolved.csv")

    if not os.path.exists(fixtures_path):
        print(f"WARNING: {fixtures_path} not found; skipping fixture_metadata_per_team.csv")
        return

    fx = pd.read_csv(fixtures_path)
    fx = _normalize_fixtures_columns(fx)
    kickoff_col = _detect_kickoff_col(fx)

    needed = {"id", "team_a", "team_h", kickoff_col}
    missing = needed - set(fx.columns)
    if missing:
        raise KeyError(f"fixtures.csv missing required columns: {missing}")

    date_sched = pd.to_datetime(fx[kickoff_col], errors="coerce", utc=True).dt.strftime("%Y-%m-%d").astype("string")

    home_rows = pd.DataFrame({
        "fpl_id":     fx["id"].astype("Int64"),
        "team":       fx["team_h"].astype("Int64"),
        "opp":        fx["team_a"].astype("Int64"),
        "venue":      pd.Series(["home"] * len(fx), dtype="string"),
        "date_sched": date_sched
    })
    away_rows = pd.DataFrame({
        "fpl_id":     fx["id"].astype("Int64"),
        "team":       fx["team_a"].astype("Int64"),
        "opp":        fx["team_h"].astype("Int64"),
        "venue":      pd.Series(["away"] * len(fx), dtype="string"),
        "date_sched": date_sched
    })

    meta_pt = pd.concat([home_rows, away_rows], ignore_index=True)
    meta_pt = (meta_pt
               .sort_values(["fpl_id", "team", "venue"], kind="mergesort")
               .drop_duplicates(["fpl_id", "team", "venue"], keep="first"))

    _write_csv_atomic(meta_pt, out_basic)
    print(f"✅ wrote {out_basic} with {len(meta_pt):,} rows")

    tmap = _teams_map(teams_path)
    if tmap is not None:
        resolved = meta_pt.copy()
        resolved = _add_names(resolved, tmap, "team", "team")
        resolved = _add_names(resolved, tmap, "opp", "opp")
        _write_csv_atomic(resolved, out_resolved)
        print(f"✅ wrote {out_resolved} with {len(resolved):,} rows")

def fixtures(season_dir_season: str) -> None:
    data = get_fixtures_data()
    parse_fixtures(data, _with_sep(season_dir_season))

def _promote_cleaned_players_to_season_subdir(season_dir: str, season_dir_season: str) -> None:
    candidates: List[str] = [
        os.path.join(season_dir, "players.csv"),
        os.path.join(season_dir, "players_clean.csv"),
        os.path.join(season_dir, "players_cleaned.csv"),
        os.path.join(season_dir, "cleaned_players.csv"),
    ]
    src = next((p for p in candidates if os.path.exists(p)), None)
    if not src:
        print("WARN: Could not find a cleaned players CSV to promote into season/.")
        return
    try:
        df = pd.read_csv(src)
        _write_csv_atomic(df, os.path.join(season_dir_season, "cleaned_players.csv"))
        print(f"✅ promoted {os.path.basename(src)} → season/cleaned_players.csv")
    except Exception as e:
        print(f"WARN: Failed to promote cleaned players CSV from {src}: {e}")

def parse_data(season_input: Optional[str], fresh: bool) -> None:
    season = _normalize_season_fmt(season_input)
    season_dir = _season_root(season)
    players_dir = os.path.join(season_dir, "players")
    gws_dir     = os.path.join(season_dir, "gws")
    season_dir_season = os.path.join(season_dir, "season")

    season_dir_sep        = _with_sep(season_dir)
    players_dir_sep       = _with_sep(players_dir)
    gws_dir_sep           = _with_sep(gws_dir)
    season_dir_season_sep = _with_sep(season_dir_season)

    _ensure_dir(season_dir)
    _ensure_dir(players_dir)
    _ensure_dir(gws_dir)
    _ensure_dir(season_dir_season)

    if fresh:
        print(f"INFO: --fresh cleanup for {season}")
        _fresh_cleanup(season_dir)

    print(f"INFO: Season set to {season}")
    print("Getting bootstrap data …")
    data = get_data()

    print("Parsing summary data …")
    parse_players(data["elements"], season_dir_sep)

    gw_num = 0
    for event in data.get("events", []):
        if event.get("is_current") is True:
            gw_num = int(event["id"])

    print("Cleaning summary data …")
    clean_players(os.path.join(season_dir, "players_raw.csv"), season_dir_sep)
    _promote_cleaned_players_to_season_subdir(season_dir, season_dir_season)

    print("Getting fixtures data …")
    fixtures(season_dir_season_sep)

    print("Getting teams data …")
    parse_team_data(data["teams"], season_dir_season_sep)

    print("Writing fixture metadata (fixture-level & per-team) …")
    create_fixture_metadata(season_dir_season)
    create_fixture_metadata_per_team(season_dir_season)

    print("Extracting player ids …")
    id_players(os.path.join(season_dir, "players_raw.csv"), season_dir_sep)
    player_ids = get_player_ids(season_dir_sep)

    print("Extracting player-specific data …")
    for pid, name in player_ids.items():
        player_data = get_individual_player_data(pid)
        parse_player_history(player_data.get("history_past", []), players_dir_sep, name, pid)
        parse_player_gw_history(player_data.get("history", []), players_dir_sep, name, pid)

    if gw_num > 0:
        print(f"Writing expected points for GW{gw_num} …")
        xp_rows = [{"id": e["id"], "xP": e.get("ep_this")} for e in data["elements"]]
        xp_path = os.path.join(gws_dir, f"xP{gw_num}.csv")
        _safe_overwrite_text(xp_path, "")
        with open(xp_path, "w", newline="") as outf:
            w = csv.DictWriter(outf, ["id", "xP"])
            w.writeheader()
            w.writerows(xp_rows)

        print("Collecting GW scores …")
        # IMPORTANT: root_directory_name must now be the literal season/ subfolder
        collect_gw(gw_num, players_dir_sep, gws_dir_sep, season_dir_season_sep)

        print("Merging GW scores …")
        merge_gw(gw_num, gws_dir_sep)

    print("✅ Done.")


def main():
    parser = argparse.ArgumentParser(description="FPL data pull + fixture metadata side-cars (idempotent).")
    parser.add_argument("--season", type=str, default="current",
                        help="Season (e.g., 2025-2026, 2025-26, 2025/26, 2025). Defaults to current.")
    parser.add_argument("--fresh", action="store_true",
                        help="Optional: remove known generated CSVs in season folder before writing.")
    args = parser.parse_args()
    parse_data(args.season, args.fresh)

if __name__ == "__main__":
    main()
