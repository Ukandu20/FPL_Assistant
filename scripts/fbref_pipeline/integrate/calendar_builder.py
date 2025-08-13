#!/usr/bin/env python3
"""
calendar_builder.py  –  Batch-capable builder for player_minutes_calendar.csv

Creates one “skinny” file per season with:
    fbref_id, fpl_id, gw_orig, date_played,
    team_id, player_id, minutes, is_active

Usage:
  ▸ Single season
      python -m scripts.fbref_pipeline.integrate.calendar_builder \
             --season 2024-2025

  ▸ All seasons under fixtures-root
      python -m scripts.fbref_pipeline.integrate.calendar_builder
"""
from __future__ import annotations
import argparse, logging
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np


def load_fixture_calendar(season_dir: Path) -> pd.DataFrame:
    """Reads data/processed/fixtures/<season>/fixture_calendar.csv"""
    fp = season_dir / "fixture_calendar.csv"
    return pd.read_csv(
        fp,
        usecols=[
            "fbref_id",
            "fpl_id",
            "gw_orig",
            "date_played",
            "team_id",
            "team",
            "venue",
            "gf",
            "ga",
            "fdr_home",
            "fdr_away",
        ],
    )


def load_roster_jsons(season_dir: Path, fbref_root: Path) -> pd.DataFrame:
    """
    Builds the roster from:
      • fbref_root/<season>/master_teams.json
      • fbref_root/<season>/season_players.json (snapshot)
    """
    season_key = season_dir.name
    master_fp = fbref_root / "master_teams.json"
    players_fp = fbref_root / season_key / "player_season" / "season_players.json"

    with master_fp.open(encoding="utf-8") as f:
        teams: Dict = json.load(f)
    with players_fp.open(encoding="utf-8") as f:
        players: Dict = json.load(f)

    # map short-code → team_id via the fixture calendar
    fix = pd.read_csv(
        season_dir / "fixture_calendar.csv",
        usecols=["team", "team_id"],
    ).drop_duplicates()
    code2id = dict(zip(fix["team"], fix["team_id"]))

    # 1) historical squad lists
    rows: list[dict[str, str]] = []
    for team_id, rec in teams.items():
        for year, blob in rec.get("career", {}).items():
            if year == season_key:
                for p in blob["players"]:
                    rows.append({"player_id": p["id"], "team_id": team_id})
    roster_mt = pd.DataFrame(rows)

    # 2) snapshot adds (e.g. deadline-day signings)
    extras: list[dict[str, str]] = []
    seen = set(roster_mt["player_id"].tolist())
    for p in players.values():
        pid, code = p["player_id"], p["team"]
        if pid in seen:
            continue
        extras.append({"player_id": pid, "team_id": code2id.get(code)})
    roster_sp = pd.DataFrame(extras)

    roster = pd.concat([roster_mt, roster_sp], ignore_index=True).drop_duplicates()

    # integrity check
    if roster["team_id"].isna().any():
        bad = roster.loc[roster.team_id.isna(), "player_id"].head(5).tolist()
        raise ValueError(
            f"{roster.team_id.isna().sum()} players missing team_id "
            f"(e.g. {bad}). Update master_teams or fixture_calendar."
        )

    return roster.reset_index(drop=True)


def load_minutes(season_dir: Path, fbref_root: Path) -> pd.DataFrame:
    """
    Reads fbref_root/<season>/player_match/summary.csv
    and fbref_root/<season>/player_match/keepers.csv,
    renames columns, and concatenates them into one DataFrame.
    """
    season_key = season_dir.name
    summary_fp = fbref_root / season_key / "player_match" / "summary.csv"
    keeper_fp  = fbref_root / season_key / "player_match" / "keepers.csv"
    def_fp  = fbref_root / season_key / "player_match" / "defense.csv"
    misc_fp  = fbref_root / season_key / "player_match" / "misc.csv"
    
    # 1. load outfield player minutes & events
    df = pd.read_csv(
        summary_fp,
        usecols=[
            "game_id", "player_id", "player", "min", "team_id",
            "crdy", "crdr", "fpl_pos", "gls", "ast", "xg", "npxg", "xag", "pkatt", "pk", "sh", "sot"
        ]
    ).rename(columns={
        "game_id": "fbref_id",
        "min": "minutes",
        "crdy":     "yellow_crd",
        "crdr":     "red_crd",
        "fpl_pos":  "pos",
        "pk":       "pk_scored",
        "sh": "shots"
    })
    

    # 2. load goalkeeper stats
    df_gk = pd.read_csv(
        keeper_fp,
        usecols=[
            "game_id", "player_id",
            "team_id", "sota", "saves", "save"
        ]
    ).rename(columns={
        "game_id": "fbref_id",
        "sota":     "sot_against",
        "save":     "save_pct"
    })

    
    # 2. load defending stats
    df_def = pd.read_csv(
        def_fp,
        usecols=[
            "game_id", "player_id",
            "team_id", "blocks", "tklw", "int", "clr"
        ]
    ).rename(columns={
        "game_id": "fbref_id",
        "tklw":     "tkl",
    })

    
    df_misc = pd.read_csv(
        misc_fp,
        usecols=[
            "game_id", "player_id",
            "team_id", "recov", "pkwon", "og"
        ]
    ).rename(columns={
        "game_id": "fbref_id",
        "recov":     "recoveries",
        "pkwon": "pk_won",
        "og": "own_goals",
    })
    
    # 3) merge GK onto outfield frame (so every row has minutes & events)
    df = df.merge(
        df_gk,
        on=["fbref_id", "player_id", "team_id"],
        how="left"
    )
    df = df.merge(
        df_def,
        on=["fbref_id", "player_id", "team_id"],
        how="left"
    )
    df = df.merge(
        df_misc,
        on=["fbref_id", "player_id", "team_id"],
        how="left"
    )
    
    # 4) fill missing GK stats with 0 for non-GKs
    for col in ("sot_against", "saves", "save_pct"):
        df[col] = df[col].fillna(0)

    return df



def build_minutes_calendar(
    season_dir: Path, fbref_root: Path, force: bool = False
) -> None:
    out_fp = season_dir / "player_minutes_calendar.csv"
    if out_fp.exists() and not force:
        logging.info("%s exists – skipping", out_fp.name)
        return

    # Load fixture calendar
    cal = load_fixture_calendar(season_dir)
    cal["date_played"] = pd.to_datetime(cal["date_played"])

    # Load player minutes directly
    minutes = load_minutes(season_dir, fbref_root)

    # Merge on both 'fbref_id' and 'team_id'
    merged = minutes.merge(cal, on=["fbref_id", "team_id"], how="left")

    # Integrity check: remove rows without fixture match (if any)
    missing_fixtures = merged["date_played"].isna().sum()
    if missing_fixtures:
        logging.warning(f"{missing_fixtures} rows with missing fixture data dropped")
        merged.dropna(subset=["date_played"], inplace=True)

    # Flag active (minutes played greater than 0)
    merged["is_active"] = np.where(merged["minutes"].gt(0), 1, 0).astype("uint8")

    # ─── Calculate days since last match for each player ────────────────────────
    #  sort by player & date so diff() compares correctly
    merged = merged.sort_values(["player_id", "date_played"])
    merged["days_since_last"] = (
        merged.groupby("player_id")["date_played"]
              .diff()                # Timedelta since previous appearance
              .dt.days               # integer days
              .fillna(0)             # first appearance → 0 days
              .astype(int)
    )

    # Output columns
    out_cols = [
        "player_id",
        "player",
        "pos",
        "fbref_id",
        "fpl_id",
        "gw_orig",
        "date_played",
        "team_id",
        "team",        
        "minutes",
        "days_since_last",
        "is_active",
        "yellow_crd",
        "red_crd",
        "venue",
        "gf",
        "ga",
        "fdr_home",
        "fdr_away",
        "gls", "ast", "shots","sot", "blocks", "tkl", "int","clr", "xg", "npxg", "xag", "pkatt", "pk_scored", "pk_won", "saves", "sot_against", "save_pct", "own_goals", "recoveries", 
    ]
    merged[out_cols].to_csv(out_fp, index=False)
    logging.info("✅ %s written (%d rows)", out_fp.name, len(merged))



def run_batch(
    seasons: List[str], fixtures_root: Path, fbref_root: Path, force: bool
) -> None:
    for season in seasons:
        season_dir = fixtures_root / season
        if not season_dir.is_dir():
            logging.warning("Season %s missing – skipped", season)
            continue
        try:
            build_minutes_calendar(season_dir, fbref_root, force=force)
        except Exception:
            logging.exception("❌ %s failed", season)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path("data/processed/fixtures"),
        help="Root dir containing season subfolders",
    )
    ap.add_argument(
        "--fbref-root",
        type=Path,
        default=Path("data/processed/fbref/ENG-Premier League"),
        help="FBref league root (for JSONs and player_match)",
    )
    ap.add_argument(
        "--season", help="e.g. 2024-2025 (omit to process all seasons)"
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(), format="%(levelname)s: %(message)s"
    )

    if args.season:
        seasons = [args.season]
    else:
        seasons = sorted(
            d.name for d in args.fixtures_root.iterdir() if d.is_dir()
        )
        if not seasons:
            logging.error("No seasons found under %s", args.fixtures_root)
            return

    run_batch(seasons, args.fixtures_root, args.fbref_root, args.force)


if __name__ == "__main__":
    main()
