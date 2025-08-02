#!/usr/bin/env python3
# scripts/scrape_season_stats.py
from __future__ import annotations
import argparse
from pathlib import Path

import soccerdata as sd
from fbref_utils import STAT_MAP, safe_write, seasons_from_league, init_logger, polite_sleep

def scrape_one(league: str, season, out_base: Path, delay: float):
    log = __import__("logging").getLogger("fbref")
    log.info("▶ Season-level: %s %s", league, season)
    fb = sd.FBref(leagues=league, seasons=season)
    out_dir = out_base / league / str(season)

    for level, fn in {
        "team_season": fb.read_team_season_stats,
        "player_season": fb.read_player_season_stats,
    }.items():
        for stat in STAT_MAP[level]:
            try:
                df = fn(stat_type=stat)
                safe_write(df, out_dir / f"{level}_{stat}")
            except Exception as e:
                log.warning("skip %s %s: %s", level, stat, e)
            polite_sleep(delay)

def main():
    p = argparse.ArgumentParser("Scrape FBref season-level tables")
    p.add_argument("--league", default="ENG-Premier League")
    p.add_argument("--out-dir", default="data/raw/fbref")
    p.add_argument("--seasons", nargs="*")
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    init_logger(args.verbose)
    seasons = args.seasons or seasons_from_league(args.league)
    for s in seasons:
        scrape_one(args.league, s, Path(args.out_dir), args.delay)

if __name__ == "__main__":
    main()
