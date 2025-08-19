#!/usr/bin/env python3
# scripts/scrape_match_stats.py
from __future__ import annotations
import argparse
from pathlib import Path

import soccerdata as sd
from scripts.fbref_pipeline.utils.fbref_utils import STAT_MAP, safe_write, seasons_from_league, init_logger, polite_sleep


def scrape_one(league: str, season, out_base: Path, delay: float):
    log = __import__("logging").getLogger("fbref")
    log.info("▶ Match-level: %s %s", league, season)
    fb = sd.FBref(leagues=league, seasons=season)
    out_dir = out_base / league / str(season)

    # team-match
    for stat in STAT_MAP["team_match"]:
        try:
            df = fb.read_team_match_stats(stat_type=stat)
            safe_write(df, out_dir / f"team_match_{stat}")
        except Exception as e:
            log.warning("skip team_match %s: %s", stat, e)
        polite_sleep(delay)

    # player-match
    try:
        schedule = fb.read_schedule()
    except Exception as e:
        log.warning("no schedule table: %s", e)
        return


    for stat in STAT_MAP["player_match"]:
        try:
            df = fb.read_player_match_stats(stat_type=stat)
            safe_write(df, out_dir / "player_match" / stat )
        except Exception as e:
            log.debug("skip %s %s: %s", stat, e)
        polite_sleep(delay)

def main():
    p = argparse.ArgumentParser("Scrape FBref match-level tables")
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
