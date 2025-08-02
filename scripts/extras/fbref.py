#!/usr/bin/env python3
"""
scrape_fbref_all_seasons.py
––––––––––––––––––––––––––––
Scrape every team- and player-level table exposed by the `soccerdata` FBref
wrapper for all (or selected) seasons of a league.  For each DataFrame it writes
*both* a CSV and a Snappy-compressed Parquet file:

data/
└── raw/
    └── fbref/
        └── ENG-Premier League/
            └── 2023-2024/
                ├── team_season_passing.csv
                ├── team_season_passing.parquet
                ├── …
                └── player_match/
                    └── summary/
                        └── 19490d57.csv        (one file per match id)
                        └── 19490d57.parquet
                        └── …

Fixes vs. your original
-----------------------
1. **`match_id` cast to `str`** before it’s appended to a `Path`.
2. Wrapped the *inner* loops in `try/except`, so one bad stat or match doesn’t
   kill the season.
3. Optional `--delay` flag (seconds) so you can scrape politely.
4. Logging is timestamped and set to INFO by default; pass `-v` for DEBUG.

Dependencies
------------
python -m pip install --upgrade soccerdata pandas pyarrow
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import soccerdata as sd

# ───────────────────────────── Configuration ────────────────────────────────
STAT_MAP: Dict[str, List[str]] = {
    "team_season": [
        "standard", "keeper", "keeper_adv", "shooting", "passing", "passing_types",
        "goal_shot_creation", "defense", "possession", "playing_time", "misc",
    ],
    "team_match": [
        "schedule", "keeper", "shooting", "passing", "passing_types",
        "goal_shot_creation", "defense", "possession", "misc",
    ],
    "player_season": [
        "standard", "shooting", "passing", "passing_types", "goal_shot_creation",
        "defense", "possession", "playing_time", "misc", "keeper", "keeper_adv",
    ],
    "player_match": [
        "summary", "keepers", "passing", "passing_types", "defense",
        "possession", "misc",
    ],
}


# ───────────────────────────── Utilities ────────────────────────────────────
def safe_write(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame as CSV and Parquet, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=True)
    df.to_parquet(path.with_suffix(".parquet"), index=True, compression="snappy")
    size = path.with_suffix(".parquet").stat().st_size / 1e6
    logging.getLogger("scraper").debug("saved %s (%.1f MB)", path.stem, size)


def loop_stats(
    fbref,
    dest_dir: Path,
    msg_prefix: str,
    fn_read,
    stat_list: List[str],
    delay: float,
):
    for stat in stat_list:
        try:
            df = fn_read(stat_type=stat)
            safe_write(df, dest_dir / f"{msg_prefix}_{stat}")
            time.sleep(delay)
        except Exception as e:
            logging.getLogger("scraper").warning("skip %s %s: %s", msg_prefix, stat, e)


def scrape_season(league: str, season, out_base: Path, delay: float = 0.5) -> None:
    log = logging.getLogger("scraper")
    log.info("▶ Scraping %s %s", league, season)

    fbref = sd.FBref(leagues=league, seasons=season)
    season_dir = out_base / league / str(season)

    # team-season + team-match + player-season
    loop_stats(fbref, season_dir, "team_season", fbref.read_team_season_stats,
               STAT_MAP["team_season"], delay)
    loop_stats(fbref, season_dir, "team_match", fbref.read_team_match_stats,
               STAT_MAP["team_match"], delay)
    loop_stats(fbref, season_dir, "player_season", fbref.read_player_season_stats,
               STAT_MAP["player_season"], delay)

    # player-match (one file per match id)
    try:
        schedule = fbref.read_schedule()
    except Exception as e:
        log.warning("no schedule table for %s %s: %s", league, season, e)
        return

    match_ids = schedule.index.get_level_values("game_id").unique()
    for match_id in match_ids:
        for stat in STAT_MAP["player_match"]:
            try:
                df = fbref.read_player_match_stats(stat_type=stat, match_id=match_id)
                safe_write(
                    df,
                    season_dir / "player_match" / stat / str(match_id),
                )
                time.sleep(delay)
            except Exception as e:
                log.debug("skip match %s %s: %s", match_id, stat, e)


# ───────────────────────────── CLI ──────────────────────────────────────────
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape FBref tables across seasons")
    p.add_argument("--league", default="ENG-Premier League",
                   help="League name exactly as soccerdata expects")
    p.add_argument("--out-dir", default="data/raw/fbref",
                   help="Root directory for output files")
    p.add_argument("--seasons", nargs="*", type=str,
                   help="Specific seasons (e.g. 2023-2024). Omit for all.")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Sleep N seconds between requests (default 0.5)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Debug-level logging")
    return p.parse_args()


def main() -> None:
    args = get_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("scraper")

    tmp = sd.FBref(leagues=args.league)
    available = tmp.read_seasons().index.get_level_values("season").unique().tolist()
    seasons = args.seasons or available
    log.info("Will scrape %d season(s): %s", len(seasons), seasons)

    for season in seasons:
        try:
            scrape_season(args.league, season, Path(args.out_dir), args.delay)
        except Exception as e:
            log.error("FAILED %s %s → %s", args.league, season, e)

    log.info("✓ All done")


if __name__ == "__main__":
    main()
