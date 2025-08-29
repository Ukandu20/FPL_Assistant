#!/usr/bin/env python3
# scripts/fbref_pipeline/scrape/match_stats_scraper.py
from __future__ import annotations
import argparse
import logging
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import soccerdata as sd

from scripts.fbref_pipeline.utils.fbref_utils import (
    STAT_MAP,          # expects keys: "team_match", "player_match"
    safe_write,        # safe_write(df, PathWithoutExt)
    seasons_from_league,
    init_logger,
    polite_sleep,
)

from scripts.fbref_pipeline.scrape.fbref_robust import (    team_match_from_soccerdata_fallback,)

# ───────────────────────── helpers ─────────────────────────

def _prev_season_str(season: str) -> str:
    s = str(season)
    if "-" not in s:
        raise ValueError(f"Season string must be 'YYYY-YYYY', got {s!r}")
    a, b = s.split("-")
    return f"{int(a)-1}-{int(b)-1}"

def _normalize(df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    if "season" not in df.columns:
        df["season"] = season
    else:
        df["season"] = df["season"].fillna(season)
    if "league" not in df.columns:
        df["league"] = league
    else:
        df["league"] = df["league"].fillna(league)
    return df

def _schema_only_from_previous(
    fb_prev: Optional[sd.FBref],
    level: str,      # "team_match" | "player_match" | "schedule"
    stat: str | None = None,
) -> pd.DataFrame:
    """
    Build an empty DF with previous-season columns; fallback to minimal schema.
    For schedule, stat is ignored.
    """
    if fb_prev is not None:
        try:
            if level == "team_match" and stat:
                prev = fb_prev.read_team_match_stats(stat_type=stat)
            elif level == "player_match" and stat:
                prev = fb_prev.read_player_match_stats(stat_type=stat)
            elif level == "schedule":
                prev = fb_prev.read_schedule()
            else:
                prev = None
            if isinstance(prev, pd.DataFrame) and not prev.empty:
                return pd.DataFrame(columns=list(prev.columns))
        except Exception:
            pass

    # Minimal schema if previous season also unavailable/empty
    if level == "schedule":
        base_cols = ["season", "league", "date", "home_team", "away_team", "score"]
    elif level == "team_match":
        base_cols = ["season", "league", "team"]
    else:  # player_match
        base_cols = ["season", "league", "player", "team"]
    return pd.DataFrame(columns=base_cols)

# ───────────────────────── core ─────────────────────────

def scrape_one(league: str, season, out_base: Path, delay: float, no_cache: bool= True):
    log = logging.getLogger("fbref")
    log.info("Match-level: %s %s", league, season)

    season_str = str(season)
    fb = sd.FBref(leagues=league, seasons=season_str, no_cache=no_cache)

    try:
        fb_prev = sd.FBref(leagues=league, seasons=_prev_season_str(season_str))
    except Exception:
        fb_prev = None

    out_dir = out_base / league / season_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Schedule
    try:
        schedule_df = fb.read_schedule()
        if not isinstance(schedule_df, pd.DataFrame) or schedule_df.empty:
            raise ValueError("empty_or_invalid")
        schedule_df = _normalize(schedule_df, league, season_str)
    except Exception as e:
        log.warning("Missing/unavailable schedule for %s %s: %s — writing schema-only CSV",
                    league, season_str, e)
        schedule_df = _schema_only_from_previous(fb_prev, "schedule", None)
        schedule_df = _normalize(schedule_df, league, season_str)
    safe_write(schedule_df, out_dir / "schedule")
    polite_sleep(delay)

    # NOTE: We scrape player-match FIRST to enable aggregation fallback for team-match.
    # 2) Player-match stats
    for stat in STAT_MAP.get("player_match", []):
        try:
            df = fb.read_player_match_stats(stat_type=stat)
            if not isinstance(df, pd.DataFrame) or df.empty:
                raise ValueError("empty_or_invalid")
            df = _normalize(df, league, season_str)
        except Exception as e:
            log.warning(
                "Missing/unavailable player_match %s for %s %s: %s — writing schema-only CSV",
                stat, league, season_str, e,
            )
            df = _schema_only_from_previous(fb_prev, "player_match", stat)
            df = _normalize(df, league, season_str)

        # Keep your original folder layout
        safe_write(df, out_dir / "player_match" / stat)
        polite_sleep(delay)

    # 3) Team-match stats (with aggregation fallback)
    for stat in STAT_MAP.get("team_match", []):
        try:
            df = fb.read_team_match_stats(stat_type=stat, opponent_stats=False)
            if not isinstance(df, pd.DataFrame) or df.empty:
                raise ValueError("empty_or_invalid")
            df = _normalize(df, league, season_str)
        except Exception as e:
            log.warning(
                "Missing/unavailable team_match %s for %s %s via soccerdata: %s — trying aggregate fallback",
                stat, league, season_str, e,
            )
            try:
                df = team_match_from_soccerdata_fallback(
                    fb=fb, stat_type=stat, league=league, season=season_str
                )
                df = _normalize(df, league, season_str)
                log.info("✓ aggregate fallback succeeded for team_match %s", stat)
            except Exception as e2:
                log.warning(
                    "Aggregate fallback failed for team_match %s: %s — writing schema-only CSV",
                    stat, e2,
                )
                df = _schema_only_from_previous(fb_prev, "team_match", stat)
                df = _normalize(df, league, season_str)

        safe_write(df, out_dir / f"team_match_{stat}")
        polite_sleep(delay)

def main():
    p = argparse.ArgumentParser("Scrape FBref match-level tables")
    p.add_argument("--league", default="ENG-Premier League")
    p.add_argument("--out-dir", default="data/raw/fbref")
    p.add_argument("--seasons", nargs="*")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--delay", type=float, default=5)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    warnings.filterwarnings(
        "ignore", category=FutureWarning, module=r".*soccerdata\.fbref.*"
    )

    init_logger(args.verbose)
    seasons = args.seasons or seasons_from_league(args.league)
    for s in seasons:
        scrape_one(args.league, s, Path(args.out_dir), args.delay, no_cache=args.no_cache)

if __name__ == "__main__":
    main()
