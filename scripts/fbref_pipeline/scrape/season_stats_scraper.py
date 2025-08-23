#!/usr/bin/env python3
from __future__ import annotations
import argparse
import logging
import warnings
from pathlib import Path

import pandas as pd
import soccerdata as sd

from scripts.fbref_pipeline.utils.fbref_utils import (
    STAT_MAP,
    safe_write,
    seasons_from_league,
    init_logger,
    polite_sleep,
)

# ───────────────────────── helpers ─────────────────────────

def _prev_season_str(season: str) -> str:
    s = str(season)
    if "-" not in s:
        raise ValueError(f"Season string must be 'YYYY-YYYY', got {s!r}")
    a, b = s.split("-")
    return f"{int(a)-1}-{int(b)-1}"

def _normalize(df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    """Ensure required columns exist and are filled."""
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
    fb_prev: sd.FBref | None,
    level: str,      # "team_season" | "player_season"
    stat: str,
) -> pd.DataFrame:
    """Build an empty DF with previous-season columns; fallback to minimal schema."""
    if fb_prev is not None:
        try:
            read_fn = (
                fb_prev.read_team_season_stats
                if level == "team_season"
                else fb_prev.read_player_season_stats
            )
            prev = read_fn(stat_type=stat)
            if isinstance(prev, pd.DataFrame) and not prev.empty:
                return pd.DataFrame(columns=list(prev.columns))
        except Exception:
            pass
    # Minimal guaranteed schema if previous season also unavailable/empty
    base_cols = (
        ["season", "league", "team"]
        if level == "team_season"
        else ["season", "league", "player"]
    )
    return pd.DataFrame(columns=base_cols)

# ───────────────────────── core ─────────────────────────

def scrape_one(league: str, season, out_base: Path, delay: float, no_cache: bool=True):
    log = logging.getLogger("fbref")
    # ASCII only — avoid Windows cp1252 UnicodeEncodeError
    log.info("Season-level: %s %s", league, season)

    season_str = str(season)
    fb = sd.FBref(leagues=league, seasons=season_str, no_cache=no_cache)

    try:
        fb_prev = sd.FBref(leagues=league, seasons=_prev_season_str(season_str))
    except Exception:
        fb_prev = None

    out_dir = out_base / league / season_str

    for level, fn in {
        "team_season": fb.read_team_season_stats,
        "player_season": fb.read_player_season_stats,
    }.items():
        for stat in STAT_MAP[level]:
            try:
                df = fn(stat_type=stat)
                if not isinstance(df, pd.DataFrame) or df.empty:
                    raise ValueError("empty_or_invalid")
                df = _normalize(df, league, season_str)
            except Exception as e:
                log.warning(
                    "Missing/unavailable %s %s for %s %s: %s — writing schema-only CSV",
                    level, stat, league, season_str, e,
                )
                df = _schema_only_from_previous(fb_prev, level, stat)
                df = _normalize(df, league, season_str)

            # Your safe_write handles the path base; we keep your convention
            safe_write(df, out_dir / f"{level}_{stat}")
            polite_sleep(delay)

def main():
    p = argparse.ArgumentParser("Scrape FBref season-level tables")
    p.add_argument("--league", default="ENG-Premier League")
    p.add_argument("--out-dir", default="data/raw/fbref")
    p.add_argument("--seasons", nargs="*")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    # Silence noisy pandas FutureWarnings coming from soccerdata internals
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module=r".*soccerdata\.fbref.*"
    )

    init_logger(args.verbose)
    seasons = args.seasons or seasons_from_league(args.league)

    for s in seasons:
        scrape_one(args.league, s, Path(args.out_dir), args.delay, no_cache=args.no_cache)

if __name__ == "__main__":
    main()
