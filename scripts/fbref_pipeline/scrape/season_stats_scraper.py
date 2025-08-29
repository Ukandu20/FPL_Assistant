#!/usr/bin/env python3
# scripts/fbref_pipeline/scrape/season_stats_scraper.py
from __future__ import annotations

import argparse
import logging
import warnings
import time
import random
from pathlib import Path
from typing import Optional, Callable, Any, Dict

import pandas as pd
import soccerdata as sd
from requests.exceptions import HTTPError  # precise import

from scripts.fbref_pipeline.utils.fbref_utils import (
    STAT_MAP,          # expects keys: "team_season", "player_season"
    safe_write,        # safe_write(df, PathWithoutExt)
    seasons_from_league,
    init_logger,
    polite_sleep,      # existing sleep helper; wrapped below
)

# ───────────────────────── helpers ─────────────────────────

def _prev_season_str(season: str) -> str:
    """Convert 'YYYY-YYYY' → previous season string."""
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

def _snooze(base: float) -> None:
    """
    Polite sleep with jitter. We add 20–40% jitter on top of 'base' to avoid
    thundering-herd patterns. Guarantees at least 'base' seconds.
    """
    jitter = base * random.uniform(0.2, 0.4)
    polite_sleep(base + jitter)

def _call_with_optional_kw(fn: Callable[..., Any], kwargs: Dict[str, Any], optional_kw: Optional[Dict[str, Any]] = None):
    """
    Call fn(**kwargs, **optional_kw) if the function supports those keywords.
    If a TypeError occurs due to unexpected keyword(s), retry without them.

    This allows passing 'force_cache' to newer soccerdata builds, while
    remaining compatible with older builds that don't accept it.
    """
    if optional_kw:
        try:
            return fn(**{**kwargs, **optional_kw})
        except TypeError as e:
            msg = str(e)
            if any(f"'{k}'" in msg or k in msg for k in optional_kw.keys()):
                return fn(**kwargs)
            raise
    return fn(**kwargs)

def _with_backoff(
    fn: Callable[..., Any],
    *,
    max_retries: int = 6,
    base_delay: float = 3.2,
    kwargs: Optional[Dict[str, Any]] = None,
    optional_kw: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Calls `fn(**kwargs, **optional_kw)` with exponential backoff on HTTP 429.
    Backoff: base_delay * 1.8^attempt + [0,1) jitter, capped at ~60s increments.

    - 'kwargs' are required parameters for the function.
    - 'optional_kw' are nice-to-have kwargs (e.g., force_cache) that may not be
      supported on all installs; if unsupported, we retry without them.
    """
    if kwargs is None:
        kwargs = {}

    delay = base_delay
    last_exc: Optional[Exception] = None
    log = logging.getLogger("fbref")

    for attempt in range(max_retries):
        try:
            return _call_with_optional_kw(fn, kwargs=kwargs, optional_kw=optional_kw)
        except Exception as e:
            last_exc = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            msg = str(e).lower()
            is_429 = (isinstance(e, HTTPError) and status == 429) or ("429" in msg) or ("too many requests" in msg)

            if is_429:
                sleep_for = min(delay, 60.0) + random.random()  # + small jitter
                log.warning("HTTP 429; backing off for %.1fs (attempt %d/%d)", sleep_for, attempt + 1, max_retries)
                time.sleep(sleep_for)
                delay *= 1.8
                continue

            # Non-429: surface immediately
            raise

    raise last_exc if last_exc else RuntimeError("Transient error: retries exhausted without a captured exception.")

def _schema_only_from_previous(
    fb_prev: sd.FBref | None,
    level: str,      # "team_season" | "player_season"
    stat: str,
) -> pd.DataFrame:
    """Build an empty DF with previous-season columns; fallback to minimal schema (cache-only)."""
    if fb_prev is not None:
        try:
            if level == "team_season":
                prev = _call_with_optional_kw(
                    fb_prev.read_team_season_stats,
                    kwargs={"stat_type": stat},
                    optional_kw={"force_cache": True},
                )
            else:
                prev = _call_with_optional_kw(
                    fb_prev.read_player_season_stats,
                    kwargs={"stat_type": stat},
                    optional_kw={"force_cache": True},
                )
            if isinstance(prev, pd.DataFrame) and not prev.empty:
                return pd.DataFrame(columns=list(prev.columns))
        except Exception:
            pass

    base_cols = ["season", "league", "team"] if level == "team_season" else ["season", "league", "player"]
    return pd.DataFrame(columns=base_cols)

# ───────────────────────── core ─────────────────────────

def scrape_one(
    league: str,
    season: str,
    out_base: Path,
    delay: float,
    no_cache: bool = False,
    force_cache: bool = False,
    max_retries: int = 6,
    backoff_base: float = 3.2,
) -> None:
    """
    Scrape season-level stats for one league+season into the raw FBref structure.
    """
    log = logging.getLogger("fbref")
    log.info("Season-level: %s %s", league, season)

    season_str = str(season)
    fb = sd.FBref(leagues=league, seasons=season_str, no_cache=no_cache)

    try:
        fb_prev = sd.FBref(leagues=league, seasons=_prev_season_str(season_str))
    except Exception:
        fb_prev = None

    out_dir = out_base / league / season_str
    out_dir.mkdir(parents=True, exist_ok=True)

    for level, fn in {
        "team_season": fb.read_team_season_stats,
        "player_season": fb.read_player_season_stats,
    }.items():
        for stat in STAT_MAP[level]:
            try:
                df = _with_backoff(
                    fn,
                    max_retries=max_retries,
                    base_delay=backoff_base,
                    kwargs={"stat_type": stat},
                    optional_kw={"force_cache": force_cache},
                )
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

            safe_write(df, out_dir / f"{level}_{stat}")
            _snooze(delay)

# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    p = argparse.ArgumentParser("Scrape FBref season-level tables")
    p.add_argument("--league", default="ENG-Premier League")
    p.add_argument("--out-dir", default="data/raw/fbref")
    p.add_argument("--seasons", nargs="*")
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass local cache (not recommended; increases 429 risk).",
    )
    p.add_argument(
        "--force-cache",
        action="store_true",
        help="Read only from cache; never hit network (safe for dev runs).",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=3.2,
        help="Base inter-request delay (seconds). Actual sleep adds 20–40%% jitter.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Max 429 backoff retries per request.",
    )
    p.add_argument(
        "--backoff-base",
        type=float,
        default=3.2,
        help="Initial backoff (seconds) on HTTP 429; grows by ~1.8x each retry.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    # Silence noisy pandas FutureWarnings coming from soccerdata internals
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module=r".*soccerdata\.fbref.*"
    )

    init_logger(args.verbose)
    seasons = args.seasons or seasons_from_league(args.league)

    for s in seasons:
        scrape_one(
            args.league,
            str(s),
            Path(args.out_dir),
            args.delay,
            no_cache=args.no_cache,
            force_cache=args.force_cache,
            max_retries=args.max_retries,
            backoff_base=args.backoff_base,
        )

if __name__ == "__main__":
    main()
