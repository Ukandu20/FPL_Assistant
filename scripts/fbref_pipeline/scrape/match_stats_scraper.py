#!/usr/bin/env python3
# scripts/fbref_pipeline/scrape/match_stats_scraper.py
"""
FBref match-level scraper (network-friendly, cache-first)

Key features:
- Exponential backoff for HTTP 429 / transient errors (honors Retry-After).
- Jittered sleeps between requests to avoid synchronized bursts.
- Cache-first defaults (no_cache=False) + --force-cache for offline dev runs.
- Previous-season schema discovery uses force_cache (never hits network).
- **Team stats can avoid network entirely**: --team-mode aggregate|auto|direct
    - aggregate  (default): build team tables from player-match (zero extra hits)
    - auto: try aggregate first; fall back to direct if aggregate unavailable
    - direct: go to FBref endpoints (may trigger 429 without generous throttling)
- Periodic rest after N network calls to cool down IP.

Typical usage:
  # First run (gentle, primes cache)
  python scripts/fbref_pipeline/scrape/match_stats_scraper.py \
    --league "ENG-Premier League" --seasons 2025-2026 --delay 3.2

  # Subsequent development runs (cache-only; zero risk of 429)
  python scripts/fbref_pipeline/scrape/match_stats_scraper.py \
    --league "ENG-Premier League" --seasons 2025-2026 --force-cache

  # Force aggregate team stats (recommended to avoid 429 on team pages)
  python scripts/fbref_pipeline/scrape/match_stats_scraper.py \
    --league "ENG-Premier League" --seasons 2025-2026 --team-mode aggregate
"""

from __future__ import annotations

import argparse
import logging
import warnings
import time
import random
from pathlib import Path
from typing import Optional, Callable, Any, Dict, Tuple

import pandas as pd
import soccerdata as sd
from requests.exceptions import HTTPError  # precise import

from scripts.fbref_pipeline.utils.fbref_utils import (
    STAT_MAP,          # expects keys: "team_match", "player_match"
    safe_write,        # safe_write(df, PathWithoutExt)
    seasons_from_league,
    init_logger,
    polite_sleep,      # your existing sleep helper (we'll wrap it)
)

from scripts.fbref_pipeline.scrape.fbref_robust import (
    team_match_from_soccerdata_fallback,
)

# ───────────────────────── global throttle state ─────────────────────────

_GLOBAL = {"net_calls": 0}

# ───────────────────────── helpers ─────────────────────────

def _prev_season_str(season: str) -> str:
    """Convert 'YYYY-YYYY' → previous season string."""
    s = str(season)
    if "-" not in s:
        raise ValueError(f"Season string must be 'YYYY-YYYY', got {s!r}")
    a, b = s.split("-")
    return f"{int(a)-1}-{int(b)-1}"

def _normalize(df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    """Ensure 'season' and 'league' columns exist and are filled."""
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

def _periodic_rest(every: int, secs: float, *, skip: bool) -> None:
    """
    Take a longer rest every `every` network calls (approximate), unless skip=True
    (e.g., when --force-cache is set).
    """
    if skip:
        return
    if every > 0 and _GLOBAL["net_calls"] > 0 and (_GLOBAL["net_calls"] % every == 0):
        logging.getLogger("fbref").info("Cooling down: resting for %.1fs after %d calls", secs, _GLOBAL["net_calls"])
        time.sleep(secs)

def _parse_retry_after(exc: Exception) -> Optional[float]:
    """Extract Retry-After seconds if present on HTTP errors."""
    resp = getattr(exc, "response", None)
    if not resp or not hasattr(resp, "headers"):
        return None
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        # Retry-After can be HTTP-date or seconds; we handle seconds here.
        return float(ra)
    except Exception:
        return None

def _call_with_optional_kw(fn: Callable[..., Any], kwargs: Dict[str, Any], optional_kw: Optional[Dict[str, Any]] = None):
    """
    Call fn(**kwargs, **optional_kw) if the function supports those keywords.
    If a TypeError occurs due to unexpected keyword(s), retry without them.

    This allows us to pass 'force_cache' to newer soccerdata builds, while
    remaining compatible with older builds that don't accept it.
    """
    if optional_kw:
        try:
            return fn(**{**kwargs, **optional_kw})
        except TypeError as e:
            # If the error mentions any of our optional keywords, retry without them.
            msg = str(e)
            if any(f"'{k}'" in msg or k in msg for k in optional_kw.keys()):
                return fn(**kwargs)
            raise
    # No optional keywords provided; just call directly.
    return fn(**kwargs)

def _with_backoff(
    fn: Callable[..., Any],
    *,
    max_retries: int = 6,
    base_delay: float = 3.2,
    kwargs: Optional[Dict[str, Any]] = None,
    optional_kw: Optional[Dict[str, Any]] = None,
    count_as_network: bool = True,
    periodic_every: int = 8,
    periodic_secs: float = 15.0,
) -> Any:
    """
    Calls `fn(**kwargs, **optional_kw)` with exponential backoff on HTTP 429.
    Backoff: base_delay * 1.8^attempt + [0,1) jitter, capped at ~60s increments.

    - 'kwargs' are required parameters for the function.
    - 'optional_kw' are nice-to-have kwargs (e.g., force_cache) that may not be
      supported on all installs; if unsupported, we retry without them (no penalty).
    - 'count_as_network' toggles whether this increments the global net_calls counter.
      We set this False when force_cache=True (best-effort).
    """
    if kwargs is None:
        kwargs = {}

    delay = base_delay
    last_exc: Optional[Exception] = None
    log = logging.getLogger("fbref")

    for attempt in range(max_retries):
        try:
            result = _call_with_optional_kw(fn, kwargs=kwargs, optional_kw=optional_kw)
            # Successful return:
            if count_as_network:
                _GLOBAL["net_calls"] += 1
            # Periodic cool-down after some number of calls
            _periodic_rest(periodic_every, periodic_secs, skip=not count_as_network)
            return result
        except Exception as e:
            last_exc = e

            # Extract status code if present; soccerdata often wraps requests exceptions.
            status = getattr(getattr(e, "response", None), "status_code", None)
            msg = str(e).lower()
            is_429 = (isinstance(e, HTTPError) and status == 429) or ("429" in msg) or ("too many requests" in msg)

            if is_429:
                # Respect Retry-After if provided
                ra = _parse_retry_after(e)
                if ra:
                    sleep_for = float(ra) + random.random()
                else:
                    sleep_for = min(delay, 60.0) + random.random()  # add small 0–1s jitter

                log.warning("HTTP 429; backing off for %.1fs (attempt %d/%d)", sleep_for, attempt + 1, max_retries)
                time.sleep(sleep_for)
                delay *= 1.8
                continue

            # Non-429: surface immediately
            raise

    # Exhausted retries
    raise last_exc if last_exc else RuntimeError("Transient error: retries exhausted without a captured exception.")

def _schema_only_from_previous(
    fb_prev: Optional[sd.FBref],
    level: str,      # "team_match" | "player_match" | "schedule"
    stat: str | None = None,
) -> pd.DataFrame:
    """
    Build an empty DF with previous-season columns; fallback to minimal schema.
    We ONLY read from cache for schema discovery (never hit the network).
    For schedule, 'stat' is ignored.
    """
    if fb_prev is not None:
        try:
            if level == "team_match" and stat:
                prev = _call_with_optional_kw(
                    fb_prev.read_team_match_stats,
                    kwargs={"stat_type": stat},
                    optional_kw={"force_cache": True},
                )
            elif level == "player_match" and stat:
                prev = _call_with_optional_kw(
                    fb_prev.read_player_match_stats,
                    kwargs={"stat_type": stat},
                    optional_kw={"force_cache": True},
                )
            elif level == "schedule":
                prev = _call_with_optional_kw(
                    fb_prev.read_schedule,
                    kwargs={},
                    optional_kw={"force_cache": True},
                )
            else:
                prev = None

            if isinstance(prev, pd.DataFrame) and not prev.empty:
                return pd.DataFrame(columns=list(prev.columns))
        except Exception:
            # If previous season isn't cached or method not supported, we fall through.
            pass

    # Minimal schema if previous season also unavailable/empty
    if level == "schedule":
        base_cols = ["season", "league", "date", "home_team", "away_team", "score"]
    elif level == "team_match":
        base_cols = ["season", "league", "team"]
    else:  # player_match
        base_cols = ["season", "league", "player", "team"]
    return pd.DataFrame(columns=base_cols)

def _try_team_aggregate(
    fb: sd.FBref,
    stat: str,
    league: str,
    season_str: str,
) -> Tuple[bool, pd.DataFrame]:
    """
    Attempt to build team match-level stats from player-match tables.
    Returns (ok, df). If ok is False, df is undefined/empty.
    """
    log = logging.getLogger("fbref")
    try:
        df = team_match_from_soccerdata_fallback(
            fb=fb, stat_type=stat, league=league, season=season_str
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = _normalize(df, league, season_str)
            log.info("✓ team_match %s built via aggregate fallback (no direct hits)", stat)
            return True, df
        return False, pd.DataFrame()
    except Exception as e:
        log.debug("Aggregate path for team_match %s failed: %s", stat, e)
        return False, pd.DataFrame()

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
    periodic_every: int = 8,
    periodic_secs: float = 15.0,
    team_mode: str = "aggregate",  # 'aggregate' | 'auto' | 'direct'
    proxy: Optional[str] = None,
    levels: str = "both",          
) -> None:
    """
    Scrape one league+season into the raw FBref folder structure.

    Parameters
    ----------
    league : str
        e.g., "ENG-Premier League"
    season : str
        "YYYY-YYYY" string
    out_base : Path
        Root output directory (e.g., data/raw/fbref)
    delay : float
        Base inter-request delay (seconds); actual sleep = delay + 20–40% jitter
    no_cache : bool
        If True, bypass local cache (NOT recommended; increases 429 risk)
    force_cache : bool
        If True, never hit the network; read-only from cache (safe for dev runs)
    max_retries : int
        Max retry attempts per request when 429 is encountered
    backoff_base : float
        Initial backoff seconds for 429; grows exponentially by 1.8x
    periodic_every : int
        After roughly this many network calls, take a longer rest
    periodic_secs : float
        Length of that periodic rest
    team_mode : str
        'aggregate' (prefer aggregate, never direct), 'auto' (aggregate→direct), 'direct'
    proxy : Optional[str]
        e.g., "tor" or "http://user:pass@host:port" (passed to soccerdata if provided)
    """
    log = logging.getLogger("fbref")
    log.info("Match-level: %s %s (team_mode=%s)", league, season, team_mode)

    season_str = str(season)
    fb_kwargs = dict(leagues=league, seasons=season_str, no_cache=no_cache)
    if proxy:
        fb_kwargs["proxy"] = proxy
    fb = sd.FBref(**fb_kwargs)

    try:
        fb_prev = sd.FBref(leagues=league, seasons=_prev_season_str(season_str))
    except Exception:
        fb_prev = None

    out_dir = out_base / league / season_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Schedule
    try:
        schedule_df = _with_backoff(
            fb.read_schedule,
            max_retries=max_retries,
            base_delay=backoff_base,
            kwargs={},
            optional_kw={"force_cache": force_cache},
            count_as_network=not force_cache,
            periodic_every=periodic_every,
            periodic_secs=periodic_secs,
        )
        if not isinstance(schedule_df, pd.DataFrame) or schedule_df.empty:
            raise ValueError("empty_or_invalid")
        schedule_df = _normalize(schedule_df, league, season_str)
    except Exception as e:
        log.warning(
            "Missing/unavailable schedule for %s %s: %s — writing schema-only CSV",
            league, season_str, e
        )
        schedule_df = _schema_only_from_previous(fb_prev, "schedule", None)
        schedule_df = _normalize(schedule_df, league, season_str)

    safe_write(schedule_df, out_dir / "schedule")
    _snooze(delay)

    # NOTE: Player-match FIRST (when enabled) to support aggregate team builds.
    if levels in ("player", "both"):
        for stat in STAT_MAP.get("player_match", []):
            try:
                df = _with_backoff(
                    fb.read_player_match_stats,
                    max_retries=max_retries,
                    base_delay=backoff_base,
                    kwargs={"stat_type": stat},
                    optional_kw={"force_cache": force_cache},
                    count_as_network=not force_cache,
                    periodic_every=periodic_every,
                    periodic_secs=periodic_secs,
                )
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
            safe_write(df, out_dir / "player_match" / stat)
            _snooze(delay)

    # 3) Team-match stats (aggregate-first options + direct fallback)
    if levels in ("team", "both"):
        for stat in STAT_MAP.get("team_match", []):
            wrote = False

            # Option A: aggregate path (zero direct hits)
            if team_mode in ("aggregate", "auto"):
                ok, df = _try_team_aggregate(fb, stat, league, season_str)
                if ok:
                    safe_write(df, out_dir / f"team_match_{stat}")
                    _snooze(delay)
                    wrote = True

            # Option B: direct call if needed/allowed
            if not wrote and team_mode in ("direct", "auto"):
                try:
                    df = _with_backoff(
                        fb.read_team_match_stats,
                        max_retries=max_retries,
                        base_delay=backoff_base,
                        kwargs={"stat_type": stat, "opponent_stats": False},
                        optional_kw={"force_cache": force_cache},
                        count_as_network=not force_cache,
                        periodic_every=periodic_every,
                        periodic_secs=periodic_secs,
                    )
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        raise ValueError("empty_or_invalid")
                    df = _normalize(df, league, season_str)
                except Exception as e:
                    logging.getLogger("fbref").warning(
                        "Direct team_match %s failed for %s %s: %s",
                        stat, league, season_str, e,
                    )
                    # Try aggregate as a fallback even in direct mode (last chance)
                    try:
                        ok, df = _try_team_aggregate(fb, stat, league, season_str)
                        if not ok:
                            raise RuntimeError("aggregate_fallback_empty")
                    except Exception as e2:
                        logging.getLogger("fbref").warning(
                            "Aggregate fallback failed for team_match %s: %s — writing schema-only CSV",
                            stat, e2,
                        )
                        df = _schema_only_from_previous(fb_prev, "team_match", stat)
                        df = _normalize(df, league, season_str)

                safe_write(df, out_dir / f"team_match_{stat}")
                _snooze(delay)
                wrote = True

            # Option C: if still not written (aggregate-only mode but aggregate failed)
            if not wrote:
                logging.getLogger("fbref").warning(
                    "team_match %s not available via aggregate and direct disabled (team_mode=%s) — writing schema-only CSV",
                    stat, team_mode,
                )
                df = _schema_only_from_previous(fb_prev, "team_match", stat)
                df = _normalize(df, league, season_str)
                safe_write(df, out_dir / f"team_match_{stat}")
                _snooze(delay)

# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    p = argparse.ArgumentParser("Scrape FBref match-level tables")
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
    p.add_argument(
        "--periodic-every",
        type=int,
        default=8,
        help="Take a longer cool-down every N network calls (approximate).",
    )
    p.add_argument(
        "--periodic-secs",
        type=float,
        default=15.0,
        help="Cool-down length in seconds for the periodic rest.",
    )
    p.add_argument(
        "--team-mode",
        choices=["aggregate", "auto", "direct"],
        default="aggregate",
        help="How to build team_match tables: aggregate (no direct hits), auto (aggregate→direct), or direct.",
     )
    p.add_argument(
        "--levels",
        choices=["player", "team", "both"],
        default="both",
        help="Which levels to scrape this run.",
    )
    p.add_argument(
        "--proxy",
        type=str,
        default=None,
        help='Optional proxy passed to soccerdata (e.g., "tor" or "http://user:pass@host:port").',
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

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
            periodic_every=args.periodic_every,
            periodic_secs=args.periodic_secs,
            team_mode=args.team_mode,
            proxy=args.proxy,
            levels=args.levels,
        )

if __name__ == "__main__":
    main()
