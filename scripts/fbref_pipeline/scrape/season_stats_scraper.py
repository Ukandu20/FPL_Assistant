#!/usr/bin/env python3
# scripts/fbref_pipeline/scrape/season_stats_scraper.py
"""
FBref season-level scraper (network-friendly, cache-first)

Key bits:
- Exponential backoff (429-aware, honors Retry-After) + jittered sleeps + periodic cool-downs
- Cache-first defaults (no_cache=False) + --force-cache for offline dev runs
- Previous-season schema discovery is cache-only
- Split layers with --levels {player,team,both}
- Team season via aggregate-from-player (no extra hits) with --team-mode aggregate|auto|direct
- NEW: Safe defaults for STAT_MAP; loud write logs; selectable output layout via --layout flat|folders
- NEW: --echo-config prints the resolved stat lists and output dir

Typical usage:
  # Player-only (prime cache)
  python scripts/fbref_pipeline/scrape/season_stats_scraper.py \
    --league "ENG-Premier League" --seasons 2025-2026 \
    --levels player --delay 3.6 --periodic-every 6 --periodic-secs 25 --echo-config

  # Team-only from player (no extra hits if cache/files exist)
  python scripts/fbref_pipeline/scrape/season_stats_scraper.py \
    --league "ENG-Premier League" --seasons 2025-2026 \
    --levels team --team-mode aggregate --force-cache --layout folders
"""

from __future__ import annotations

import argparse
import logging
import warnings
import time
import random
from pathlib import Path
from typing import Optional, Callable, Any, Dict, Tuple, Sequence

import pandas as pd
import numpy as np
import soccerdata as sd
from requests.exceptions import HTTPError

from scripts.fbref_pipeline.utils.fbref_utils import (
    STAT_MAP,          # expects keys: "team_season", "player_season" (may be missing; we add defaults)
    safe_write,        # safe_write(df, PathWithoutExt)
    seasons_from_league,
    init_logger,
    polite_sleep,
)

# ───────────────────────── config & defaults ─────────────────────────

# Reasonable defaults if STAT_MAP lacks season-level entries (prevents "no-op" runs)
PLAYER_SEASON_DEFAULTS: Sequence[str] = (
    "standard", "shooting", "passing", "passing_types",
    "defense", "possession", "misc", "gca", "xg", "keepers", "keepersadv"
)
TEAM_SEASON_DEFAULTS: Sequence[str] = (
    "standard", "shooting", "passing", "passing_types",
    "defense", "possession", "misc", "gca", "xg", "keepers", "keepersadv"
)

_GLOBAL = {"net_calls": 0}

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

def _snooze(base: float) -> None:
    jitter = base * random.uniform(0.2, 0.4)
    polite_sleep(base + jitter)

def _periodic_rest(every: int, secs: float, *, skip: bool) -> None:
    if skip:
        return
    if every > 0 and _GLOBAL["net_calls"] > 0 and (_GLOBAL["net_calls"] % every == 0):
        logging.getLogger("fbref").info("Cooling down: resting for %.1fs after %d calls", secs, _GLOBAL["net_calls"])
        time.sleep(secs)

def _parse_retry_after(exc: Exception) -> Optional[float]:
    resp = getattr(exc, "response", None)
    if not resp or not hasattr(resp, "headers"):
        return None
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        return float(ra)  # seconds
    except Exception:
        return None

def _call_with_optional_kw(fn: Callable[..., Any], kwargs: Dict[str, Any], optional_kw: Optional[Dict[str, Any]] = None):
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
    count_as_network: bool = True,
    periodic_every: int = 8,
    periodic_secs: float = 15.0,
) -> Any:
    if kwargs is None:
        kwargs = {}

    delay = base_delay
    last_exc: Optional[Exception] = None
    log = logging.getLogger("fbref")

    for attempt in range(max_retries):
        try:
            result = _call_with_optional_kw(fn, kwargs=kwargs, optional_kw=optional_kw)
            if count_as_network:
                _GLOBAL["net_calls"] += 1
            _periodic_rest(periodic_every, periodic_secs, skip=not count_as_network)
            return result
        except Exception as e:
            last_exc = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            msg = str(e).lower()
            is_429 = (isinstance(e, HTTPError) and status == 429) or ("429" in msg) or ("too many requests" in msg)
            if is_429:
                ra = _parse_retry_after(e)
                sleep_for = (float(ra) if ra else min(delay, 60.0)) + random.random()
                log.warning("HTTP 429; backing off for %.1fs (attempt %d/%d)", sleep_for, attempt + 1, max_retries)
                time.sleep(sleep_for)
                delay *= 1.8
                continue
            raise

    raise last_exc if last_exc else RuntimeError("Transient error: retries exhausted without a captured exception.")

def _schema_only_from_previous(
    fb_prev: sd.FBref | None,
    level: str,      # "team_season" | "player_season"
    stat: str,
) -> pd.DataFrame:
    if fb_prev is not None:
        try:
            if level == "team_season":
                prev = _call_with_optional_kw(
                    fb_prev.read_team_season_stats, kwargs={"stat_type": stat}, optional_kw={"force_cache": True}
                )
            else:
                prev = _call_with_optional_kw(
                    fb_prev.read_player_season_stats, kwargs={"stat_type": stat}, optional_kw={"force_cache": True}
                )
            if isinstance(prev, pd.DataFrame) and not prev.empty:
                return pd.DataFrame(columns=list(prev.columns))
        except Exception:
            pass

    base_cols = ["season", "league", "team"] if level == "team_season" else ["season", "league", "player"]
    return pd.DataFrame(columns=base_cols)

# ────────── aggregate team-season from player-season (no extra hits) ──────────

_EXCLUDE_RATE_TOKENS = (
    "per90", "per_90", "per-90", " per 90", "90s", "rate", "pct", "%", "share",
    "/90", "per match", "per_match", "per-game", "per game", "p90"
)

def _likely_rate(col: str) -> bool:
    c = col.lower()
    return any(tok in c for tok in _EXCLUDE_RATE_TOKENS)

def _find_team_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ("team", "squad", "Team", "Squad"):
        if cand in df.columns:
            return cand
    return None

def _aggregate_team_from_player(p_df: pd.DataFrame, league: str, season: str) -> Optional[pd.DataFrame]:
    team_col = _find_team_col(p_df)
    if team_col is None:
        return None
    p_df = _normalize(p_df, league, season).copy()
    numeric_cols = [c for c in p_df.columns if pd.api.types.is_numeric_dtype(p_df[c])]
    keep_cols = [c for c in numeric_cols if not _likely_rate(c)]
    if not keep_cols:
        return None
    grouped = (
        p_df.groupby(["season", "league", team_col], dropna=False)[keep_cols]
        .sum(min_count=1)
        .reset_index()
    )
    if team_col != "team":
        grouped = grouped.rename(columns={team_col: "team"})
    return grouped

def _try_team_aggregate(
    fb: sd.FBref,
    stat: str,
    league: str,
    season_str: str,
    *,
    force_cache: bool,
    backoff_base: float,
    max_retries: int,
    periodic_every: int,
    periodic_secs: float,
) -> Tuple[bool, pd.DataFrame]:
    log = logging.getLogger("fbref")
    try:
        p_df = _with_backoff(
            fb.read_player_season_stats,
            max_retries=max_retries,
            base_delay=backoff_base,
            kwargs={"stat_type": stat},
            optional_kw={"force_cache": force_cache},
            count_as_network=not force_cache,
            periodic_every=periodic_every,
            periodic_secs=periodic_secs,
        )
        if not isinstance(p_df, pd.DataFrame) or p_df.empty:
            return False, pd.DataFrame()
        grouped = _aggregate_team_from_player(p_df, league, season_str)
        if grouped is None or grouped.empty:
            return False, pd.DataFrame()
        log.info("✓ team_season %s built via aggregate from player season (no direct hits)", stat)
        return True, grouped
    except Exception as e:
        log.debug("Aggregate team season for %s failed: %s", stat, e)
        return False, pd.DataFrame()

# ───────────────────────── core ─────────────────────────

def _resolve_stats() -> tuple[list[str], list[str]]:
    player_stats = list(STAT_MAP.get("player_season") or PLAYER_SEASON_DEFAULTS)
    team_stats = list(STAT_MAP.get("team_season") or TEAM_SEASON_DEFAULTS)
    return player_stats, team_stats

def _save_with_log(df: pd.DataFrame, base: Path, label: str, *, layout: str, log: logging.Logger) -> None:
    """
    Write CSV and log absolute final path + rows.
    layout='flat'  -> base like <...>/player_season_standard  -> writes that .csv
    layout='folders' -> base like <...>/player_season/standard -> writes that .csv
    """
    if layout == "folders":
        # If label contains an underscore separation already, keep it simple
        parts = label.split("_", 1)
        folder = parts[0]
        name = parts[1] if len(parts) > 1 else parts[0]
        path_base = base / folder / name
    else:
        path_base = base / label

    path_abs = path_base.with_suffix(".csv")
    path_abs.parent.mkdir(parents=True, exist_ok=True)
    safe_write(df, path_base)
    log.info("WROTE: %s  (rows=%d, cols=%d)", path_abs.resolve(), int(len(df)), int(df.shape[1]))

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
    levels: str = "both",          # 'player' | 'team' | 'both'
    layout: str = "flat",          # 'flat' | 'folders'
    echo_config: bool = False,
) -> None:
    log = logging.getLogger("fbref")
    season_str = str(season)
    out_dir = (out_base / league / season_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    fb_kwargs = dict(leagues=league, seasons=season_str, no_cache=no_cache)
    if proxy:
        fb_kwargs["proxy"] = proxy
    fb = sd.FBref(**fb_kwargs)

    try:
        fb_prev = sd.FBref(leagues=league, seasons=_prev_season_str(season_str))
    except Exception:
        fb_prev = None

    player_stats, team_stats = _resolve_stats()

    if echo_config:
        log.info("Output dir: %s", out_dir.resolve())
        log.info("Levels: %s | Team mode: %s | Layout: %s", levels, team_mode, layout)
        log.info("Player season stats: %s", ", ".join(player_stats))
        log.info("Team season stats:   %s", ", ".join(team_stats))

    # 1) Player season (when enabled)
    if levels in ("player", "both"):
        for stat in player_stats:
            try:
                df = _with_backoff(
                    fb.read_player_season_stats,
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
                    "Missing/unavailable player_season %s for %s %s: %s — writing schema-only CSV",
                    stat, league, season_str, e,
                )
                df = _schema_only_from_previous(fb_prev, "player_season", stat)
                df = _normalize(df, league, season_str)

            _save_with_log(df, out_dir, f"player_season_{stat}", layout=layout, log=log)
            _snooze(delay)

    # 2) Team season (aggregate-first options + direct fallback)
    if levels in ("team", "both"):
        for stat in team_stats:
            wrote = False

            # Option A: aggregate path (prefer)
            if team_mode in ("aggregate", "auto"):
                ok, df = _try_team_aggregate(
                    fb, stat, league, season_str,
                    force_cache=force_cache,
                    backoff_base=backoff_base,
                    max_retries=max_retries,
                    periodic_every=periodic_every,
                    periodic_secs=periodic_secs,
                )
                if ok:
                    _save_with_log(df, out_dir, f"team_season_{stat}", layout=layout, log=log)
                    _snooze(delay)
                    wrote = True

            # Option B: direct call if needed/allowed
            if not wrote and team_mode in ("direct", "auto"):
                try:
                    df = _with_backoff(
                        fb.read_team_season_stats,
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
                    logging.getLogger("fbref").warning(
                        "Direct team_season %s failed for %s %s: %s",
                        stat, league, season_str, e,
                    )
                    # Try aggregate as a last resort even in direct mode
                    try:
                        ok, df = _try_team_aggregate(
                            fb, stat, league, season_str,
                            force_cache=force_cache,
                            backoff_base=backoff_base,
                            max_retries=max_retries,
                            periodic_every=periodic_every,
                            periodic_secs=periodic_secs,
                        )
                        if not ok:
                            raise RuntimeError("aggregate_fallback_empty")
                    except Exception as e2:
                        logging.getLogger("fbref").warning(
                            "Aggregate fallback failed for team_season %s: %s — writing schema-only CSV",
                            stat, e2,
                        )
                        df = _schema_only_from_previous(fb_prev, "team_season", stat)
                        df = _normalize(df, league, season_str)

                _save_with_log(df, out_dir, f"team_season_{stat}", layout=layout, log=log)
                _snooze(delay)
                wrote = True

            # Option C: if still not written (aggregate-only mode but aggregate failed)
            if not wrote:
                logging.getLogger("fbref").warning(
                    "team_season %s not available via aggregate and direct disabled (team_mode=%s) — writing schema-only CSV",
                    stat, team_mode,
                )
                df = _schema_only_from_previous(fb_prev, "team_season", stat)
                df = _normalize(df, league, season_str)
                _save_with_log(df, out_dir, f"team_season_{stat}", layout=layout, log=log)
                _snooze(delay)

# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    p = argparse.ArgumentParser("Scrape FBref season-level tables")
    p.add_argument("--league", default="ENG-Premier League")
    p.add_argument("--out-dir", default="data/raw/fbref")
    p.add_argument("--seasons", nargs="*")
    p.add_argument("--no-cache", action="store_true", help="Bypass local cache (not recommended).")
    p.add_argument("--force-cache", action="store_true", help="Read only from cache; never hit network.")
    p.add_argument("--delay", type=float, default=3.2, help="Base inter-request delay (seconds).")
    p.add_argument("--max-retries", type=int, default=6, help="Max 429 backoff retries per request.")
    p.add_argument("--backoff-base", type=float, default=3.2, help="Initial backoff seconds on 429.")
    p.add_argument("--periodic-every", type=int, default=8, help="Cool-down every N network calls.")
    p.add_argument("--periodic-secs", type=float, default=15.0, help="Cool-down length in seconds.")
    p.add_argument("--team-mode", choices=["aggregate", "auto", "direct"], default="aggregate",
                   help="Team tables: aggregate from player, aggregate→direct, or direct FBref endpoints.")
    p.add_argument("--levels", choices=["player", "team", "both"], default="both", help="Which levels to scrape.")
    p.add_argument("--layout", choices=["flat", "folders"], default="flat",
                   help="Output naming: flat 'player_season_standard.csv' or folders 'player_season/standard.csv'.")
    p.add_argument("--proxy", type=str, default=None, help='Optional proxy for soccerdata (e.g., "tor" or URL).')
    p.add_argument("--echo-config", action="store_true", help="Print resolved stats and output dir before scraping.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning, module=r".*soccerdata\.fbref.*")
    init_logger(args.verbose)

    seasons = args.seasons or seasons_from_league(args.league)
    out_base = Path(args.out_dir)

    for s in seasons:
        logging.getLogger("fbref").info(
            "Season-level: %s %s (team_mode=%s, levels=%s, layout=%s)",
            args.league, s, args.team_mode, args.levels, args.layout
        )
        scrape_one(
            args.league, str(s), out_base, args.delay,
            no_cache=args.no_cache, force_cache=args.force_cache,
            max_retries=args.max_retries, backoff_base=args.backoff_base,
            periodic_every=args.periodic_every, periodic_secs=args.periodic_secs,
            team_mode=args.team_mode, proxy=args.proxy, levels=args.levels,
            layout=args.layout, echo_config=args.echo_config,
        )

if __name__ == "__main__":
    main()
