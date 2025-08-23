from __future__ import annotations

import argparse
import logging
import warnings
import re, os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import soccerdata as sd

os.environ.setdefault("SOCCERDATA_LOGLEVEL", "ERROR")
os.environ.setdefault("SOCCERDATA_DIR", str(Path("data/_soccerdata_cache").absolute()))

from scripts.fbref_pipeline.utils.fbref_utils import (
    safe_write,          # safe_write(df, PathWithoutExt)
    seasons_from_league, # utility to enumerate seasons when --seasons not provided
    init_logger,         # your logger config
    polite_sleep,        # rate limiter between requests
)

# ───────────────────────── Warning hygiene ─────────────────────────

warnings.filterwarnings("ignore", category=FutureWarning, module=r".*soccerdata.*")

# ───────────────────────── Season helpers ─────────────────────────

def _to_ws_season_int(season: Union[str, int]) -> int:
    """
    Convert season to WhoScored's integer form:
      '2025-2026' -> 2025
      '2024-25'   -> 2024
      '25-26'     -> 2025  (assume 2000s)
      2025        -> 2025
    """
    if isinstance(season, int):
        return season

    s = str(season).strip()

    # 'YYYY-YYYY' -> YYYY
    m = re.fullmatch(r"(\d{4})-(\d{4})", s)
    if m:
        return int(m.group(1))

    # 'YYYY-YY' -> YYYY
    m = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if m:
        return int(m.group(1))

    # 'YY-YY' -> 2000+YY (assume 2000s)
    m = re.fullmatch(r"(\d{2})-(\d{2})", s)
    if m:
        return 2000 + int(m.group(1))

    # 'YYYY' -> YYYY
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        return int(m.group(1))

    # Last resort: try int conversion
    try:
        return int(s)
    except Exception:
        raise ValueError(f"Unrecognized season format: {season!r}")

def _prev_ws_season_int(season_int: int) -> int:
    return season_int - 1

# ───────────────────────── DataFrame helpers ─────────────────────────

def _normalize(df: pd.DataFrame, league: str, season_int: int) -> pd.DataFrame:
    """
    Ensure a flat frame and required columns.
    """
    if isinstance(df.index, (pd.MultiIndex, pd.Index)):
        try:
            df = df.reset_index()
        except Exception:
            pass
    if "season" not in df.columns:
        df["season"] = season_int
    else:
        df["season"] = df["season"].fillna(season_int)
    if "league" not in df.columns:
        df["league"] = league
    else:
        df["league"] = df["league"].fillna(league)
    return df

def _schema_only_from_previous(
    ws_prev: Optional[sd.WhoScored],
    level: str,  # "schedule" | "missing_players"
) -> pd.DataFrame:
    """
    Build an empty DF with columns cloned from the previous season; fallback to a minimal schema.
    """
    if ws_prev is not None:
        try:
            if level == "schedule":
                prev = ws_prev.read_schedule()
            else:
                prev = ws_prev.read_missing_players()
            if isinstance(prev, pd.DataFrame) and not prev.empty:
                prev = prev.reset_index(drop=False)
                return pd.DataFrame(columns=list(prev.columns))
        except Exception:
            pass

    # Minimal schemas if even previous-season fetch failed
    if level == "schedule":
        base_cols = [
            "league", "season", "stage_id", "game_id", "status", "start_time",
            "home_team_id", "home_team", "away_team_id", "away_team", "date"
        ]
    else:  # missing_players
        base_cols = [
            "league", "season", "game", "team", "player",
            "game_id", "player_id", "reason", "status"
        ]
    return pd.DataFrame(columns=base_cols)

# ───────────────────────── WhoScored session ─────────────────────────

def _build_ws(
    league: str,
    season_int: int,
    proxy: Optional[Union[str, dict, list]] = None,
    no_cache: bool = False,
    no_store: bool = False,
    path_to_browser: Optional[str] = None,
    headless: bool = True,
) -> sd.WhoScored:
    kwargs = dict(
        leagues=league,
        seasons=season_int,   # IMPORTANT: integer season for WhoScored
        proxy=proxy,
        no_cache=no_cache,
        no_store=no_store,
        headless=headless,
    )
    if path_to_browser:
        kwargs["path_to_browser"] = path_to_browser
    return sd.WhoScored(**kwargs)

def _close_ws(ws: sd.WhoScored) -> None:
    """
    Try to explicitly close the underlying browser/driver to avoid destructor warnings on Windows.
    """
    for attr in ("_browser", "browser", "driver", "_driver", "session"):
        try:
            obj = getattr(ws, attr, None)
            if obj and hasattr(obj, "quit"):
                obj.quit()
        except Exception:
            pass
    try:
        if hasattr(ws, "close"):
            ws.close()  # type: ignore[attr-defined]
    except Exception:
        pass

# ───────────────────────── Core scraper ─────────────────────────

def scrape_one(
    league: str,
    season: Union[str, int],
    out_base: Path,
    delay: float,
    proxy: Optional[Union[str, dict, list]],
    no_cache: bool,
    no_store: bool,
    path_to_browser: Optional[str],
    headless: bool,
):
    log = logging.getLogger("whoscored")

    season_int = _to_ws_season_int(season)
    prev_int   = _prev_ws_season_int(season_int)

    log.info("WhoScored missing-players: %s %s", league, season_int)

    ws = _build_ws(
        league, season_int, proxy=proxy, no_cache=no_cache, no_store=no_store,
        path_to_browser=path_to_browser, headless=headless
    )
    try:
        ws_prev = _build_ws(
            league, prev_int, proxy=proxy, no_cache=no_cache, no_store=no_store,
            path_to_browser=path_to_browser, headless=headless
        )
    except Exception:
        ws_prev = None

    out_dir = out_base / "WhoScored" / league / str(season_int)

    try:
        # 1) Schedule
        try:
            schedule_df = ws.read_schedule()
            if not isinstance(schedule_df, pd.DataFrame) or schedule_df.empty:
                raise ValueError("empty_or_invalid")
            schedule_df = _normalize(schedule_df, league, season_int)
        except Exception as e:
            log.warning(
                "Missing/unavailable WhoScored schedule %s %s: %s — writing schema-only CSV",
                league, season_int, e
            )
            schedule_df = _schema_only_from_previous(ws_prev, "schedule")
            schedule_df = _normalize(schedule_df, league, season_int)
        safe_write(schedule_df, out_dir / "ws_schedule")
        polite_sleep(delay)

        # 2) Missing players (injuries/suspensions)
        try:
            mp_df = ws.read_missing_players()
            if not isinstance(mp_df, pd.DataFrame) or mp_df.empty:
                raise ValueError("empty_or_invalid")
            mp_df = _normalize(mp_df, league, season_int)
        except Exception as e:
            log.warning(
                "Missing/unavailable missing_players %s %s: %s — writing schema-only CSV",
                league, season_int, e
            )
            mp_df = _schema_only_from_previous(ws_prev, "missing_players")
            mp_df = _normalize(mp_df, league, season_int)

        # Ensure canonical columns exist for downstream merges
        for col in ["game_id", "player_id", "reason", "status"]:
            if col not in mp_df.columns:
                mp_df[col] = pd.Series(dtype="object")

        safe_write(mp_df, out_dir / "missing_players")
        polite_sleep(delay)

    finally:
        try:
            _close_ws(ws)
        except Exception:
            pass
        try:
            del ws
        except Exception:
            pass
        if ws_prev is not None:
            try:
                _close_ws(ws_prev)
            except Exception:
                pass
            try:
                del ws_prev
            except Exception:
                pass

# ───────────────────────── CLI ─────────────────────────

def main():
    parser = argparse.ArgumentParser(
        "Scrape WhoScored schedule and missing players (injuries/suspensions)"
    )
    parser.add_argument("--league", default="ENG-Premier League")
    parser.add_argument("--out-dir", default="data/raw/whoscored")
    parser.add_argument("--seasons", nargs="*", help="Accepts 'YYYY-YYYY', 'YYYY-YY', 'YY-YY' or a single year like 2025")
    parser.add_argument("--delay", type=float, default=0.75)
    parser.add_argument("--proxy", default=None, help="e.g. 'tor' or http://user:pass@host:port")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-store", action="store_true")
    parser.add_argument("--browser", dest="path_to_browser", default=None, help="Path to Chrome executable")
    parser.add_argument("--headless", dest="headless", action="store_true", default=True)
    parser.add_argument("--headed", dest="headless", action="store_false", help="Run Chrome with a visible window")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    init_logger(args.verbose)

    # If seasons not provided, derive from league (likely "YYYY-YYYY"); we'll normalize to integers.
    seasons = args.seasons or seasons_from_league(args.league)

    out_base = Path(args.out_dir)
    for s in seasons:
        scrape_one(
            league=args.league,
            season=s,
            out_base=out_base,
            delay=args.delay,
            proxy=args.proxy,
            no_cache=args.no_cache,
            no_store=args.no_store,
            path_to_browser=args.path_to_browser,
            headless=args.headless,
        )

if __name__ == "__main__":
    main()
