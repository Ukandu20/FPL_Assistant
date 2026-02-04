#!/usr/bin/env python3
# scripts/fbref_pipeline/scrape/whoscored_match_stats_scraper.py
"""
WhoScored match-level scraper (cache-first).

Tables:
- schedule
- missing_players
- events (default), raw, spadl, atomic-spadl

Notes:
- WhoScored seasons are integer-based (e.g., 2025 for 2025-2026).
- This uses selenium via soccerdata.WhoScored.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
import soccerdata as sd

os.environ.setdefault("SOCCERDATA_LOGLEVEL", "ERROR")
os.environ.setdefault("SOCCERDATA_DIR", str(Path("data/_soccerdata_cache").absolute()))

from scripts.fbref_pipeline.automation.auto_scrape import ScrapeJobId, record_last_run
from scripts.fbref_pipeline.utils.fbref_utils import (
    init_logger,
    polite_sleep,
    safe_write,
    seasons_from_league,
)

warnings.filterwarnings("ignore", category=FutureWarning, module=r".*soccerdata.*")

ALL_KNOWN_LEAGUES: List[str] = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]

TABLE_CHOICES = ["schedule", "missing_players", "events"]
EVENT_FORMATS = ["events", "raw", "spadl", "atomic-spadl", "none"]

COLS_EVENTS = [
    "game_id",
    "period",
    "minute",
    "second",
    "expanded_minute",
    "type",
    "outcome_type",
    "team_id",
    "team",
    "player_id",
    "player",
    "x",
    "y",
    "end_x",
    "end_y",
    "goal_mouth_y",
    "goal_mouth_z",
    "blocked_x",
    "blocked_y",
    "qualifiers",
    "is_touch",
    "is_shot",
    "is_goal",
    "card_type",
    "related_event_id",
    "related_player_id",
]


def _parse_list(raw: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not raw:
        return None
    out: List[str] = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out or None


def _to_ws_season_int(season: Union[str, int]) -> int:
    if isinstance(season, int):
        return season

    s = str(season).strip()
    m = re.fullmatch(r"(\d{4})-(\d{4})", s)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"(\d{2})-(\d{2})", s)
    if m:
        return 2000 + int(m.group(1))
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        return int(m.group(1))
    try:
        return int(s)
    except Exception as e:
        raise ValueError(f"Unrecognized season format: {season!r}") from e


def _normalize(df: pd.DataFrame, league: str, season_int: int) -> pd.DataFrame:
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


def _schema_only_schedule() -> pd.DataFrame:
    cols = [
        "league",
        "season",
        "stage_id",
        "game_id",
        "status",
        "start_time",
        "home_team_id",
        "home_team",
        "away_team_id",
        "away_team",
        "date",
    ]
    return pd.DataFrame(columns=cols)


def _schema_only_missing_players() -> pd.DataFrame:
    cols = [
        "league",
        "season",
        "game",
        "team",
        "player",
        "game_id",
        "player_id",
        "reason",
        "status",
    ]
    return pd.DataFrame(columns=cols)


def _schema_only_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["game"] + COLS_EVENTS)


def _build_ws(
    league: str,
    season_int: int,
    *,
    proxy: Optional[Union[str, dict, list]] = None,
    no_cache: bool = False,
    no_store: bool = False,
    path_to_browser: Optional[str] = None,
    headless: bool = True,
) -> sd.WhoScored:
    kwargs: Dict[str, Any] = dict(
        leagues=league,
        seasons=season_int,
        proxy=proxy,
        no_cache=no_cache,
        no_store=no_store,
        headless=headless,
    )
    if path_to_browser:
        kwargs["path_to_browser"] = path_to_browser
    return sd.WhoScored(**kwargs)


def _close_ws(ws: sd.WhoScored) -> None:
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


def _csv_exists(base: Path) -> bool:
    if base.is_file():
        return True
    if base.suffix == "":
        csv_path = base.with_suffix(".csv")
        if csv_path.is_file():
            return True
    return False


def _events_out_path(events_dir: Path, fmt: str) -> Path:
    if fmt == "events":
        return events_dir / "events"
    if fmt == "raw":
        return events_dir / "raw"
    if fmt == "spadl":
        return events_dir / "spadl"
    if fmt == "atomic-spadl":
        return events_dir / "atomic_spadl"
    return events_dir / "events"


def scrape_one(
    *,
    league: str,
    season: Union[str, int],
    out_base: Path,
    delay: float,
    proxy: Optional[Union[str, dict, list]],
    no_cache: bool,
    no_store: bool,
    force_cache: bool,
    path_to_browser: Optional[str],
    headless: bool,
    tables: Sequence[str],
    events_format: str,
    match_ids: Optional[Sequence[int]],
    skip_existing: bool,
    live: bool,
    retry_missing: bool,
    on_error: str,
) -> Dict[str, Any]:
    log = logging.getLogger("whoscored")

    season_int = _to_ws_season_int(season)
    log.info("WhoScored scrape: %s %s", league, season_int)

    ws = _build_ws(
        league,
        season_int,
        proxy=proxy,
        no_cache=no_cache,
        no_store=no_store,
        path_to_browser=path_to_browser,
        headless=headless,
    )

    out_dir = out_base / "WhoScored" / league / str(season_int)
    events_dir = out_dir / "events"

    summary: Dict[str, Any] = {
        "schedule": None,
        "missing_players": None,
        "events": None,
        "events_format": events_format,
    }

    try:
        if "schedule" in tables:
            out_path = out_dir / "ws_schedule"
            if skip_existing and _csv_exists(out_path):
                summary["schedule"] = "skipped_existing"
            else:
                try:
                    schedule_df = ws.read_schedule(force_cache=force_cache)
                    if not isinstance(schedule_df, pd.DataFrame) or schedule_df.empty:
                        raise ValueError("empty_or_invalid")
                    schedule_df = _normalize(schedule_df, league, season_int)
                    safe_write(schedule_df, out_path)
                    summary["schedule"] = "scraped_ok"
                except Exception as e:
                    log.warning(
                        "Failed to scrape schedule for %s %s: %s",
                        league,
                        season_int,
                        e,
                    )
                    empty = _schema_only_schedule()
                    empty = _normalize(empty, league, season_int)
                    safe_write(empty, out_path)
                    summary["schedule"] = "schema_only"
                polite_sleep(delay)

        if "missing_players" in tables:
            out_path = out_dir / "missing_players"
            if skip_existing and _csv_exists(out_path):
                summary["missing_players"] = "skipped_existing"
            else:
                try:
                    mp_df = ws.read_missing_players(match_id=match_ids, force_cache=force_cache)
                    if not isinstance(mp_df, pd.DataFrame) or mp_df.empty:
                        raise ValueError("empty_or_invalid")
                    mp_df = _normalize(mp_df, league, season_int)
                    for col in ["game_id", "player_id", "reason", "status"]:
                        if col not in mp_df.columns:
                            mp_df[col] = pd.Series(dtype="object")
                    safe_write(mp_df, out_path)
                    summary["missing_players"] = "scraped_ok"
                except Exception as e:
                    log.warning(
                        "Failed to scrape missing_players for %s %s: %s",
                        league,
                        season_int,
                        e,
                    )
                    empty = _schema_only_missing_players()
                    empty = _normalize(empty, league, season_int)
                    safe_write(empty, out_path)
                    summary["missing_players"] = "schema_only"
                polite_sleep(delay)

        if "events" in tables:
            if events_format == "none":
                ws.read_events(
                    match_id=match_ids,
                    force_cache=force_cache,
                    live=live,
                    output_fmt=None,
                    retry_missing=retry_missing,
                    on_error=on_error,
                )
                summary["events"] = "cached_only"
            else:
                out_path = _events_out_path(events_dir, events_format)
                if skip_existing and (
                    _csv_exists(out_path)
                    or out_path.with_suffix(".json").is_file()
                ):
                    summary["events"] = "skipped_existing"
                else:
                    try:
                        ev = ws.read_events(
                            match_id=match_ids,
                            force_cache=force_cache,
                            live=live,
                            output_fmt=events_format,
                            retry_missing=retry_missing,
                            on_error=on_error,
                        )
                        if events_format == "raw":
                            payload = ev if isinstance(ev, dict) else {}
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            out_path.with_suffix(".json").write_text(
                                json.dumps(payload, ensure_ascii=True, indent=2),
                                encoding="utf-8",
                            )
                        else:
                            df = ev if isinstance(ev, pd.DataFrame) else pd.DataFrame()
                            if df.empty and events_format == "events":
                                df = _schema_only_events()
                            df = _normalize(df, league, season_int)
                            safe_write(df, out_path)
                        summary["events"] = "scraped_ok"
                    except Exception as e:
                        log.warning(
                            "Failed to scrape events for %s %s: %s",
                            league,
                            season_int,
                            e,
                        )
                        if events_format == "raw":
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            out_path.with_suffix(".json").write_text("{}", encoding="utf-8")
                        else:
                            df = _schema_only_events() if events_format == "events" else pd.DataFrame()
                            df = _normalize(df, league, season_int)
                            safe_write(df, out_path)
                        summary["events"] = "schema_only"
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

    return summary


def main() -> None:
    p = argparse.ArgumentParser("Scrape WhoScored match-level data (schedule, missing_players, events)")
    p.add_argument(
        "--league",
        nargs="+",
        default=["ENG-Premier League"],
        help="One or more league identifiers (repeatable).",
    )
    p.add_argument(
        "--all-known-leagues",
        action="store_true",
        help="Ignore --league and scrape all leagues listed in ALL_KNOWN_LEAGUES.",
    )
    p.add_argument(
        "--seasons",
        nargs="*",
        help="Seasons to scrape (e.g. 2025-2026, 2024-25, 2025).",
    )
    p.add_argument("--out-dir", default="data/raw/whoscored")
    p.add_argument(
        "--tables",
        nargs="+",
        choices=TABLE_CHOICES,
        default=TABLE_CHOICES,
        help="Subset of tables to scrape.",
    )
    p.add_argument(
        "--events-format",
        choices=EVENT_FORMATS,
        default="events",
        help="Events output format.",
    )
    p.add_argument("--match-ids", action="append", help="Match IDs (comma-separated ok).")
    p.add_argument("--delay", type=float, default=0.75)
    p.add_argument("--proxy", default=None, help="e.g. 'tor' or http://user:pass@host:port")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-store", action="store_true")
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--browser", dest="path_to_browser", default=None, help="Path to Chrome executable")
    p.add_argument("--headless", dest="headless", action="store_true", default=True)
    p.add_argument("--headed", dest="headless", action="store_false", help="Run Chrome with a visible window")
    p.add_argument("--skip-existing", action="store_true", help="Skip outputs that already exist.")
    p.add_argument("--live", action="store_true", help="Bypass cached events for live data.")
    p.add_argument("--retry-missing", dest="retry_missing", action="store_true", default=True)
    p.add_argument("--no-retry-missing", dest="retry_missing", action="store_false")
    p.add_argument(
        "--on-error",
        choices=["raise", "skip"],
        default="raise",
        help="Whether to raise or skip errors during events scraping.",
    )
    p.add_argument(
        "--meta-path",
        type=Path,
        default=Path("data/meta/scraper_runs.json"),
        help="Meta JSON path for scraper runs.",
    )
    p.add_argument(
        "--run-mode",
        choices=["manual", "automation"],
        default="manual",
        help="Meta mode tag.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    init_logger(args.verbose)
    log = logging.getLogger("whoscored")

    if args.no_cache and args.force_cache:
        raise SystemExit("--no-cache and --force-cache are mutually exclusive.")

    leagues = ALL_KNOWN_LEAGUES if args.all_known_leagues else args.league
    match_ids_raw = _parse_list(args.match_ids)
    match_ids = [int(x) for x in match_ids_raw] if match_ids_raw else None

    out_base = Path(args.out_dir)
    meta_path = Path(args.meta_path)

    for league in leagues:
        seasons = args.seasons or seasons_from_league(league)
        log.info("Resolved seasons for %s: %s", league, seasons)
        for s in seasons:
            summary = scrape_one(
                league=league,
                season=s,
                out_base=out_base,
                delay=args.delay,
                proxy=args.proxy,
                no_cache=args.no_cache,
                no_store=args.no_store,
                force_cache=args.force_cache,
                path_to_browser=args.path_to_browser,
                headless=args.headless,
                tables=args.tables,
                events_format=args.events_format,
                match_ids=match_ids,
                skip_existing=args.skip_existing,
                live=args.live,
                retry_missing=args.retry_missing,
                on_error=args.on_error,
            )

            job = ScrapeJobId(
                scraper="whoscored_match",
                league=league,
                season=str(_to_ws_season_int(s)),
                levels="tables",
            )
            run_info = {
                "scrape_ts": datetime.now(timezone.utc).isoformat(),
                "mode": args.run_mode,
                "stats_summary": summary,
                "tables": list(args.tables),
                "events_format": args.events_format,
            }
            record_last_run(meta_path, job, run_info=run_info)
            log.info("Recorded last-run meta for %s", job.key())


if __name__ == "__main__":
    main()
