#!/usr/bin/env python3
# scripts/fbref_pipeline/scrape/match_stats_scraper.py
"""
FBref match-level scraper (network-friendly, cache-first)

Key features:
- Exponential backoff for HTTP 429 / transient errors (honors Retry-After).
- Jittered sleeps between requests to avoid synchronized bursts.
- Cache-first defaults (no_cache=False) + --force-cache for offline dev runs.
- Previous-season schema discovery uses force_cache (never hits network, when supported).
- **Team stats can avoid network entirely**: --team-mode aggregate|auto|direct
    - aggregate  (default): build team tables from player-match (zero extra direct hits)
    - auto: try aggregate first; fall back to direct if aggregate unavailable
    - direct: go to FBref endpoints (may trigger 429 without generous throttling)
- Periodic rest after N network calls to cool down IP.
- Weekly refresh support: --refresh bypasses soccerdata HTTP cache for the run.
- Multi-league support:
    - --league can accept multiple leagues in one go.
    - --all-known-leagues iterates over ALL_KNOWN_LEAGUES constant.
- Fine-grained control:
    - --player-stats / --team-stats let you scrape a subset of stats.
    - --skip-existing:
        * Reuse existing schedule CSV instead of re-scraping.
        * For each stat, skip re-scrape if existing CSV is "up to date":
          it covers all fixtures with date <= (now - 1 day).
        * If existing stat CSV is incomplete (missing any such fixture),
          it is re-scraped.
- NEW: --rerun-failed:
    - Reads the last recorded run for this job key from --meta-path
    - Re-runs ONLY stats in selected buckets (default: schema_only + incomplete_existing)
    - Works for levels=player, team, or both

Meta:
- scrape_one returns a dict describing:
    * latest fixture up to cutoff date
    * cutoff date (now - 1 day)
    * per-level stat status (skipped_up_to_date, scraped_ok,
      incomplete_existing, schema_only)
- main() passes that into record_last_run(..., run_info=...) so your
  scraper_runs.json has rich metadata per league+season.
"""

from __future__ import annotations

import argparse
import logging
import warnings
import time
import random
import json
from pathlib import Path
from typing import Optional, Callable, Any, Dict, Tuple, List, Set

from datetime import datetime, timedelta, timezone

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

from scripts.fbref_pipeline.automation.auto_scrape import ScrapeJobId, record_last_run

# ───────────────────────── known leagues ─────────────────────────

ALL_KNOWN_LEAGUES: List[str] = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]

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
        logging.getLogger("fbref").info(
            "Cooling down: resting for %.1fs after %d calls",
            secs,
            _GLOBAL["net_calls"],
        )
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


def _call_with_optional_kw(
    fn: Callable[..., Any],
    kwargs: Dict[str, Any],
    optional_kw: Optional[Dict[str, Any]] = None,
):
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
            msg = str(e)
            # If the error mentions any of our optional keywords, retry without them.
            if any(f"'{k}'" in msg or k in msg for k in optional_kw.keys()):
                logging.getLogger("fbref").warning(
                    "Optional kwargs %s not accepted by %s; retrying without them. "
                    "If you expected cache-only behaviour, be aware this call "
                    "may hit the network.",
                    list(optional_kw.keys()),
                    getattr(fn, "__name__", repr(fn)),
                )
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
    """
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
                if ra:
                    sleep_for = float(ra) + random.random()
                else:
                    sleep_for = min(delay, 60.0) + random.random()

                log.warning(
                    "HTTP 429; backing off for %.1fs (attempt %d/%d)",
                    sleep_for,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(sleep_for)
                delay *= 1.8
                continue

            raise

    raise last_exc if last_exc else RuntimeError(
        "Transient error: retries exhausted without a captured exception."
    )


def _schema_only_from_previous(
    fb_prev: Optional[sd.FBref],
    level: str,      # "team_match" | "player_match" | "schedule"
    stat: str | None = None,
) -> pd.DataFrame:
    """
    Build an empty DF with previous-season columns; fallback to minimal schema.
    We ONLY read from cache for schema discovery (never hit the network, when
    soccerdata supports force_cache).
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
            pass

    if level == "schedule":
        base_cols = ["season", "league", "date", "home_team", "away_team", "score"]
    elif level == "team_match":
        base_cols = ["season", "league", "team"]
    else:
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
    Returns (ok, df).
    """
    log = logging.getLogger("fbref")
    try:
        log.info("Scraping team_match via aggregate fallback: %s %s (%s)", league, season_str, stat)
        df = team_match_from_soccerdata_fallback(
            fb=fb, stat_type=stat, league=league, season=season_str
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = _normalize(df, league, season_str)
            log.info("✓ team_match %s built via aggregate fallback (no direct team endpoint hits)", stat)
            return True, df
        return False, pd.DataFrame()
    except Exception as e:
        log.debug("Aggregate path for team_match %s failed: %s", stat, e)
        return False, pd.DataFrame()


def _csv_exists(base: Path) -> bool:
    """
    Return True if a CSV for this logical path already exists.

    safe_write usually treats 'path' as a stem and appends '.csv', so we check
    both the bare path and '<path>.csv'.
    """
    if base.is_file():
        return True
    if base.suffix == "":
        csv_path = base.with_suffix(".csv")
        if csv_path.is_file():
            return True
    return False


def _load_existing_csv(base: Path) -> Optional[pd.DataFrame]:
    """
    Load an existing CSV given a logical base path (without extension).
    """
    log = logging.getLogger("fbref")
    candidates: List[Path] = []
    if base.suffix:
        candidates.append(base)
    else:
        candidates.append(base.with_suffix(".csv"))
        candidates.append(base)

    for p in candidates:
        if p.is_file():
            try:
                return pd.read_csv(p)
            except Exception as e:
                log.warning("Failed to read existing CSV %s: %s", p, e)
                return None
    return None


def _fixture_keys_from_stat(stat_df: pd.DataFrame) -> Set[Tuple[pd.Timestamp, str, str]]:
    """
    Build canonical fixture keys from a stat table.
    """
    if {"date", "home_team", "away_team"}.issubset(stat_df.columns):
        df = stat_df[["date", "home_team", "away_team"]].copy()
    elif {"date", "team", "opponent"}.issubset(stat_df.columns):
        df = stat_df[["date", "team", "opponent"]].copy()
        df = df.rename(columns={"team": "home_team", "opponent": "away_team"})
    else:
        return set()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "home_team", "away_team"])

    keys: Set[Tuple[pd.Timestamp, str, str]] = set()
    for _, row in df.iterrows():
        t1, t2 = sorted([str(row["home_team"]), str(row["away_team"])])
        keys.add((row["date"], t1, t2))
    return keys


def _expected_fixtures_up_to_cutoff(
    schedule_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> Tuple[Set[Tuple[pd.Timestamp, str, str]], Optional[Dict[str, object]]]:
    """
    From the full schedule, compute expected fixture keys with date <= cutoff_date
    and the latest fixture meta.
    """
    required = {"date", "home_team", "away_team"}
    if not required.issubset(schedule_df.columns):
        return set(), None

    extra = {"round", "score"}
    available_cols = set(schedule_df.columns)
    cols = list(required | (extra & available_cols))

    df = schedule_df[cols].copy()
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date_parsed"])

    df = df[df["date_parsed"] <= cutoff_date]
    if df.empty:
        return set(), None

    expected: Set[Tuple[pd.Timestamp, str, str]] = set()
    for _, row in df.iterrows():
        t1, t2 = sorted([str(row["home_team"]), str(row["away_team"])])
        expected.add((row["date_parsed"], t1, t2))

    df_latest = df.sort_values(["date_parsed", "home_team", "away_team"])
    latest = df_latest.iloc[-1]

    latest_fixture_meta: Dict[str, object] = {
        "date": latest["date_parsed"].strftime("%Y-%m-%d"),
        "home_team": str(latest["home_team"]),
        "away_team": str(latest["away_team"]),
        "score": str(latest.get("score", "")) if "score" in df.columns else "",
        "round": str(latest.get("round", "")),
    }

    return expected, latest_fixture_meta


def _is_stat_up_to_date(
    stat_df: pd.DataFrame,
    expected_keys: Set[Tuple[pd.Timestamp, str, str]],
    schedule_has_rows: bool,
) -> bool:
    """
    Up to date means stat has rows for every fixture with date <= cutoff_date.
    """
    if not schedule_has_rows:
        return True
    if not expected_keys:
        return False

    observed = _fixture_keys_from_stat(stat_df)
    if not observed:
        return False

    missing = expected_keys - observed
    return len(missing) == 0


def _load_failed_stats_from_meta(
    meta_path: Path,
    job: ScrapeJobId,
    sources: List[str],
) -> Dict[str, List[str]]:
    """
    Returns {"player_match": [...], "team_match": [...]} from last run's stats_summary
    for the given job key.

    sources controls which buckets are considered "failed", e.g.
    ["schema_only", "incomplete_existing"].
    """
    log = logging.getLogger("fbref")
    if not meta_path.exists():
        log.info("rerun-failed: meta file not found at %s", meta_path)
        return {"player_match": [], "team_match": []}

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("rerun-failed: failed to parse meta json %s: %s", meta_path, e)
        return {"player_match": [], "team_match": []}

    block = meta.get(job.key(), {})
    last_run = block.get("last_run", {})
    summary = (last_run.get("stats_summary") or {})
    pm = (summary.get("player_match") or {})
    tm = (summary.get("team_match") or {})

    def collect(bucket_dict: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        for src in sources:
            vals = bucket_dict.get(src) or []
            out.extend([str(x) for x in vals])
        # dedupe preserve order
        seen: Set[str] = set()
        deduped: List[str] = []
        for x in out:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        return deduped

    return {
        "player_match": collect(pm),
        "team_match": collect(tm),
    }


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
    player_stats: Optional[List[str]] = None,
    team_stats: Optional[List[str]] = None,
    skip_existing: bool = False,
    skip_schedule: bool = False,
) -> Dict[str, Any]:
    """
    Scrape one league+season into the raw FBref folder structure.
    """
    log = logging.getLogger("fbref")
    season_str = str(season)

    now_utc = datetime.now(timezone.utc)
    cutoff_date = (now_utc.date() - timedelta(days=1))
    cutoff_ts = pd.Timestamp(cutoff_date)
    cutoff_date_str = cutoff_ts.strftime("%Y-%m-%d")

    log.info(
        "Match-level scrape START: league=%s season=%s "
        "[levels=%s team_mode=%s no_cache=%s force_cache=%s skip_existing=%s cutoff_date=%s]",
        league,
        season_str,
        levels,
        team_mode,
        no_cache,
        force_cache,
        skip_existing,
        cutoff_date_str,
    )

    fb_kwargs = dict(leagues=league, seasons=season_str, no_cache=no_cache)
    if proxy:
        fb_kwargs["proxy"] = proxy
    fb = sd.FBref(**fb_kwargs)

    try:
        fb_prev = sd.FBref(
            leagues=league,
            seasons=_prev_season_str(season_str),
            no_cache=no_cache,
            proxy=proxy,
        )
    except Exception:
        fb_prev = None

    out_dir = out_base / league / season_str
    out_dir.mkdir(parents=True, exist_ok=True)

    player_stats_summary: Dict[str, List[str]] = {
        "skipped_up_to_date": [],
        "scraped_ok": [],
        "incomplete_existing": [],
        "schema_only": [],
    }
    team_stats_summary: Dict[str, List[str]] = {
        "skipped_up_to_date": [],
        "scraped_ok": [],
        "incomplete_existing": [],
        "schema_only": [],
    }

    # 1) Schedule (reuse option)
    schedule_base = out_dir / "schedule"
    schedule_df: pd.DataFrame
    last_match_date_str: Optional[str] = None
    latest_fixture_meta: Optional[Dict[str, Any]] = None

    reused_schedule = False
    if skip_existing and skip_schedule and _csv_exists(schedule_base) and args_skip_schedule:
        log.info(
            "Reusing existing schedule CSV for %s %s (--skip-existing + --skip-schedule).",
            league,
            season_str,
        )
        existing = _load_existing_csv(schedule_base)
        if existing is not None and not existing.empty:
            schedule_df = _normalize(existing, league, season_str)
            reused_schedule = True
        else:
            log.warning(
                "Existing schedule for %s %s is empty/unreadable; falling back to scrape.",
                league,
                season_str,
            )


    if not reused_schedule:
        log.info("Scraping schedule for %s %s", league, season_str)
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
                league,
                season_str,
                e,
            )
            schedule_df = _schema_only_from_previous(fb_prev, "schedule", None)
            schedule_df = _normalize(schedule_df, league, season_str)

        safe_write(schedule_df, schedule_base)
        _snooze(delay)

    # Visibility on schedule date range
    if "date" in schedule_df.columns and not schedule_df.empty:
        try:
            full_dates = pd.to_datetime(schedule_df["date"], errors="coerce")
            full_dates = full_dates[full_dates.notna()]
            if not full_dates.empty:
                log.info(
                    "Schedule rows=%d, full date range [%s -> %s]",
                    len(schedule_df),
                    full_dates.min(),
                    full_dates.max(),
                )
            else:
                log.info("Schedule rows=%d but date column has no valid dates", len(schedule_df))
        except Exception:
            log.info("Schedule rows=%d (date column not parseable)", len(schedule_df))
    else:
        log.info("Schedule rows=%d (no 'date' column found)", len(schedule_df))

    expected_fixture_keys, latest_fixture_meta = _expected_fixtures_up_to_cutoff(
        schedule_df,
        cutoff_ts,
    )
    schedule_has_rows = bool(expected_fixture_keys)

    if latest_fixture_meta is not None:
        last_match_date_str = str(latest_fixture_meta.get("date"))

    # 2) Player-match stats
    if levels in ("player", "both"):
        all_player_stats = STAT_MAP.get("player_match", [])
        all_player_stats_set = set(all_player_stats)

        if player_stats is not None:
            requested = list(dict.fromkeys(player_stats))
            unknown = sorted(set(requested) - all_player_stats_set)
            if unknown:
                raise SystemExit(
                    f"Unknown player_match stat_type(s): {unknown}. "
                    f"Valid options: {sorted(all_player_stats_set)}"
                )
            stats_to_run = [s for s in all_player_stats if s in set(requested)]
        else:
            stats_to_run = all_player_stats

        log.info("player_match stats_to_run=%s", stats_to_run)

        for stat in stats_to_run:
            out_path_base = out_dir / "player_match" / stat

            if skip_existing and _csv_exists(out_path_base):
                existing_df = _load_existing_csv(out_path_base)
                if existing_df is not None and not existing_df.empty:
                    if _is_stat_up_to_date(existing_df, expected_fixture_keys, schedule_has_rows):
                        log.info(
                            "Skipping player_match %s for %s %s "
                            "(existing CSV is complete up to cutoff_date=%s).",
                            stat,
                            league,
                            season_str,
                            cutoff_date_str,
                        )
                        player_stats_summary["skipped_up_to_date"].append(stat)
                        continue
                    else:
                        log.info(
                            "Existing player_match %s for %s %s is NOT complete up to cutoff_date=%s; re-scraping.",
                            stat,
                            league,
                            season_str,
                            cutoff_date_str,
                        )
                        player_stats_summary["incomplete_existing"].append(stat)
                else:
                    log.info(
                        "Existing player_match %s for %s %s is empty/unreadable; re-scraping.",
                        stat,
                        league,
                        season_str,
                    )
                    player_stats_summary["incomplete_existing"].append(stat)

            log.info("Scraping player_match stat=%s for %s %s", stat, league, season_str)
            wrote_schema_only = False
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
                log.info("player_match %s: %d rows", stat, len(df))
            except Exception as e:
                log.warning(
                    "Missing/unavailable player_match %s for %s %s: %s — writing schema-only CSV",
                    stat,
                    league,
                    season_str,
                    e,
                )
                df = _schema_only_from_previous(fb_prev, "player_match", stat)
                df = _normalize(df, league, season_str)
                wrote_schema_only = True

            safe_write(df, out_path_base)
            if wrote_schema_only:
                player_stats_summary["schema_only"].append(stat)
            else:
                player_stats_summary["scraped_ok"].append(stat)
            _snooze(delay)

    # 3) Team-match stats
    if levels in ("team", "both"):
        all_team_stats = STAT_MAP.get("team_match", [])
        all_team_stats_set = set(all_team_stats)

        if team_stats is not None:
            requested = list(dict.fromkeys(team_stats))
            unknown = sorted(set(requested) - all_team_stats_set)
            if unknown:
                raise SystemExit(
                    f"Unknown team_match stat_type(s): {unknown}. "
                    f"Valid options: {sorted(all_team_stats_set)}"
                )
            stats_to_run = [s for s in all_team_stats if s in set(requested)]
        else:
            stats_to_run = all_team_stats

        log.info("team_match stats_to_run=%s", stats_to_run)

        for stat in stats_to_run:
            wrote = False
            wrote_schema_only = False
            out_path_base = out_dir / f"team_match_{stat}"

            if skip_existing and _csv_exists(out_path_base):
                existing_df = _load_existing_csv(out_path_base)
                if existing_df is not None and not existing_df.empty:
                    if _is_stat_up_to_date(existing_df, expected_fixture_keys, schedule_has_rows):
                        log.info(
                            "Skipping team_match %s for %s %s "
                            "(existing CSV is complete up to cutoff_date=%s).",
                            stat,
                            league,
                            season_str,
                            cutoff_date_str,
                        )
                        team_stats_summary["skipped_up_to_date"].append(stat)
                        continue
                    else:
                        log.info(
                            "Existing team_match %s for %s %s is NOT complete up to cutoff_date=%s; re-scraping.",
                            stat,
                            league,
                            season_str,
                            cutoff_date_str,
                        )
                        team_stats_summary["incomplete_existing"].append(stat)
                else:
                    log.info(
                        "Existing team_match %s for %s %s is empty/unreadable; re-scraping.",
                        stat,
                        league,
                        season_str,
                    )
                    team_stats_summary["incomplete_existing"].append(stat)

            if team_mode in ("aggregate", "auto"):
                ok, df = _try_team_aggregate(fb, stat, league, season_str)
                if ok:
                    log.info("team_match %s (aggregate): %d rows", stat, len(df))
                    safe_write(df, out_path_base)
                    _snooze(delay)
                    wrote = True
                    team_stats_summary["scraped_ok"].append(stat)

            if not wrote and team_mode in ("direct", "auto"):
                log.info("Scraping team_match stat=%s via direct endpoint for %s %s", stat, league, season_str)
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
                    log.info("team_match %s (direct): %d rows", stat, len(df))
                except Exception as e:
                    logging.getLogger("fbref").warning(
                        "Direct team_match %s failed for %s %s: %s",
                        stat,
                        league,
                        season_str,
                        e,
                    )
                    try:
                        ok, df = _try_team_aggregate(fb, stat, league, season_str)
                        if not ok:
                            raise RuntimeError("aggregate_fallback_empty")
                        log.info("team_match %s (aggregate fallback in direct mode): %d rows", stat, len(df))
                    except Exception as e2:
                        logging.getLogger("fbref").warning(
                            "Aggregate fallback failed for team_match %s: %s — writing schema-only CSV",
                            stat,
                            e2,
                        )
                        df = _schema_only_from_previous(fb_prev, "team_match", stat)
                        df = _normalize(df, league, season_str)
                        wrote_schema_only = True

                safe_write(df, out_path_base)
                _snooze(delay)
                wrote = True
                if wrote_schema_only:
                    team_stats_summary["schema_only"].append(stat)
                else:
                    team_stats_summary["scraped_ok"].append(stat)

            if not wrote:
                logging.getLogger("fbref").warning(
                    "team_match %s not available via aggregate and direct disabled (team_mode=%s) — writing schema-only CSV",
                    stat,
                    team_mode,
                )
                df = _schema_only_from_previous(fb_prev, "team_match", stat)
                df = _normalize(df, league, season_str)
                safe_write(df, out_path_base)
                _snooze(delay)
                team_stats_summary["schema_only"].append(stat)

    log.info(
        "Match-level scrape DONE: league=%s season=%s [levels=%s]. "
        "Approx network calls so far: %d",
        league,
        season_str,
        levels,
        _GLOBAL["net_calls"],
    )

    return {
        "last_match_date": last_match_date_str,
        "latest_fixture": latest_fixture_meta,
        "cutoff_date": cutoff_date_str,
        "stats_summary": {
            "player_match": player_stats_summary,
            "team_match": team_stats_summary,
        },
    }


# ───────────────────────── CLI ─────────────────────────


def main() -> None:
    p = argparse.ArgumentParser("Scrape FBref match-level tables")

    p.add_argument(
        "--league",
        nargs="+",
        default=["ENG-Premier League"],
        help=(
            "One or more league identifiers understood by soccerdata, "
            'e.g. "ENG-Premier League". If multiple are provided, all '
            "will be scraped in this run (unless --all-known-leagues is set)."
        ),
    )
    p.add_argument(
        "--all-known-leagues",
        action="store_true",
        help=(
            "Ignore --league and scrape all leagues listed in ALL_KNOWN_LEAGUES "
            "in this script."
        ),
    )
    p.add_argument("--out-dir", default="data/raw/fbref")
    p.add_argument(
        "--seasons",
        nargs="*",
        help="Specific seasons to scrape (e.g. 2025-2026). If omitted, seasons_from_league(...) is used per league.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass soccerdata HTTP cache for this run (useful for manual refresh).",
    )
    p.add_argument(
        "--force-cache",
        action="store_true",
        help="Read only from cache; never hit network (safe for dev runs, if soccerdata supports it).",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Weekly-refresh mode: bypass soccerdata HTTP cache for the run. "
            "Equivalent to setting --no-cache, but more self-documenting. "
            "Do NOT combine with --force-cache."
        ),
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
        "--meta-path",
        type=str,
        default="data/meta/scraper_runs.json",
        help="Path to JSON file where last-run metadata is stored.",
    )
    p.add_argument(
        "--proxy",
        type=str,
        default=None,
        help='Optional proxy passed to soccerdata (e.g., "tor" or "http://user:pass@host:port").',
    )
    p.add_argument(
        "--player-stats",
        nargs="*",
        default=None,
        help=(
            "Optional subset of STAT_MAP['player_match'] stat_type names to scrape. "
            "If omitted, all player_match stats in STAT_MAP are scraped. "
            "Any unknown name causes a hard error."
        ),
    )
    p.add_argument(
        "--team-stats",
        nargs="*",
        default=None,
        help=(
            "Optional subset of STAT_MAP['team_match'] stat_type names to scrape. "
            "If omitted, all team_match stats in STAT_MAP are scraped. "
            "Any unknown name causes a hard error."
        ),
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "If set, reuse existing schedule CSV and skip any player/team stat "
            "whose existing CSV is complete up to cutoff_date (now - 1 day). "
            "If a stat CSV exists but is incomplete, it is re-scraped."
        ),
    )
    p.add_argument(
        "--skip-schedule",
        action="store_true",
        help=(
            "If set, and a schedule CSV already exists, reuse it instead of scraping schedule. "
            "By default schedule is scraped every run."
        ),
    )

    p.add_argument(
        "--rerun-failed",
        action="store_true",
        help=(
            "If set, read last recorded meta for this job key and re-run only stats "
            "in the selected buckets (default: schema_only + incomplete_existing). "
            "This overrides --player-stats/--team-stats."
        ),
    )
    p.add_argument(
        "--failed-sources",
        nargs="*",
        default=["schema_only", "incomplete_existing"],
        choices=["schema_only", "incomplete_existing", "skipped_up_to_date"],
        help="Which buckets count as 'failed' when --rerun-failed is enabled.",
    )
    p.add_argument(
        "--run-mode",
        choices=["manual", "automation"],
        default="manual",
        help="Tag this run in meta as 'manual' (default) or 'automation'.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.refresh and args.force_cache:
        raise SystemExit("Cannot use --refresh and --force-cache together. Pick one.")

    warnings.filterwarnings(
        "ignore", category=FutureWarning, module=r".*soccerdata\.fbref.*"
    )

    init_logger(args.verbose)
    log = logging.getLogger("fbref")

    effective_no_cache = bool(args.no_cache or args.refresh)

    if args.all_known_leagues:
        leagues = ALL_KNOWN_LEAGUES
    else:
        leagues = args.league

    log.info(
        "FBref scrape configuration: leagues=%s seasons=%s levels=%s "
        "team_mode=%s no_cache=%s force_cache=%s refresh=%s out_dir=%s skip_existing=%s "
        "rerun_failed=%s failed_sources=%s run_mode=%s",
        leagues,
        args.seasons if args.seasons else "auto (seasons_from_league)",
        args.levels,
        args.team_mode,
        effective_no_cache,
        args.force_cache,
        args.refresh,
        args.out_dir,
        args.skip_existing,
        args.rerun_failed,
        args.failed_sources,
        args.run_mode,
    )

    out_base = Path(args.out_dir)
    meta_path = Path(args.meta_path)

    for league in leagues:
        seasons = args.seasons or seasons_from_league(league)
        log.info("Resolved seasons for %s: %s", league, seasons)

        for s in seasons:
            # If rerun-failed is enabled, override player_stats/team_stats for THIS job key.
            if args.rerun_failed:
                job_key = ScrapeJobId(scraper="match", league=league, season=str(s), levels=args.levels)
                failed = _load_failed_stats_from_meta(meta_path, job_key, list(args.failed_sources))

                if args.levels in ("player", "both"):
                    args.player_stats = failed["player_match"] or []
                else:
                    args.player_stats = None

                if args.levels in ("team", "both"):
                    args.team_stats = failed["team_match"] or []
                else:
                    args.team_stats = None

                log.info(
                    "RERUN-FAILED resolved for %s: player_stats=%s team_stats=%s (sources=%s)",
                    job_key.key(),
                    args.player_stats,
                    args.team_stats,
                    args.failed_sources,
                )

                # If nothing to rerun for requested levels, skip cleanly.
                if args.levels == "player" and not args.player_stats:
                    log.info("Nothing to rerun (player) for %s; skipping.", job_key.key())
                    continue
                if args.levels == "team" and not args.team_stats:
                    log.info("Nothing to rerun (team) for %s; skipping.", job_key.key())
                    continue
                if args.levels == "both" and not (args.player_stats or args.team_stats):
                    log.info("Nothing to rerun (both) for %s; skipping.", job_key.key())
                    continue

            result = scrape_one(
                league,
                str(s),
                out_base,
                args.delay,
                no_cache=effective_no_cache,
                force_cache=args.force_cache,
                max_retries=args.max_retries,
                backoff_base=args.backoff_base,
                periodic_every=args.periodic_every,
                periodic_secs=args.periodic_secs,
                team_mode=args.team_mode,
                proxy=args.proxy,
                levels=args.levels,
                player_stats=args.player_stats,
                team_stats=args.team_stats,
                skip_existing=args.skip_existing,
                skip_schedule=args.skip_schedule,
            )

            job = ScrapeJobId(
                scraper="match",
                league=league,
                season=str(s),
                levels=args.levels,
            )

            run_info = {
                "scrape_ts": datetime.now(timezone.utc).isoformat(),
                "mode": args.run_mode,
                "cutoff_date": result.get("cutoff_date"),
                "last_match_date": result.get("last_match_date"),
                "latest_fixture": result.get("latest_fixture"),
                "stats_summary": result.get("stats_summary"),
            }

            record_last_run(meta_path, job, run_info=run_info)
            log.info(
                "Recorded last-run meta for %s (last_match_date=%s, cutoff_date=%s)",
                job.key(),
                result.get("last_match_date"),
                result.get("cutoff_date"),
            )

    log.info(
        "FBref scrape completed for leagues=%s. Total approximate network calls: %d",
        leagues,
        _GLOBAL["net_calls"],
    )


if __name__ == "__main__":
    main()
