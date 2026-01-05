#!/usr/bin/env python3
# scripts/fbref_pipeline/scrape/season_stats_scraper.py
"""
FBref season-level scraper (network-friendly, cache-first)

Upgrades included (match-scraper parity)
- --refresh: weekly-refresh mode (bypass soccerdata HTTP cache for the run)
- Multi-league support: --league takes 1+ leagues, --all-known-leagues iterates ALL_KNOWN_LEAGUES
- Clear log.info lines for each stat being scraped and per-league/season summaries
- Subset scraping: --player-stats / --team-stats
- Skip existing outputs: --skip-existing (if existing CSV exists and is non-empty, skip)
- Meta tracking: writes last-run metadata to --meta-path
- Re-run only failures: --rerun-failed with --failed-sources {schema_only,incomplete_existing}

Notes
- "season-level" tables are not fixture-bound; so "up-to-date" checks are not possible without
  deeper logic. Therefore skip logic is intentionally simple: if file exists and has rows, skip.
- Team-season aggregate mode uses player-season data (no direct team endpoint hits if possible).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable, Any, Dict, Tuple, Sequence, List, Set

import pandas as pd
import soccerdata as sd
from requests.exceptions import HTTPError

from scripts.fbref_pipeline.utils.fbref_utils import (
    STAT_MAP,
    safe_write,
    seasons_from_league,
    init_logger,
    polite_sleep,
)

# Keep meta import consistent with match scraper
from scripts.fbref_pipeline.automation.auto_scrape import ScrapeJobId, record_last_run

# ───────────────────────── config & defaults ─────────────────────────

PLAYER_SEASON_DEFAULTS: Sequence[str] = (
    "standard", "shooting", "passing", "passing_types",
    "defense", "possession", "misc", "gca", "xg", "keepers", "keepersadv",
)
TEAM_SEASON_DEFAULTS: Sequence[str] = (
    "standard", "shooting", "passing", "passing_types",
    "defense", "possession", "misc", "gca", "xg", "keepers", "keepersadv",
)

ALL_KNOWN_LEAGUES: List[str] = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]

_GLOBAL = {"net_calls": 0}

# "rerun failed" sources that we understand from meta
FAILED_SOURCES_ALLOWED = {"schema_only", "incomplete_existing"}

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
        logging.getLogger("fbref").info(
            "Cooling down: resting for %.1fs after %d calls",
            secs, _GLOBAL["net_calls"]
        )
        time.sleep(secs)

def _parse_retry_after(exc: Exception) -> Optional[float]:
    resp = getattr(exc, "response", None)
    if not resp or not hasattr(resp, "headers"):
        return None
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        return float(ra)
    except Exception:
        return None

def _call_with_optional_kw(
    fn: Callable[..., Any],
    kwargs: Dict[str, Any],
    optional_kw: Optional[Dict[str, Any]] = None,
):
    """
    Call fn(**kwargs, **optional_kw) if supported, else retry without optional_kw.
    """
    if optional_kw:
        try:
            return fn(**{**kwargs, **optional_kw})
        except TypeError as e:
            msg = str(e)
            if any((f"'{k}'" in msg) or (k in msg) for k in optional_kw.keys()):
                logging.getLogger("fbref").warning(
                    "Optional kwargs %s not accepted by %s; retrying without them. "
                    "If you expected cache-only behaviour, this call may hit the network.",
                    list(optional_kw.keys()),
                    getattr(fn, "__name__", repr(fn)),
                )
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
    """
    Build an empty DF with previous-season columns; fallback to minimal schema.
    Uses cache-only schema discovery when soccerdata supports force_cache.
    """
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

def _resolve_stats() -> tuple[list[str], list[str]]:
    player_stats = list(STAT_MAP.get("player_season") or PLAYER_SEASON_DEFAULTS)
    team_stats = list(STAT_MAP.get("team_season") or TEAM_SEASON_DEFAULTS)
    return player_stats, team_stats

def _csv_candidates(base: Path) -> List[Path]:
    # safe_write typically writes base + ".csv"
    if base.suffix:
        return [base]
    return [base.with_suffix(".csv"), base]

def _csv_exists_nonempty(base: Path) -> bool:
    """
    True if a CSV exists and has at least 1 data row.
    """
    for p in _csv_candidates(base):
        if p.is_file():
            try:
                df = pd.read_csv(p)
                return isinstance(df, pd.DataFrame) and not df.empty
            except Exception:
                return False
    return False

def _save_with_log(
    df: pd.DataFrame,
    base: Path,
    label: str,
    *,
    layout: str,
    log: logging.Logger,
) -> Path:
    """
    Write CSV and log absolute final path + rows.
    layout='flat'    -> base like <...>/player_season_standard  -> writes that .csv
    layout='folders' -> base like <...>/player_season/standard  -> writes that .csv
    Returns the absolute CSV path that should exist after write.
    """
    if layout == "folders":
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
    return path_abs

# ────────── aggregate team-season from player-season (no extra direct hits) ──────────

_EXCLUDE_RATE_TOKENS = (
    "per90", "per_90", "per-90", " per 90", "90s", "rate", "pct", "%", "share",
    "/90", "per match", "per_match", "per-game", "per game", "p90",
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
        log.info("Scraping team_season via aggregate-from-player: %s %s (%s)", league, season_str, stat)
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
        log.info("✓ team_season %s built via aggregate from player season", stat)
        return True, grouped
    except Exception as e:
        log.debug("Aggregate team season for %s failed: %s", stat, e)
        return False, pd.DataFrame()

# ───────────────────────── meta helpers (rerun-failed) ─────────────────────────

def _load_meta_json(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _stats_from_last_run(meta: Dict[str, Any], job: ScrapeJobId, level_key: str, sources: Set[str]) -> Set[str]:
    """
    Extract stats that should be re-run based on last_run.stats_summary for a job.
    level_key: "player_season" or "team_season"
    sources: e.g., {"schema_only", "incomplete_existing"}
    """
    block = meta.get(job.key(), {})
    last = block.get("last_run", {}) if isinstance(block, dict) else {}
    summary = last.get("stats_summary", {}) if isinstance(last, dict) else {}
    level = summary.get(level_key, {}) if isinstance(summary, dict) else {}
    out: Set[str] = set()
    for src in sources:
        vals = level.get(src, [])
        if isinstance(vals, list):
            out |= set(str(x) for x in vals)
    return out

# ───────────────────────── core ─────────────────────────

def scrape_one(
    league: str,
    season: str,
    out_base: Path,
    delay: float,
    *,
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
    # new:
    player_stats: Optional[List[str]] = None,
    team_stats: Optional[List[str]] = None,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    log = logging.getLogger("fbref")
    season_str = str(season)
    out_dir = out_base / league / season_str
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Season-level scrape START: league=%s season=%s "
        "[levels=%s team_mode=%s no_cache=%s force_cache=%s skip_existing=%s layout=%s]",
        league, season_str, levels, team_mode, no_cache, force_cache, skip_existing, layout,
    )

    fb_kwargs = dict(leagues=league, seasons=season_str, no_cache=no_cache)
    if proxy:
        fb_kwargs["proxy"] = proxy
    fb = sd.FBref(**fb_kwargs)

    try:
        fb_prev = sd.FBref(leagues=league, seasons=_prev_season_str(season_str), no_cache=no_cache, proxy=proxy)
    except Exception:
        fb_prev = None

    all_player_stats, all_team_stats = _resolve_stats()

    # Validate / resolve subsets (and preserve STAT_MAP order)
    all_player_set = set(all_player_stats)
    all_team_set = set(all_team_stats)

    if player_stats is not None:
        req = list(dict.fromkeys(player_stats))
        unknown = sorted(set(req) - all_player_set)
        if unknown:
            raise SystemExit(f"Unknown player_season stat_type(s): {unknown}. Valid: {sorted(all_player_set)}")
        player_run = [s for s in all_player_stats if s in set(req)]
    else:
        player_run = list(all_player_stats)

    if team_stats is not None:
        req = list(dict.fromkeys(team_stats))
        unknown = sorted(set(req) - all_team_set)
        if unknown:
            raise SystemExit(f"Unknown team_season stat_type(s): {unknown}. Valid: {sorted(all_team_set)}")
        team_run = [s for s in all_team_stats if s in set(req)]
    else:
        team_run = list(all_team_stats)

    if echo_config:
        log.info("Output dir: %s", out_dir.resolve())
        log.info("Levels=%s | Team mode=%s | Layout=%s | Skip existing=%s", levels, team_mode, layout, skip_existing)
        log.info("Player season stats to run: %s", ", ".join(player_run))
        log.info("Team season stats to run:   %s", ", ".join(team_run))

    # Meta summary
    player_summary: Dict[str, List[str]] = {
        "skipped_existing": [],
        "scraped_ok": [],
        "schema_only": [],
        "incomplete_existing": [],  # kept for parity; for season-level we don't detect incompleteness
    }
    team_summary: Dict[str, List[str]] = {
        "skipped_existing": [],
        "scraped_ok": [],
        "schema_only": [],
        "incomplete_existing": [],
    }

    # 1) Player season
    if levels in ("player", "both"):
        for stat in player_run:
            label = f"player_season_{stat}"
            # compute output base path for skip check
            if layout == "folders":
                out_base_path = out_dir / "player" / "season"  # not used directly
                path_base = out_dir / "player_season" / stat
            else:
                path_base = out_dir / label

            if skip_existing and _csv_exists_nonempty(path_base):
                log.info("Skipping player_season stat=%s for %s %s (existing non-empty CSV)", stat, league, season_str)
                player_summary["skipped_existing"].append(stat)
                continue

            log.info("Scraping player_season stat=%s for %s %s", stat, league, season_str)
            wrote_schema_only = False
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
                log.info("player_season %s: %d rows, %d cols", stat, len(df), df.shape[1])
            except Exception as e:
                log.warning(
                    "Missing/unavailable player_season %s for %s %s: %s — writing schema-only CSV",
                    stat, league, season_str, e
                )
                df = _schema_only_from_previous(fb_prev, "player_season", stat)
                df = _normalize(df, league, season_str)
                wrote_schema_only = True

            _save_with_log(df, out_dir, label, layout=layout, log=log)
            if wrote_schema_only:
                player_summary["schema_only"].append(stat)
            else:
                player_summary["scraped_ok"].append(stat)
            _snooze(delay)

    # 2) Team season
    if levels in ("team", "both"):
        for stat in team_run:
            label = f"team_season_{stat}"
            if layout == "folders":
                path_base = out_dir / "team_season" / stat
            else:
                path_base = out_dir / label

            if skip_existing and _csv_exists_nonempty(path_base):
                log.info("Skipping team_season stat=%s for %s %s (existing non-empty CSV)", stat, league, season_str)
                team_summary["skipped_existing"].append(stat)
                continue

            wrote = False
            wrote_schema_only = False

            # Option A: aggregate-from-player
            if team_mode in ("aggregate", "auto"):
                ok, df = _try_team_aggregate(
                    fb,
                    stat,
                    league,
                    season_str,
                    force_cache=force_cache,
                    backoff_base=backoff_base,
                    max_retries=max_retries,
                    periodic_every=periodic_every,
                    periodic_secs=periodic_secs,
                )
                if ok:
                    log.info("team_season %s (aggregate): %d rows, %d cols", stat, len(df), df.shape[1])
                    _save_with_log(df, out_dir, label, layout=layout, log=log)
                    team_summary["scraped_ok"].append(stat)
                    _snooze(delay)
                    wrote = True

            # Option B: direct endpoint
            if not wrote and team_mode in ("direct", "auto"):
                log.info("Scraping team_season stat=%s via direct endpoint for %s %s", stat, league, season_str)
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
                    log.info("team_season %s (direct): %d rows, %d cols", stat, len(df), df.shape[1])
                except Exception as e:
                    log.warning("Direct team_season %s failed for %s %s: %s", stat, league, season_str, e)
                    # last-chance: aggregate fallback
                    try:
                        ok, df = _try_team_aggregate(
                            fb,
                            stat,
                            league,
                            season_str,
                            force_cache=force_cache,
                            backoff_base=backoff_base,
                            max_retries=max_retries,
                            periodic_every=periodic_every,
                            periodic_secs=periodic_secs,
                        )
                        if not ok:
                            raise RuntimeError("aggregate_fallback_empty")
                        log.info("team_season %s (aggregate fallback): %d rows, %d cols", stat, len(df), df.shape[1])
                    except Exception as e2:
                        log.warning("Aggregate fallback failed for team_season %s: %s — writing schema-only CSV", stat, e2)
                        df = _schema_only_from_previous(fb_prev, "team_season", stat)
                        df = _normalize(df, league, season_str)
                        wrote_schema_only = True

                _save_with_log(df, out_dir, label, layout=layout, log=log)
                if wrote_schema_only:
                    team_summary["schema_only"].append(stat)
                else:
                    team_summary["scraped_ok"].append(stat)
                _snooze(delay)
                wrote = True

            # Option C: schema-only if aggregate-only and it failed
            if not wrote:
                log.warning(
                    "team_season %s not available via aggregate and direct disabled (team_mode=%s) — writing schema-only CSV",
                    stat, team_mode
                )
                df = _schema_only_from_previous(fb_prev, "team_season", stat)
                df = _normalize(df, league, season_str)
                _save_with_log(df, out_dir, label, layout=layout, log=log)
                team_summary["schema_only"].append(stat)
                _snooze(delay)

    log.info(
        "Season-level scrape DONE: league=%s season=%s [levels=%s]. Approx network calls so far: %d",
        league, season_str, levels, _GLOBAL["net_calls"]
    )

    return {
        "cutoff_date": None,  # season tables don't use the match-style cutoff_date logic
        "stats_summary": {
            "player_season": player_summary,
            "team_season": team_summary,
        },
    }

# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    p = argparse.ArgumentParser("Scrape FBref season-level tables")

    p.add_argument(
        "--league",
        nargs="+",
        default=["ENG-Premier League"],
        help="One or more league identifiers understood by soccerdata.",
    )
    p.add_argument(
        "--all-known-leagues",
        action="store_true",
        help="Ignore --league and scrape all leagues listed in ALL_KNOWN_LEAGUES.",
    )
    p.add_argument("--out-dir", default="data/raw/fbref")
    p.add_argument(
        "--seasons",
        nargs="*",
        help="Specific seasons to scrape (e.g. 2025-2026). If omitted, seasons_from_league(...) is used per league.",
    )

    p.add_argument("--no-cache", action="store_true", help="Bypass soccerdata HTTP cache for this run.")
    p.add_argument("--force-cache", action="store_true", help="Read only from cache; never hit network (if supported).")
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Weekly-refresh mode: bypass soccerdata HTTP cache for the run (same as --no-cache). Do not combine with --force-cache.",
    )

    p.add_argument("--delay", type=float, default=3.2, help="Base inter-request delay (seconds).")
    p.add_argument("--max-retries", type=int, default=6, help="Max 429 backoff retries per request.")
    p.add_argument("--backoff-base", type=float, default=3.2, help="Initial backoff seconds on 429.")
    p.add_argument("--periodic-every", type=int, default=8, help="Cool-down every N network calls.")
    p.add_argument("--periodic-secs", type=float, default=15.0, help="Cool-down length in seconds.")

    p.add_argument(
        "--team-mode",
        choices=["aggregate", "auto", "direct"],
        default="aggregate",
        help="Team tables: aggregate from player, aggregate→direct, or direct FBref endpoints.",
    )
    p.add_argument("--levels", choices=["player", "team", "both"], default="both", help="Which levels to scrape.")
    p.add_argument("--layout", choices=["flat", "folders"], default="flat", help="Output naming layout.")
    p.add_argument("--proxy", type=str, default=None, help='Optional proxy for soccerdata (e.g., "tor" or URL).')

    p.add_argument(
        "--player-stats",
        nargs="*",
        default=None,
        help="Optional subset of player_season stat_type names to scrape. If omitted, all are scraped.",
    )
    p.add_argument(
        "--team-stats",
        nargs="*",
        default=None,
        help="Optional subset of team_season stat_type names to scrape. If omitted, all are scraped.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="If set, skip any stat whose output CSV already exists and is non-empty.",
    )

    p.add_argument(
        "--meta-path",
        type=str,
        default="data/meta/scraper_runs.json",
        help="Path to JSON file where last-run metadata is stored.",
    )
    p.add_argument(
        "--run-mode",
        choices=["manual", "automation"],
        default="manual",
        help="Tag this run in meta as 'manual' or 'automation'.",
    )

    # Rerun failed support
    p.add_argument(
        "--rerun-failed",
        action="store_true",
        help="If set, override requested stats and re-run only stats that failed last run for this job.",
    )
    p.add_argument(
        "--failed-sources",
        nargs="*",
        default=["schema_only", "incomplete_existing"],
        help="When used with --rerun-failed, which failure buckets to rerun: schema_only and/or incomplete_existing.",
    )

    p.add_argument("--echo-config", action="store_true", help="Print resolved stats and output dir before scraping.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.refresh and args.force_cache:
        raise SystemExit("Cannot use --refresh and --force-cache together. Pick one.")

    warnings.filterwarnings("ignore", category=FutureWarning, module=r".*soccerdata\.fbref.*")
    init_logger(args.verbose)
    log = logging.getLogger("fbref")

    effective_no_cache = bool(args.no_cache or args.refresh)

    # leagues
    leagues = ALL_KNOWN_LEAGUES if args.all_known_leagues else args.league

    meta_path = Path(args.meta_path)
    meta_blob = _load_meta_json(meta_path) if args.rerun_failed else {}

    failed_sources = set(args.failed_sources or [])
    unknown_sources = sorted(failed_sources - FAILED_SOURCES_ALLOWED)
    if unknown_sources:
        raise SystemExit(f"Unknown --failed-sources value(s): {unknown_sources}. Allowed: {sorted(FAILED_SOURCES_ALLOWED)}")

    log.info(
        "FBref season-level scrape configuration: leagues=%s seasons=%s levels=%s team_mode=%s "
        "no_cache=%s force_cache=%s refresh=%s layout=%s out_dir=%s "
        "skip_existing=%s rerun_failed=%s failed_sources=%s run_mode=%s",
        leagues,
        args.seasons if args.seasons else "auto (seasons_from_league)",
        args.levels,
        args.team_mode,
        effective_no_cache,
        args.force_cache,
        args.refresh,
        args.layout,
        args.out_dir,
        args.skip_existing,
        args.rerun_failed,
        sorted(failed_sources),
        args.run_mode,
    )

    out_base = Path(args.out_dir)

    for league in leagues:
        seasons = args.seasons or seasons_from_league(league)
        log.info("Resolved seasons for %s: %s", league, seasons)

        for s in seasons:
            season_str = str(s)

            # If rerun-failed, compute exactly which stats to run from meta
            player_stats = args.player_stats
            team_stats = args.team_stats

            job = ScrapeJobId(scraper="season", league=league, season=season_str, levels=args.levels)

            if args.rerun_failed:
                # Pull failures from meta per level, but only for enabled levels
                rerun_player: Set[str] = set()
                rerun_team: Set[str] = set()

                if args.levels in ("player", "both"):
                    rerun_player = _stats_from_last_run(meta_blob, job, "player_season", failed_sources)
                if args.levels in ("team", "both"):
                    rerun_team = _stats_from_last_run(meta_blob, job, "team_season", failed_sources)

                # Convert empty sets to empty lists: means "run nothing" for that side
                # (and we log it clearly)
                player_stats = sorted(rerun_player) if rerun_player else []
                team_stats = sorted(rerun_team) if rerun_team else []

                log.info(
                    "RERUN FAILED for %s: player_stats=%s | team_stats=%s",
                    job.key(),
                    player_stats if player_stats else "(none)",
                    team_stats if team_stats else "(none)",
                )

                # If both empty, do nothing but still record a meta run event so you know it executed
                if args.levels == "both" and not player_stats and not team_stats:
                    log.warning("No failed stats found in meta for %s — nothing to re-run.", job.key())

            result = scrape_one(
                league,
                season_str,
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
                layout=args.layout,
                echo_config=args.echo_config,
                player_stats=player_stats,
                team_stats=team_stats,
                skip_existing=args.skip_existing,
            )

            run_info = {
                "scrape_ts": datetime.now(timezone.utc).isoformat(),
                "mode": args.run_mode,
                "cutoff_date": None,
                "last_match_date": None,
                "latest_fixture": None,
                "stats_summary": result.get("stats_summary"),
            }

            record_last_run(meta_path, job, run_info=run_info)
            log.info("Recorded last-run meta for %s", job.key())

    log.info(
        "FBref season-level scrape completed for leagues=%s. Total approximate network calls: %d",
        leagues, _GLOBAL["net_calls"]
    )

if __name__ == "__main__":
    main()
