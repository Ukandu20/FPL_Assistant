#!/usr/bin/env python3
# scripts/clubelo_pipeline/scrape/clubelo_scraper.py
"""
ClubElo scraper (CSV API, cache-first)

Key features:
- Exponential backoff for HTTP 429 / transient errors (honors Retry-After).
- Jittered sleeps between requests to avoid synchronized bursts.
- Cache-first defaults (no_cache=False) + --force-cache for offline dev runs.
- Date snapshots and per-team history in one tool.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Sequence, Union

import pandas as pd
import requests
from requests.exceptions import HTTPError

try:
    from unidecode import unidecode
except Exception:
    def unidecode(value: str) -> str:
        return value

from scripts.fbref_pipeline.automation.auto_scrape import ScrapeJobId, record_last_run

CLUB_ELO_API = "http://api.clubelo.com"
DEFAULT_CACHE_DIR = Path("data/_clubelo_cache/ClubElo")
DEFAULT_OUT_DIR = Path("data/raw/clubelo")

_GLOBAL = {"net_calls": 0}


def init_logger(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def safe_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=True)
    logging.getLogger("clubelo").debug("saved %s", path.with_suffix("").name)


def standardize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    cols: List[str] = []
    for col in df.columns:
        name = str(col).strip().lower()
        name = re.sub(r"[^a-z0-9_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        cols.append(name)
    out = df.copy()
    out.columns = cols
    return out


def _snooze(base: float) -> None:
    jitter = base * random.uniform(0.2, 0.4)
    if base + jitter > 0:
        time.sleep(base + jitter)


def _periodic_rest(every: int, secs: float, *, skip: bool) -> None:
    if skip:
        return
    if every > 0 and _GLOBAL["net_calls"] > 0 and (_GLOBAL["net_calls"] % every == 0):
        logging.getLogger("clubelo").info(
            "Cooling down: resting for %.1fs after %d calls",
            secs,
            _GLOBAL["net_calls"],
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


def _with_backoff(
    fn,
    *,
    max_retries: int = 6,
    base_delay: float = 3.2,
    kwargs: Optional[Dict[str, Any]] = None,
    count_as_network: bool = True,
    periodic_every: int = 8,
    periodic_secs: float = 15.0,
) -> Any:
    if kwargs is None:
        kwargs = {}

    delay = base_delay
    last_exc: Optional[Exception] = None
    log = logging.getLogger("clubelo")

    for attempt in range(max_retries):
        try:
            result = fn(**kwargs)
            if count_as_network:
                _GLOBAL["net_calls"] += 1
            _periodic_rest(periodic_every, periodic_secs, skip=not count_as_network)
            return result
        except Exception as e:
            last_exc = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            msg = str(e).lower()
            is_429 = (
                (isinstance(e, HTTPError) and status == 429)
                or ("429" in msg)
                or ("too many requests" in msg)
            )
            if is_429:
                ra = _parse_retry_after(e)
                sleep_for = (float(ra) if ra else min(delay, 60.0)) + random.random()
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


def _parse_csv(data: IO[bytes]) -> pd.DataFrame:
    df = pd.read_csv(data)
    for col in ("From", "To"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
    return df


def _team_slug(team: str) -> str:
    return re.sub(r"[^a-zA-Z]", "", unidecode(team))


def _team_candidates(team: str) -> List[str]:
    raw = str(team).strip()
    candidates = {
        raw,
        unidecode(raw),
        raw.replace(" ", ""),
        raw.replace("'", ""),
    }
    out = [c for c in candidates if c]
    return list(dict.fromkeys(out))


def _safe_filename(text: str) -> str:
    raw = unidecode(str(text)).strip()
    raw = re.sub(r"\s+", "_", raw)
    raw = re.sub(r"[^A-Za-z0-9._-]", "", raw)
    return raw or "team"


def _parse_headers(user_agent: Optional[str], headers: Optional[List[str]]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    if user_agent:
        parsed["User-Agent"] = user_agent.strip()
    if headers:
        for raw in headers:
            if ":" not in raw:
                raise SystemExit("Invalid --header; expected 'Key: Value'")
            key, value = raw.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise SystemExit("Invalid --header; empty key")
            parsed[key] = value
    return parsed


def _cache_path_date(data_dir: Path, datestring: str) -> Path:
    return data_dir / "dates" / f"{datestring}.csv"


def _cache_path_team(data_dir: Path, slug: str) -> Path:
    return data_dir / "teams" / f"{slug}.csv"


def _is_stale(path: Path, max_age: Optional[timedelta]) -> bool:
    if max_age is None:
        return False
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return datetime.now(tz=timezone.utc) - mtime > max_age
    except Exception:
        return True


def _request_csv(
    session: requests.Session,
    url: str,
    filepath: Optional[Path],
    *,
    no_cache: bool,
    no_store: bool,
    force_cache: bool,
    max_age: Optional[timedelta],
    timeout: float = 30.0,
) -> IO[bytes]:
    use_cache = (
        filepath is not None
        and filepath.exists()
        and not no_cache
        and not _is_stale(filepath, max_age)
    )

    if use_cache and filepath is not None:
        payload = filepath.read_bytes()
        if payload not in (b"", b"{}", b"[]"):
            return io.BytesIO(payload)

    if force_cache:
        raise FileNotFoundError(f"Cache miss for {url} at {filepath}")

    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.content

    if not no_store and filepath is not None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_bytes(payload)

    return io.BytesIO(payload)


def _apply_team_replacements(df: pd.DataFrame, replacements: Dict[str, str]) -> pd.DataFrame:
    if not replacements:
        return df
    if "team" in df.columns:
        df = df.copy()
        df["team"] = df["team"].replace(replacements)
    elif df.index.name == "team":
        df = df.copy()
        df.index = df.index.to_series().replace(replacements).values
    return df


def _normalize_date_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.pipe(standardize_colnames)
        .rename(columns={"club": "team"})
        .replace("None", pd.NA)
    )
    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    if "country" in df.columns and "level" in df.columns:
        df["league"] = df["country"].astype(str) + "_" + df["level"].astype(str)
    if "team" in df.columns:
        df = df.set_index("team")
    return df


def _normalize_team_history(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.pipe(standardize_colnames)
        .rename(columns={"club": "team"})
        .replace("None", pd.NA)
    )
    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    if "from" in df.columns:
        df["from"] = pd.to_datetime(df["from"], errors="coerce")
        df = df.set_index("from").sort_index()
    return df


class ClubEloClient:
    def __init__(
        self,
        *,
        proxy: Optional[str] = None,
        no_cache: bool = False,
        no_store: bool = False,
        data_dir: Path = DEFAULT_CACHE_DIR,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> None:
        self.no_cache = no_cache
        self.no_store = no_store
        self.data_dir = data_dir
        self.timeout = float(timeout)
        self._session = requests.Session()
        if headers:
            self._session.headers.update(headers)
        if proxy:
            if proxy == "tor":
                proxy = "socks5h://127.0.0.1:9050"
            self._session.proxies.update({"http": proxy, "https": proxy})

    def read_by_date(self, date: Optional[Union[str, datetime]] = None, *, force_cache: bool = False) -> pd.DataFrame:
        if not date:
            date = datetime.now(tz=timezone.utc)
        elif isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").astimezone(timezone.utc)

        if not isinstance(date, datetime):
            raise TypeError("'date' must be a datetime object or string like 'YYYY-MM-DD'")

        datestring = date.strftime("%Y-%m-%d")
        filepath = _cache_path_date(self.data_dir, datestring)
        url = f"{CLUB_ELO_API}/{datestring}"

        data = _request_csv(
            self._session,
            url,
            filepath,
            no_cache=self.no_cache,
            no_store=self.no_store,
            force_cache=force_cache,
            max_age=None,
            timeout=self.timeout,
        )

        return _normalize_date_snapshot(_parse_csv(data))

    def read_team_history(
        self,
        team: str,
        *,
        max_age_days: Optional[int] = 1,
        force_cache: bool = False,
    ) -> pd.DataFrame:
        if max_age_days is None:
            max_age = None
        elif max_age_days < 0:
            raise ValueError("max_age_days must be >= 0 or None")
        else:
            max_age = timedelta(days=max_age_days)

        last_exc: Optional[Exception] = None
        for candidate in _team_candidates(team):
            slug = _team_slug(candidate)
            if not slug:
                continue

            filepath = _cache_path_team(self.data_dir, slug)
            url = f"{CLUB_ELO_API}/{slug}"

            try:
                data = _request_csv(
                    self._session,
                    url,
                    filepath,
                    no_cache=self.no_cache,
                    no_store=self.no_store,
                    force_cache=force_cache,
                    max_age=max_age,
                    timeout=self.timeout,
                )
            except FileNotFoundError as e:
                last_exc = e
                continue

            df = _normalize_team_history(_parse_csv(data))
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df

        if last_exc is not None:
            raise last_exc
        raise ValueError(f"No data found for team {team}")


def _csv_exists_nonempty(base: Path) -> bool:
    candidates = [base]
    if base.suffix == "":
        candidates.append(base.with_suffix(".csv"))
    for p in candidates:
        if p.is_file():
            try:
                df = pd.read_csv(p)
                return isinstance(df, pd.DataFrame) and not df.empty
            except Exception:
                return False
    return False


def _load_team_replacements(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"teamname replacements file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        raise SystemExit(f"Failed to parse teamname replacements: {e}")
    raise SystemExit("teamname replacements must be a JSON object of name->replacement")


def _parse_date_args(raw: Optional[Sequence[str]]) -> List[str]:
    if not raw:
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        return [today]
    out: List[str] = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            datetime.strptime(part, "%Y-%m-%d")
            out.append(part)
    return out


def _season_anchor_date(season: str) -> str:
    """
    Convert a season string to a mid-season snapshot date.
    Examples:
      2024-2025 -> 2025-01-01
      2024-25   -> 2025-01-01
      24-25     -> 2025-01-01 (assumes 2000s)
      2025      -> 2025-01-01
    """
    s = str(season).strip()

    m = re.fullmatch(r"(\d{4})[/-](\d{4})", s)
    if m:
        end_year = int(m.group(2))
        return f"{end_year}-01-01"

    m = re.fullmatch(r"(\d{4})[/-](\d{2})", s)
    if m:
        end_year = int(m.group(1)[:2] + m.group(2))
        return f"{end_year}-01-01"

    m = re.fullmatch(r"(\d{2})[/-](\d{2})", s)
    if m:
        end_year = 2000 + int(m.group(2))
        return f"{end_year}-01-01"

    m = re.fullmatch(r"(\d{4})", s)
    if m:
        return f"{int(m.group(1))}-01-01"

    raise SystemExit(f"Unrecognized season format: {season!r}")


def _teams_from_snapshot(df: pd.DataFrame, league: str) -> List[str]:
    if "team" not in df.columns:
        if df.index.name == "team":
            df = df.reset_index()
        else:
            raise ValueError("Snapshot missing 'team' column")
    if "league" not in df.columns:
        raise ValueError("Snapshot missing 'league' column")

    mask = df["league"].astype(str).str.lower() == str(league).lower()
    teams = df.loc[mask, "team"].dropna().astype(str).tolist()
    return list(dict.fromkeys(teams))


def _resolve_league_teams(
    client: ClubEloClient,
    leagues: Sequence[str],
    dates: Sequence[str],
    *,
    delay: float,
    force_cache: bool,
    max_retries: int,
    backoff_base: float,
    periodic_every: int,
    periodic_secs: float,
) -> List[str]:
    log = logging.getLogger("clubelo")
    teams: List[str] = []

    for datestring in dates:
        try:
            df = _with_backoff(
                client.read_by_date,
                max_retries=max_retries,
                base_delay=backoff_base,
                kwargs={"date": datestring, "force_cache": force_cache},
                count_as_network=not force_cache,
                periodic_every=periodic_every,
                periodic_secs=periodic_secs,
            )
        except Exception as e:
            log.warning(
                "Failed to load snapshot %s for league team discovery: %s",
                datestring,
                e,
            )
            _snooze(delay)
            continue

        for league in leagues:
            try:
                league_teams = _teams_from_snapshot(df, league)
            except Exception as e:
                log.warning(
                    "Failed to resolve teams for league %s at %s: %s",
                    league,
                    datestring,
                    e,
                )
                continue
            if not league_teams:
                log.warning("No teams found for league %s at %s", league, datestring)
                continue
            teams.extend(league_teams)

        _snooze(delay)

    return list(dict.fromkeys(teams))


def scrape_dates(
    client: ClubEloClient,
    dates: Sequence[str],
    out_dir: Path,
    *,
    delay: float,
    force_cache: bool,
    max_retries: int,
    backoff_base: float,
    periodic_every: int,
    periodic_secs: float,
    skip_existing: bool,
    replacements: Dict[str, str],
) -> Dict[str, List[str]]:
    log = logging.getLogger("clubelo")
    summary: Dict[str, List[str]] = {"skipped_existing": [], "scraped_ok": [], "failed": []}

    for datestring in dates:
        out_path_base = out_dir / "by_date" / datestring
        if skip_existing and _csv_exists_nonempty(out_path_base):
            log.info("Skipping date snapshot %s (existing CSV)", datestring)
            summary["skipped_existing"].append(datestring)
            continue

        try:
            df = _with_backoff(
                client.read_by_date,
                max_retries=max_retries,
                base_delay=backoff_base,
                kwargs={"date": datestring, "force_cache": force_cache},
                count_as_network=not force_cache,
                periodic_every=periodic_every,
                periodic_secs=periodic_secs,
            )
            df = _apply_team_replacements(df, replacements)
            safe_write(df, out_path_base)
            summary["scraped_ok"].append(datestring)
        except Exception as e:
            log.warning("Failed to scrape ClubElo date snapshot %s: %s", datestring, e)
            summary["failed"].append(datestring)
        _snooze(delay)

    return summary


def scrape_teams(
    client: ClubEloClient,
    teams: Sequence[str],
    out_dir: Path,
    *,
    delay: float,
    force_cache: bool,
    max_age_days: Optional[int],
    max_retries: int,
    backoff_base: float,
    periodic_every: int,
    periodic_secs: float,
    skip_existing: bool,
    replacements: Dict[str, str],
) -> Dict[str, List[str]]:
    log = logging.getLogger("clubelo")
    summary: Dict[str, List[str]] = {"skipped_existing": [], "scraped_ok": [], "failed": []}

    for team in teams:
        safe_name = _safe_filename(team)
        out_path_base = out_dir / "team_history" / safe_name
        if skip_existing and _csv_exists_nonempty(out_path_base):
            log.info("Skipping team history %s (existing CSV)", team)
            summary["skipped_existing"].append(team)
            continue

        try:
            df = _with_backoff(
                client.read_team_history,
                max_retries=max_retries,
                base_delay=backoff_base,
                kwargs={
                    "team": team,
                    "max_age_days": max_age_days,
                    "force_cache": force_cache,
                },
                count_as_network=not force_cache,
                periodic_every=periodic_every,
                periodic_secs=periodic_secs,
            )
            df = _apply_team_replacements(df, replacements)
            safe_write(df, out_path_base)
            summary["scraped_ok"].append(team)
        except Exception as e:
            log.warning("Failed to scrape ClubElo team history %s: %s", team, e)
            summary["failed"].append(team)
        _snooze(delay)

    return summary


def main() -> None:
    p = argparse.ArgumentParser("Scrape ClubElo date snapshots and team history")
    p.add_argument(
        "--date",
        nargs="*",
        default=None,
        help="Date snapshot(s) to pull (YYYY-MM-DD). If omitted and no --team, uses today.",
    )
    p.add_argument(
        "--team",
        nargs="*",
        default=None,
        help="Team name(s) to pull full ELO history.",
    )
    p.add_argument(
        "--league",
        nargs="*",
        default=None,
        help=(
            "ClubElo league code(s) (e.g., ENG_1). "
            "Used to pull all team histories for a league; requires --season or --date."
        ),
    )
    p.add_argument(
        "--season",
        nargs="*",
        default=None,
        help=(
            "Season(s) like 2024-2025. Used with --league to derive a mid-season "
            "snapshot date (Jan 1 of the end year)."
        ),
    )
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--data-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--no-cache", action="store_true", help="Bypass local cache for this run.")
    p.add_argument("--no-store", action="store_true", help="Do not store downloaded data to cache.")
    p.add_argument("--force-cache", action="store_true", help="Read only from cache; never hit network.")
    p.add_argument(
        "--max-age-days",
        type=int,
        default=1,
        help="Max age (days) of cached team history before re-download. Use 0 for always refresh.",
    )
    p.add_argument(
        "--no-max-age",
        action="store_true",
        help="Disable max-age checks and always use cached team history if present.",
    )
    p.add_argument("--delay", type=float, default=1.5, help="Base inter-request delay (seconds).")
    p.add_argument("--max-retries", type=int, default=6, help="Max 429 backoff retries per request.")
    p.add_argument("--backoff-base", type=float, default=3.2, help="Initial backoff (seconds) on HTTP 429.")
    p.add_argument("--periodic-every", type=int, default=8, help="Cool-down every N network calls.")
    p.add_argument("--periodic-secs", type=float, default=15.0, help="Cool-down length in seconds.")
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds.",
    )
    p.add_argument(
        "--proxy",
        type=str,
        default=None,
        help='Optional proxy (e.g., "tor" or "http://user:pass@host:port").',
    )
    p.add_argument(
        "--user-agent",
        type=str,
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        help="Override User-Agent header for ClubElo requests.",
    )
    p.add_argument(
        "--header",
        action="append",
        default=None,
        help="Extra HTTP header in the form 'Key: Value'. Can be repeated.",
    )
    p.add_argument(
        "--teamname-replacements",
        type=str,
        default=None,
        help="Optional JSON file mapping team names to replacements.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip outputs if the existing CSV is present and non-empty.",
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
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.force_cache and args.no_cache:
        raise SystemExit("Cannot use --force-cache and --no-cache together.")
    if args.no_max_age and args.max_age_days is not None:
        args.max_age_days = None

    init_logger(args.verbose)
    log = logging.getLogger("clubelo")

    headers = _parse_headers(args.user_agent, args.header)
    replacements = _load_team_replacements(args.teamname_replacements)

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.data_dir)
    meta_path = Path(args.meta_path)

    teams = [str(t) for t in (args.team or [])]
    leagues = [str(l) for l in (args.league or [])]
    seasons = [str(s) for s in (args.season or [])]

    if args.date is None:
        dates = [] if (teams or leagues) else _parse_date_args(None)
    else:
        dates = _parse_date_args(args.date)

    log.info(
        "ClubElo scrape configuration: dates=%s teams=%s leagues=%s seasons=%s "
        "no_cache=%s no_store=%s force_cache=%s out_dir=%s data_dir=%s "
        "skip_existing=%s max_age_days=%s timeout=%s",
        dates,
        teams if teams else "(none)",
        leagues if leagues else "(none)",
        seasons if seasons else "(none)",
        args.no_cache,
        args.no_store,
        args.force_cache,
        out_dir,
        cache_dir,
        args.skip_existing,
        args.max_age_days,
        args.timeout,
    )

    client = ClubEloClient(
        proxy=args.proxy,
        no_cache=args.no_cache,
        no_store=args.no_store,
        data_dir=cache_dir,
        headers=headers,
        timeout=args.timeout,
    )

    run_info: Dict[str, Any] = {
        "scrape_ts": datetime.now(timezone.utc).isoformat(),
        "mode": args.run_mode,
        "stats_summary": {},
    }

    if leagues:
        if args.date is not None:
            league_dates = _parse_date_args(args.date)
        elif seasons:
            league_dates = [_season_anchor_date(s) for s in seasons]
        else:
            raise SystemExit("When using --league, provide --season or --date for team discovery.")

        league_teams = _resolve_league_teams(
            client,
            leagues,
            league_dates,
            delay=args.delay,
            force_cache=args.force_cache,
            max_retries=args.max_retries,
            backoff_base=args.backoff_base,
            periodic_every=args.periodic_every,
            periodic_secs=args.periodic_secs,
        )

        if league_teams:
            log.info(
                "Resolved %d teams from leagues=%s dates=%s",
                len(league_teams),
                leagues,
                league_dates,
            )
            teams.extend(league_teams)
        else:
            log.warning("No teams resolved for leagues=%s dates=%s", leagues, league_dates)

    teams = list(dict.fromkeys(teams))

    if dates:
        summary_dates = scrape_dates(
            client,
            dates,
            out_dir,
            delay=args.delay,
            force_cache=args.force_cache,
            max_retries=args.max_retries,
            backoff_base=args.backoff_base,
            periodic_every=args.periodic_every,
            periodic_secs=args.periodic_secs,
            skip_existing=args.skip_existing,
            replacements=replacements,
        )
        run_info["stats_summary"]["date_snapshots"] = summary_dates
        for datestring in dates:
            job = ScrapeJobId(scraper="clubelo", league="clubelo", season=datestring, levels="date")
            record_last_run(meta_path, job, run_info=run_info)

    if teams:
        summary_teams = scrape_teams(
            client,
            teams,
            out_dir,
            delay=args.delay,
            force_cache=args.force_cache,
            max_age_days=args.max_age_days,
            max_retries=args.max_retries,
            backoff_base=args.backoff_base,
            periodic_every=args.periodic_every,
            periodic_secs=args.periodic_secs,
            skip_existing=args.skip_existing,
            replacements=replacements,
        )
        run_info["stats_summary"]["team_history"] = summary_teams
        for team in teams:
            job = ScrapeJobId(scraper="clubelo", league="clubelo", season=_safe_filename(team), levels="team_history")
            record_last_run(meta_path, job, run_info=run_info)

    log.info("ClubElo scrape completed. Total approximate network calls: %d", _GLOBAL["net_calls"])


if __name__ == "__main__":
    main()

