#!/usr/bin/env python3
# scripts/fotmob_pipeline/scrape/fotmob_stats_scraper.py
"""
FotMob scraper (API endpoints, cache-first)

Key features:
- Exponential backoff for HTTP 429 / transient errors (honors Retry-After).
- Jittered sleeps between requests to avoid synchronized bursts.
- Cache-first defaults (no_cache=False) + --force-cache for offline dev runs.
- Multi-league support: --league can accept multiple leagues in one go.
- Fine-grained control: --tables and --stat-types for team_match stats.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import requests
from requests.exceptions import HTTPError

from scripts.fbref_pipeline.automation.auto_scrape import ScrapeJobId, record_last_run

_GLOBAL = {"net_calls": 0}

FOTMOB_BASE = "https://www.fotmob.com"
FOTMOB_API = f"{FOTMOB_BASE}/api/"
FOTMOB_API_DATA = f"{FOTMOB_BASE}/api/data/"
COOKIE_SERVER = "http://46.101.91.154:6006/"

DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{FOTMOB_BASE}/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
}

STAT_TYPES = [
    "Top stats",
    "Shots",
    "Expected goals (xG)",
    "Passes",
    "Defence",
    "Duels",
    "Discipline",
]

TABLE_CHOICES = [
    "schedule",
    "league_table",
    "team_match",
    "player_match",
    "shot_events",
]


def init_logger(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def safe_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=True)
    logging.getLogger("fotmob").debug("saved %s", path.with_suffix("").name)


def make_game_id(row: pd.Series) -> str:
    date = row.get("date")
    if isinstance(date, pd.Timestamp):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = str(date)
    home = str(row.get("home_team", "")).strip()
    away = str(row.get("away_team", "")).strip()
    return f"{date_str}_{home}_vs_{away}"


def _snooze(base: float) -> None:
    jitter = base * random.uniform(0.2, 0.4)
    if base + jitter > 0:
        time.sleep(base + jitter)


def _periodic_rest(every: int, secs: float, *, skip: bool) -> None:
    if skip:
        return
    if every > 0 and _GLOBAL["net_calls"] > 0 and (_GLOBAL["net_calls"] % every == 0):
        logging.getLogger("fotmob").info(
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
    fn: Callable[..., Any],
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
    log = logging.getLogger("fotmob")

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


def _csv_exists(base: Path) -> bool:
    if base.is_file():
        return True
    if base.suffix == "":
        csv_path = base.with_suffix(".csv")
        if csv_path.is_file():
            return True
    return False


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


def _slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


def _season_display(season_id: Union[str, int]) -> str:
    s = str(season_id).strip().replace("/", "-")
    m = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(1)[:2]}{m.group(2)}"
    return s


def _normalize_team(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


class FotMobApi:
    def __init__(
        self,
        leagues: Optional[Union[str, List[str]]] = None,
        seasons: Optional[Union[str, int, Iterable[Union[str, int]]]] = None,
        proxy: Optional[str] = None,
        no_cache: bool = False,
        no_store: bool = False,
        data_dir: Path = Path("data/_fotmob_cache"),
        delay: float = 1.0,
        max_retries: int = 6,
        backoff_base: float = 3.2,
        periodic_every: int = 8,
        periodic_secs: float = 15.0,
        force_cache: bool = False,
        skip_cookie_server: bool = False,
    ) -> None:
        self.leagues = self._as_list(leagues)
        self.seasons = self._as_list(seasons)
        self.no_cache = no_cache
        self.no_store = no_store
        self.data_dir = data_dir
        self.delay = delay
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.periodic_every = periodic_every
        self.periodic_secs = periodic_secs
        self.force_cache = force_cache

        self._session = requests.Session()
        if proxy:
            if proxy == "tor":
                proxy = "socks5h://127.0.0.1:9050"
            self._session.proxies.update({"http": proxy, "https": proxy})

        self._headers = dict(DEFAULT_HEADERS)
        if not skip_cookie_server:
            self._try_cookie_headers()

    @staticmethod
    def _as_list(val: Optional[Union[str, int, Iterable[Union[str, int]]]]) -> List[str]:
        if val is None:
            return []
        if isinstance(val, (str, int)):
            return [str(val)]
        return [str(v) for v in val]

    def _try_cookie_headers(self) -> None:
        log = logging.getLogger("fotmob")
        try:
            resp = requests.get(COOKIE_SERVER, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                self._headers.update({k: v for k, v in data.items() if isinstance(v, str)})
                log.debug("Loaded session headers from cookie server.")
        except Exception as e:
            log.warning("Cookie server unavailable; proceeding without session headers (%s).", e)

    def _read_cache(self, filepath: Optional[Path], no_cache: bool) -> Optional[bytes]:
        if filepath is None:
            return None
        if filepath.exists() and not no_cache and not self.no_cache:
            payload = filepath.read_bytes()
            if payload not in (b"", b"{}", b"[]"):
                return payload
        return None

    def _write_cache(self, filepath: Optional[Path], payload: bytes) -> None:
        if self.no_store or filepath is None:
            return
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as fh:
            fh.write(payload)

    def _fetch(self, url: str) -> bytes:
        response = self._session.get(url, headers=self._headers, timeout=30)
        response.raise_for_status()
        return response.content

    def _get_payload(
        self,
        url: str,
        filepath: Optional[Path],
        *,
        no_cache: bool = False,
    ) -> Tuple[bytes, bool]:
        cached = self._read_cache(filepath, no_cache=no_cache)
        if cached is not None:
            return cached, True
        if self.force_cache:
            raise FileNotFoundError(f"Force-cache set but missing cache file: {filepath}")

        payload = _with_backoff(
            self._fetch,
            max_retries=self.max_retries,
            base_delay=self.backoff_base,
            kwargs={"url": url},
            count_as_network=True,
            periodic_every=self.periodic_every,
            periodic_secs=self.periodic_secs,
        )
        self._write_cache(filepath, payload)
        _snooze(self.delay)
        return payload, False

    def _get_json(self, url: str, filepath: Optional[Path], *, no_cache: bool = False) -> Dict[str, Any]:
        payload, _ = self._get_payload(url, filepath, no_cache=no_cache)
        return json.load(io.BytesIO(payload))

    def _get_match_details(self, match_id: int, filepath: Path) -> Dict[str, Any]:
        urls = [
            FOTMOB_API_DATA + f"matchDetails?matchId={match_id}",
            FOTMOB_API + f"matchDetails?matchId={match_id}",
        ]
        last_exc: Optional[Exception] = None
        for url in urls:
            try:
                return self._get_json(url, filepath)
            except HTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (403, 404):
                    last_exc = e
                    continue
                raise
            except Exception as e:
                last_exc = e
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("No matchDetails URLs available.")

    def read_leagues(self) -> pd.DataFrame:
        url = FOTMOB_API + "allLeagues"
        filepath = self.data_dir / "allLeagues.json"
        data = self._get_json(url, filepath)

        leagues: List[Dict[str, Any]] = []
        for key, val in data.items():
            if key == "international":
                if val:
                    for int_league in val[0].get("leagues", []):
                        leagues.append(
                            {
                                "region": val[0].get("ccode", ""),
                                "league_id": int_league.get("id"),
                                "league": int_league.get("name"),
                                "url": FOTMOB_BASE + int_league.get("pageUrl", ""),
                            }
                        )
            elif key not in ("favourite", "popular", "userSettings"):
                for country in val:
                    for dom_league in country.get("leagues", []):
                        leagues.append(
                            {
                                "region": country.get("ccode", ""),
                                "league": dom_league.get("name"),
                                "league_id": dom_league.get("id"),
                                "url": FOTMOB_BASE + dom_league.get("pageUrl", ""),
                            }
                        )

        if not leagues:
            return pd.DataFrame(columns=["region", "league_id", "league", "url"]).set_index("league")

        df = (
            pd.DataFrame(leagues)
            .assign(league=lambda x: x.region + "-" + x.league)
            .set_index("league")
            .sort_index()
        )

        if not self.leagues:
            return df

        requested = set(self.leagues)
        id_set = {int(x) for x in requested if str(x).isdigit()}
        name_set = {x for x in requested if not str(x).isdigit()}

        mask = df.index.isin(name_set)
        if id_set:
            mask = mask | df["league_id"].isin(id_set)
        return df.loc[mask]

    def read_seasons(self) -> pd.DataFrame:
        filemask = "leagues/{}.json"
        urlmask = FOTMOB_API + "leagues?id={}"

        df_leagues = self.read_leagues()
        seasons: List[Dict[str, Any]] = []
        for lkey, league in df_leagues.iterrows():
            url = urlmask.format(league.league_id)
            filepath = self.data_dir / filemask.format(league.league_id)
            data = self._get_json(url, filepath)
            for season in data.get("allAvailableSeasons", []):
                seasons.append(
                    {
                        "league": lkey,
                        "season": _season_display(season),
                        "league_id": league.league_id,
                        "season_id": season,
                        "url": str(league.url) + "?season=" + str(season),
                    }
                )

        if not seasons:
            return pd.DataFrame(columns=["league", "season", "league_id", "season_id", "url"]).set_index(
                ["league", "season"]
            )

        df = pd.DataFrame(seasons).set_index(["league", "season"]).sort_index()
        if not self.seasons:
            return df

        requested = {_season_display(s) for s in self.seasons}
        req_raw = set(self.seasons)
        mask = df.index.get_level_values("season").isin(requested) | df["season_id"].isin(req_raw)
        return df.loc[mask]

    def read_league_table(self, *, force_cache: bool = False) -> pd.DataFrame:
        filemask = "seasons/{}_{}.json"
        urlmask = FOTMOB_API + "leagues?id={}&season={}"

        idx = ["league", "season"]
        cols = ["team", "MP", "W", "D", "L", "GF", "GA", "GD", "Pts"]

        seasons = self.read_seasons()
        mult_tables = []
        for (lkey, skey), season in seasons.iterrows():
            filepath = self.data_dir / filemask.format(season.league_id, season.season_id)
            prev_force = self.force_cache
            if force_cache:
                self.force_cache = True
            url = urlmask.format(season.league_id, season.season_id)
            season_data = self._get_json(url, filepath)
            self.force_cache = prev_force

            table_data = season_data.get("table", [{}])[0].get("data", {})
            if "tables" in table_data:
                if "stage" not in idx:
                    idx.append("stage")
                groups_data = table_data["tables"]
                all_groups = []
                for group in groups_data:
                    group_table = pd.json_normalize(group["table"]["all"])
                    group_table["stage"] = group.get("leagueName")
                    all_groups.append(group_table)
                df_table = pd.concat(all_groups, axis=0)
            else:
                df_table = pd.json_normalize(table_data.get("table", {}).get("all", []))

            if df_table.empty:
                continue

            df_table[["GF", "GA"]] = df_table["scoresStr"].str.split("-", expand=True)
            df_table = df_table.rename(
                columns={
                    "name": "team",
                    "played": "MP",
                    "wins": "W",
                    "draws": "D",
                    "losses": "L",
                    "goalConDiff": "GD",
                    "pts": "Pts",
                }
            )
            df_table["league"] = lkey
            df_table["season"] = skey

            mult_tables.append(df_table)

        if not mult_tables:
            return pd.DataFrame(columns=idx + cols).set_index(idx)

        df = pd.concat(mult_tables, axis=0)
        return df.set_index(idx).sort_index()[cols]

    def read_schedule(self, *, force_cache: bool = False) -> pd.DataFrame:
        filemask = "seasons/{}_{}.json"
        urlmask = FOTMOB_API + "leagues?id={}&season={}"

        cols = [
            "round",
            "week",
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "status",
            "game_id",
            "url",
        ]

        df_seasons = self.read_seasons()
        all_schedules = []
        for (lkey, skey), season in df_seasons.iterrows():
            filepath = self.data_dir / filemask.format(season.league_id, season.season_id)
            prev_force = self.force_cache
            if force_cache:
                self.force_cache = True
            url = urlmask.format(season.league_id, season.season_id)
            season_data = self._get_json(url, filepath)
            self.force_cache = prev_force

            matches = season_data.get("fixtures", {}).get("allMatches", [])
            if not matches:
                continue

            df = pd.json_normalize(matches)
            df["league"] = lkey
            df["season"] = skey
            all_schedules.append(df)

        if not all_schedules:
            return pd.DataFrame(columns=["league", "season", "game"] + cols).set_index(
                ["league", "season", "game"]
            )

        df = (
            pd.concat(all_schedules)
            .rename(
                columns={
                    "roundName": "round",
                    "round": "week",
                    "home.name": "home_team",
                    "away.name": "away_team",
                    "status.reason.short": "status",
                    "pageUrl": "url",
                    "id": "game_id",
                }
            )
        )

        df["date"] = pd.to_datetime(df["status.utcTime"], errors="coerce", utc=True).dt.tz_convert(None)
        df["game"] = df.apply(make_game_id, axis=1)
        df["url"] = FOTMOB_BASE + df["url"].fillna("")
        df[["home_score", "away_score"]] = df["status.scoreStr"].str.split("-", expand=True)

        return df.set_index(["league", "season", "game"]).sort_index()[cols]

    def read_team_match_stats(
        self,
        *,
        stat_type: str = "Top stats",
        opponent_stats: bool = True,
        team: Optional[Sequence[str]] = None,
        force_cache: bool = False,
        match_ids: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        filemask = "matches/{}.json"

        df_matches = self.read_schedule(force_cache=force_cache)
        df_complete = df_matches.loc[df_matches["status"].isin(["FT", "AET", "Pen"])]

        if match_ids:
            match_ids_set = {int(m) for m in match_ids}
            df_complete = df_complete.loc[df_complete["game_id"].isin(match_ids_set)]

        if team is not None:
            wanted = {_normalize_team(t) for t in team}
            team_mask = df_complete["home_team"].map(_normalize_team).isin(wanted) | df_complete[
                "away_team"
            ].map(_normalize_team).isin(wanted)
            df_complete = df_complete.loc[team_mask]

        if df_complete.empty:
            return pd.DataFrame(columns=["league", "season", "game", "team"]).set_index(
                ["league", "season", "game", "team"]
            )

        stats = []
        for i, game in df_complete.reset_index().iterrows():
            lkey, skey, gkey = game["league"], game["season"], game["game"]
            filepath = self.data_dir / filemask.format(game["game_id"])
            prev_force = self.force_cache
            if force_cache:
                self.force_cache = True
            logger = logging.getLogger("fotmob")
            logger.info("[%s/%s] Retrieving matchId=%s", i + 1, len(df_complete), game["game_id"])

            game_data = self._get_match_details(int(game["game_id"]), filepath)
            self.force_cache = prev_force
            all_stats = (
                game_data.get("content", {})
                .get("stats", {})
                .get("Periods", {})
                .get("All", {})
                .get("stats", [])
            )
            try:
                selected_stats = next(stat for stat in all_stats if stat.get("title") == stat_type)
            except StopIteration:
                raise ValueError(f"Invalid stat type: {stat_type}")

            df_raw_stats = pd.DataFrame(selected_stats.get("stats", []))
            if df_raw_stats.empty or "stats" not in df_raw_stats.columns:
                continue
            game_teams = [game["home_team"], game["away_team"]]

            for idx, team_name in enumerate(game_teams):
                df_team_stats = df_raw_stats.copy()

                def _stat_at(stats: Any) -> Any:
                    if isinstance(stats, (list, tuple)) and len(stats) > idx:
                        return stats[idx]
                    return None

                df_team_stats["stat"] = df_team_stats["stats"].apply(_stat_at)
                df_team_stats["league"] = lkey
                df_team_stats["season"] = skey
                df_team_stats["game"] = gkey
                df_team_stats["team"] = team_name

                if not opponent_stats and team is not None:
                    wanted = {_normalize_team(t) for t in team}
                    if _normalize_team(team_name) not in wanted:
                        continue

                if "type" in df_team_stats.columns:
                    df_team_stats = df_team_stats[df_team_stats["type"] != "title"]
                df_team_stats = df_team_stats.pivot_table(
                    index=["league", "season", "game", "team"],
                    columns="title",
                    values="stat",
                    aggfunc="first",
                ).reset_index()
                df_team_stats.columns.name = None
                stats.append(df_team_stats)

        if not stats:
            return pd.DataFrame(columns=["league", "season", "game", "team"]).set_index(
                ["league", "season", "game", "team"]
            )

        df = pd.concat(stats, axis=0).set_index(["league", "season", "game", "team"]).sort_index()
        pct_cols = [col for col in df.columns if df[col].astype(str).str.contains("%").any()]
        for col in pct_cols:
            df[[col, col + " (%)"]] = df[col].astype(str).str.split(expand=True, n=1)
            df[col + " (%)"] = (
                df[col + " (%)"].str.extract(r"(\d+(?:\.\d+)?)").astype(float).div(100)
            )
        return df

    def read_player_match_stats(
        self,
        *,
        stat_groups: Optional[Sequence[str]] = None,
        team: Optional[Sequence[str]] = None,
        force_cache: bool = False,
        match_ids: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        filemask = "matches/{}.json"

        df_matches = self.read_schedule(force_cache=force_cache)
        df_complete = df_matches.loc[df_matches["status"].isin(["FT", "AET", "Pen"])]

        if match_ids:
            match_ids_set = {int(m) for m in match_ids}
            df_complete = df_complete.loc[df_complete["game_id"].isin(match_ids_set)]

        if team is not None:
            wanted = {_normalize_team(t) for t in team}
            team_mask = df_complete["home_team"].map(_normalize_team).isin(wanted) | df_complete[
                "away_team"
            ].map(_normalize_team).isin(wanted)
            df_complete = df_complete.loc[team_mask]

        if df_complete.empty:
            return pd.DataFrame(
                columns=[
                    "league",
                    "season",
                    "game",
                    "player_id",
                    "player_name",
                    "team_id",
                    "team",
                    "stat_group",
                    "stat_group_key",
                    "stat_name",
                    "stat_key",
                    "value",
                    "total",
                    "stat_type",
                    "bonus",
                    "medal",
                    "hide_in_popup",
                ]
            )

        rows: List[Dict[str, Any]] = []
        group_filter = {g.lower() for g in stat_groups} if stat_groups else None

        for i, game in df_complete.reset_index().iterrows():
            lkey, skey, gkey = game["league"], game["season"], game["game"]
            filepath = self.data_dir / filemask.format(game["game_id"])
            prev_force = self.force_cache
            if force_cache:
                self.force_cache = True
            logger = logging.getLogger("fotmob")
            logger.info("[%s/%s] Retrieving matchId=%s", i + 1, len(df_complete), game["game_id"])

            game_data = self._get_match_details(int(game["game_id"]), filepath)
            self.force_cache = prev_force

            player_stats = game_data.get("content", {}).get("playerStats", {}) or {}
            if not isinstance(player_stats, dict):
                continue

            for player_id, pdata in player_stats.items():
                base = {
                    "league": lkey,
                    "season": skey,
                    "game": gkey,
                    "player_id": int(player_id),
                    "player_name": pdata.get("name"),
                    "team_id": pdata.get("teamId"),
                    "team": pdata.get("teamName"),
                }

                for group in pdata.get("stats", []):
                    group_title = group.get("title")
                    if group_filter and str(group_title).lower() not in group_filter:
                        continue
                    group_key = group.get("key")
                    stats_dict = group.get("stats", {}) or {}
                    if not isinstance(stats_dict, dict):
                        continue
                    for stat_name, entry in stats_dict.items():
                        stat_key = entry.get("key")
                        stat_block = entry.get("stat", {}) or {}
                        rows.append(
                            {
                                **base,
                                "stat_group": group_title,
                                "stat_group_key": group_key,
                                "stat_name": stat_name,
                                "stat_key": stat_key,
                                "value": stat_block.get("value"),
                                "total": stat_block.get("total"),
                                "stat_type": stat_block.get("type"),
                                "bonus": stat_block.get("bonus"),
                                "medal": entry.get("medal"),
                                "hide_in_popup": entry.get("hideInPopupCard"),
                            }
                        )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "league",
                    "season",
                    "game",
                    "player_id",
                    "player_name",
                    "team_id",
                    "team",
                    "stat_group",
                    "stat_group_key",
                    "stat_name",
                    "stat_key",
                    "value",
                    "total",
                    "stat_type",
                    "bonus",
                    "medal",
                    "hide_in_popup",
                ]
            )

        return pd.DataFrame(rows)

    def read_shot_events(
        self,
        *,
        force_cache: bool = False,
        match_ids: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        filemask = "matches/{}.json"

        df_matches = self.read_schedule(force_cache=force_cache)
        df_complete = df_matches.loc[df_matches["status"].isin(["FT", "AET", "Pen"])]

        if match_ids:
            match_ids_set = {int(m) for m in match_ids}
            df_complete = df_complete.loc[df_complete["game_id"].isin(match_ids_set)]

        if df_complete.empty:
            return pd.DataFrame(
                columns=["league", "season", "game", "match_id"]
            )

        frames: List[pd.DataFrame] = []
        for i, game in df_complete.reset_index().iterrows():
            lkey, skey, gkey = game["league"], game["season"], game["game"]
            filepath = self.data_dir / filemask.format(game["game_id"])
            prev_force = self.force_cache
            if force_cache:
                self.force_cache = True
            logger = logging.getLogger("fotmob")
            logger.info("[%s/%s] Retrieving matchId=%s", i + 1, len(df_complete), game["game_id"])

            game_data = self._get_match_details(int(game["game_id"]), filepath)
            self.force_cache = prev_force

            shots = (
                game_data.get("content", {})
                .get("shotmap", {})
                .get("shots", [])
            )
            if not shots:
                continue
            df_shots = pd.json_normalize(shots)
            df_shots["league"] = lkey
            df_shots["season"] = skey
            df_shots["game"] = gkey
            df_shots["match_id"] = game["game_id"]
            frames.append(df_shots)

        if not frames:
            return pd.DataFrame(
                columns=["league", "season", "game", "match_id"]
            )

        return pd.concat(frames, axis=0).reset_index(drop=True)


def _seasons_from_league(
    league: str,
    *,
    no_cache: bool,
    proxy: Optional[str],
    data_dir: Path,
    delay: float,
    max_retries: int,
    backoff_base: float,
    periodic_every: int,
    periodic_secs: float,
    force_cache: bool,
    skip_cookie_server: bool,
) -> List[str]:
    fm = FotMobApi(
        leagues=league,
        no_cache=no_cache,
        proxy=proxy,
        data_dir=data_dir,
        delay=delay,
        max_retries=max_retries,
        backoff_base=backoff_base,
        periodic_every=periodic_every,
        periodic_secs=periodic_secs,
        force_cache=force_cache,
        skip_cookie_server=skip_cookie_server,
    )
    df = fm.read_seasons()
    return df.index.get_level_values("season").unique().tolist()


def scrape_one(
    *,
    league: str,
    season: str,
    out_base: Path,
    delay: float,
    no_cache: bool,
    no_store: bool,
    force_cache: bool,
    max_retries: int,
    backoff_base: float,
    periodic_every: int,
    periodic_secs: float,
    proxy: Optional[str],
    tables: Sequence[str],
    stat_types: Sequence[str],
    player_stat_groups: Optional[Sequence[str]],
    team: Optional[Sequence[str]],
    opponent_stats: bool,
    match_ids: Optional[Sequence[int]],
    skip_existing: bool,
    data_dir: Path,
    skip_cookie_server: bool,
) -> Dict[str, Any]:
    log = logging.getLogger("fotmob")
    log.info("FotMob: %s %s", league, season)

    fm = FotMobApi(
        leagues=league,
        seasons=season,
        proxy=proxy,
        no_cache=no_cache,
        no_store=no_store,
        data_dir=data_dir,
        delay=delay,
        max_retries=max_retries,
        backoff_base=backoff_base,
        periodic_every=periodic_every,
        periodic_secs=periodic_secs,
        force_cache=force_cache,
        skip_cookie_server=skip_cookie_server,
    )

    out_dir = out_base / league / str(season)
    summary: Dict[str, Any] = {
        "schedule": None,
        "league_table": None,
        "team_match": {},
        "player_match": None,
        "shot_events": None,
    }

    if "schedule" in tables:
        out_path = out_dir / "schedule"
        if skip_existing and _csv_exists(out_path):
            summary["schedule"] = "skipped_existing"
        else:
            df_schedule = fm.read_schedule(force_cache=force_cache)
            safe_write(df_schedule, out_path)
            summary["schedule"] = "scraped_ok"

    if "league_table" in tables:
        out_path = out_dir / "league_table"
        if skip_existing and _csv_exists(out_path):
            summary["league_table"] = "skipped_existing"
        else:
            df_table = fm.read_league_table(force_cache=force_cache)
            safe_write(df_table, out_path)
            summary["league_table"] = "scraped_ok"

    if "team_match" in tables:
        for stat_type in stat_types:
            stat_slug = _slugify(stat_type)
            out_path = out_dir / "team_match" / stat_slug
            if skip_existing and _csv_exists(out_path):
                summary["team_match"][stat_type] = "skipped_existing"
                continue
            df_stats = fm.read_team_match_stats(
                stat_type=stat_type,
                opponent_stats=opponent_stats,
                team=team,
                force_cache=force_cache,
                match_ids=match_ids,
            )
            safe_write(df_stats, out_path)
            summary["team_match"][stat_type] = "scraped_ok"

    if "player_match" in tables:
        out_path = out_dir / "player_match"
        if skip_existing and _csv_exists(out_path):
            summary["player_match"] = "skipped_existing"
        else:
            df_player = fm.read_player_match_stats(
                stat_groups=player_stat_groups,
                team=team,
                force_cache=force_cache,
                match_ids=match_ids,
            )
            safe_write(df_player, out_path)
            summary["player_match"] = "scraped_ok"

    if "shot_events" in tables:
        out_path = out_dir / "shot_events"
        if skip_existing and _csv_exists(out_path):
            summary["shot_events"] = "skipped_existing"
        else:
            df_shots = fm.read_shot_events(
                force_cache=force_cache,
                match_ids=match_ids,
            )
            safe_write(df_shots, out_path)
            summary["shot_events"] = "scraped_ok"

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="FotMob scraper (cache-first)")
    p.add_argument("--league", action="append", help="League name or ID (repeatable).")
    p.add_argument("--all-leagues", action="store_true", help="Scrape all leagues in FotMob.")
    p.add_argument("--seasons", action="append", help="Seasons to scrape (repeatable).")
    p.add_argument(
        "--tables",
        nargs="+",
        choices=TABLE_CHOICES,
        default=TABLE_CHOICES,
        help="Subset of tables to scrape.",
    )
    p.add_argument(
        "--stat-types",
        nargs="+",
        default=STAT_TYPES,
        help="FotMob stat types for team_match (default: all known types).",
    )
    p.add_argument(
        "--player-stat-groups",
        nargs="+",
        default=None,
        help='Filter player_match to these groups (default: all).',
    )
    p.add_argument("--team", action="append", help="Limit team_match stats to these teams.")
    p.add_argument(
        "--opponent-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include opponent stats (default: true).",
    )
    p.add_argument("--match-ids", action="append", help="Match IDs (comma-separated ok).")
    p.add_argument("--out-dir", default="data/raw/fotmob", help="Output base directory.")
    p.add_argument("--cache-dir", default="data/_fotmob_cache", help="Cache directory for raw JSON.")
    p.add_argument("--delay", type=float, default=1.0, help="Base delay between requests (seconds).")
    p.add_argument("--max-retries", type=int, default=6, help="Max retries for 429/backoff.")
    p.add_argument("--backoff-base", type=float, default=3.2, help="Base delay for backoff.")
    p.add_argument("--periodic-every", type=int, default=8, help="Rest every N network calls.")
    p.add_argument("--periodic-secs", type=float, default=15.0, help="Rest duration in seconds.")
    p.add_argument("--proxy", type=str, default=None, help='Proxy (e.g., "tor" or URL).')
    p.add_argument("--no-cache", action="store_true", help="Bypass cache for this run.")
    p.add_argument("--no-store", action="store_true", help="Do not write cache to disk.")
    p.add_argument("--force-cache", action="store_true", help="Only read from cache; no network.")
    p.add_argument("--skip-existing", action="store_true", help="Skip outputs that already exist.")
    p.add_argument(
        "--skip-cookie-server",
        action="store_true",
        help="Skip FotMob cookie-header bootstrap server.",
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
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")

    args = p.parse_args()
    init_logger(args.verbose)
    log = logging.getLogger("fotmob")

    if args.no_cache and args.force_cache:
        raise ValueError("--no-cache and --force-cache are mutually exclusive.")

    leagues = _parse_list(args.league) or []
    if args.all_leagues:
        leagues = []

    match_ids_raw = _parse_list(args.match_ids)
    match_ids = [int(x) for x in match_ids_raw] if match_ids_raw else None

    data_dir = Path(args.cache_dir)
    out_base = Path(args.out_dir)
    meta_path = Path(args.meta_path)

    if args.all_leagues or not leagues:
        fm = FotMobApi(
            no_cache=args.no_cache,
            proxy=args.proxy,
            data_dir=data_dir,
            delay=args.delay,
            max_retries=args.max_retries,
            backoff_base=args.backoff_base,
            periodic_every=args.periodic_every,
            periodic_secs=args.periodic_secs,
            force_cache=args.force_cache,
            skip_cookie_server=args.skip_cookie_server,
        )
        leagues = fm.read_leagues().index.tolist()
        if not leagues:
            log.warning("No leagues resolved; exiting.")
            return

    for league in leagues:
        seasons = _parse_list(args.seasons) or _seasons_from_league(
            league,
            no_cache=args.no_cache,
            proxy=args.proxy,
            data_dir=data_dir,
            delay=args.delay,
            max_retries=args.max_retries,
            backoff_base=args.backoff_base,
            periodic_every=args.periodic_every,
            periodic_secs=args.periodic_secs,
            force_cache=args.force_cache,
            skip_cookie_server=args.skip_cookie_server,
        )
        if not seasons:
            log.warning("No seasons resolved for %s; skipping.", league)
            continue

        log.info("Resolved seasons for %s: %s", league, seasons)

        for s in seasons:
            summary = scrape_one(
                league=league,
                season=str(s),
                out_base=out_base,
                delay=args.delay,
                no_cache=args.no_cache,
                no_store=args.no_store,
                force_cache=args.force_cache,
                max_retries=args.max_retries,
                backoff_base=args.backoff_base,
                periodic_every=args.periodic_every,
                periodic_secs=args.periodic_secs,
                proxy=args.proxy,
                tables=args.tables,
                stat_types=args.stat_types,
                player_stat_groups=args.player_stat_groups,
                team=_parse_list(args.team),
                opponent_stats=args.opponent_stats,
                match_ids=match_ids,
                skip_existing=args.skip_existing,
                data_dir=data_dir,
                skip_cookie_server=args.skip_cookie_server,
            )

            job = ScrapeJobId(
                scraper="fotmob",
                league=league,
                season=str(s),
                levels="tables",
            )

            run_info = {
                "scrape_ts": datetime.now(timezone.utc).isoformat(),
                "mode": args.run_mode,
                "stats_summary": summary,
                "tables": list(args.tables),
            }

            record_last_run(meta_path, job, run_info=run_info)
            log.info("Recorded last-run meta for %s", job.key())

    log.info(
        "FotMob scrape completed for leagues=%s. Total approximate network calls: %d",
        leagues,
        _GLOBAL["net_calls"],
    )


if __name__ == "__main__":
    main()
