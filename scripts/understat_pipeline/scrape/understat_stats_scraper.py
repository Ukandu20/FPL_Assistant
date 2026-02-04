#!/usr/bin/env python3
# scripts/understat_pipeline/scrape/understat_stats_scraper.py
"""
Understat scraper (API endpoints, cache-first)

Key features:
- Exponential backoff for HTTP 429 / transient errors (honors Retry-After).
- Jittered sleeps between requests to avoid synchronized bursts.
- Cache-first defaults (no_cache=False) + --force-cache for offline dev runs.
- Multi-league support: --league can accept multiple leagues in one go.
- Fine-grained control: --tables, --match-ids, --skip-existing.
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
from typing import Any, Callable, Dict, IO, Iterable, List, Optional, Sequence, Union

import pandas as pd
import requests
from requests.exceptions import HTTPError

from scripts.fbref_pipeline.automation.auto_scrape import ScrapeJobId, record_last_run

_GLOBAL = {"net_calls": 0}

UNDERSTAT_URL = "https://understat.com"
UNDERSTAT_HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Referer": UNDERSTAT_URL,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
}
UNDERSTAT_DATADIR = Path.home() / "soccerdata" / "data" / "Understat"

SHOT_SITUATIONS = {
    "OpenPlay": "Open Play",
    "FromCorner": "From Corner",
    "SetPiece": "Set Piece",
    "DirectFreekick": "Direct Freekick",
}

SHOT_BODY_PARTS = {
    "RightFoot": "Right Foot",
    "LeftFoot": "Left Foot",
    "OtherBodyParts": "Other",
}

SHOT_RESULTS = {
    "Goal": "Goal",
    "OwnGoal": "Own Goal",
    "BlockedShot": "Blocked Shot",
    "SavedShot": "Saved Shot",
    "MissedShots": "Missed Shot",
    "ShotOnPost": "Shot On Post",
}

TABLE_CHOICES = [
    "schedule",
    "team_match",
    "player_season",
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
    logging.getLogger("understat").debug("saved %s", path.with_suffix("").name)


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
        logging.getLogger("understat").info(
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
    log = logging.getLogger("understat")

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


def _load_existing_csv(base: Path) -> Optional[pd.DataFrame]:
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
                logging.getLogger("understat").warning("Failed to read existing CSV %s: %s", p, e)
                return None
    return None


def _parse_match_ids(raw: Optional[Sequence[str]]) -> Optional[List[int]]:
    if not raw:
        return None
    out: List[int] = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            out.append(int(part))
    return out or None


def _as_bool(value: Any) -> Optional[bool]:
    try:
        return bool(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> Optional[str]:
    try:
        return str(value)
    except (TypeError, ValueError):
        return None


class UnderstatApi:
    def __init__(
        self,
        leagues: Optional[Union[str, List[str]]] = None,
        seasons: Optional[Union[str, int, Iterable[Union[str, int]]]] = None,
        proxy: Optional[str] = None,
        no_cache: bool = False,
        no_store: bool = False,
        data_dir: Path = UNDERSTAT_DATADIR,
    ) -> None:
        self.leagues = self._as_list(leagues)
        self.seasons = self._as_list(seasons)
        self.no_cache = no_cache
        self.no_store = no_store
        self.data_dir = data_dir
        self._session = requests.Session()
        if proxy:
            if proxy == "tor":
                proxy = "socks5h://127.0.0.1:9050"
            self._session.proxies.update({"http": proxy, "https": proxy})
        self._cookies_initialized = False

    @staticmethod
    def _as_list(val: Optional[Union[str, int, Iterable[Union[str, int]]]]) -> List[str]:
        if val is None:
            return []
        if isinstance(val, (str, int)):
            return [str(val)]
        return [str(v) for v in val]

    def _ensure_cookies(self) -> None:
        if not self._cookies_initialized:
            self._session.get(UNDERSTAT_URL, timeout=30)
            self._cookies_initialized = True

    def _request_api(self, url: str, filepath: Optional[Path] = None, no_cache: bool = False) -> IO[bytes]:
        is_cached = (
            filepath is not None and filepath.exists() and not no_cache and not self.no_cache
        )
        if is_cached and filepath is not None:
            payload = filepath.read_bytes()
            if payload not in (b"", b"{}", b"[]"):
                return io.BytesIO(payload)

        self._ensure_cookies()
        response = self._session.get(url, headers=UNDERSTAT_HEADERS, timeout=30)
        response.raise_for_status()
        payload = response.content

        if not self.no_store and filepath is not None:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with filepath.open(mode="wb") as fh:
                fh.write(payload)

        return io.BytesIO(payload)

    def _read_leagues(self, no_cache: bool = False) -> Dict[str, Any]:
        url = UNDERSTAT_URL + "/getStatData"
        filepath = self.data_dir / "leagues.json"
        reader = self._request_api(url, filepath, no_cache=no_cache)
        data = json.load(reader)
        if isinstance(data, list):
            return {"statData": data}
        if isinstance(data, dict):
            if "stat" in data:
                return {"statData": data["stat"]}
            if "statData" in data:
                return {"statData": data["statData"]}
        raise KeyError(f"Unexpected /getStatData response keys: {list(getattr(data, 'keys', lambda: [])())}")

    @staticmethod
    def _extract_team_name(html: str) -> str:
        match = re.search(r"<h3><a[^>]*>([^<]+)</a></h3>", html)
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def _season_year(season: str) -> Optional[int]:
        try:
            return int(str(season).split("/")[0].split("-")[0])
        except Exception:
            return None

    def _is_complete(self, season: str) -> bool:
        year = self._season_year(season)
        if year is None:
            return False
        return year < datetime.now().year

    def read_leagues(self) -> pd.DataFrame:
        data = self._read_leagues()
        leagues: Dict[int, Dict[str, Any]] = {}

        league_data = data["statData"]
        for league_stat in league_data:
            league_id = league_stat["league_id"]
            if league_id not in leagues:
                league = league_stat["league"]
                league_slug = league.replace(" ", "_")
                leagues[league_id] = {
                    "league_id": league_id,
                    "league": league,
                    "url": UNDERSTAT_URL + f"/league/{league_slug}",
                }

        index = "league"
        if len(leagues) == 0:
            return pd.DataFrame(index=index)

        df = pd.DataFrame.from_records(list(leagues.values())).set_index(index).sort_index()
        if self.leagues:
            valid_leagues = [league for league in self.leagues if league in df.index]
            return df.loc[valid_leagues]
        return df

    def read_seasons(self) -> pd.DataFrame:
        data = self._read_leagues()

        seasons: Dict[tuple, Dict[str, Any]] = {}
        league_data = data["statData"]
        for league_stat in league_data:
            league_id = league_stat["league_id"]
            year = int(league_stat["year"])
            month = int(league_stat["month"])
            season_id = year if month >= 7 else year - 1
            key = (league_id, season_id)
            if key not in seasons:
                league = league_stat["league"]
                league_slug = league.replace(" ", "_")
                seasons[key] = {
                    "league_id": league_id,
                    "league": league,
                    "season_id": season_id,
                    "season": str(season_id),
                    "url": UNDERSTAT_URL + f"/league/{league_slug}/{season_id}",
                }

        index = ["league", "season"]
        if len(seasons) == 0:
            return pd.DataFrame(index=index)

        df = pd.DataFrame.from_records(list(seasons.values())).set_index(index).sort_index()

        if self.leagues and self.seasons:
            all_seasons = [(l, s) for l in self.leagues for s in self.seasons]
            valid_seasons = [season for season in all_seasons if season in df.index]
            return df.loc[valid_seasons]
        if self.leagues:
            return df.loc[(self.leagues, slice(None)), :]
        if self.seasons:
            return df.loc[(slice(None), self.seasons), :]
        return df

    def _read_league_season(self, url: str, league_id: int, season_id: int, no_cache: bool = False) -> Dict[str, Any]:
        parts = url.rstrip("/").split("/")
        league_slug = parts[-2]
        season = parts[-1]
        api_url = UNDERSTAT_URL + f"/getLeagueData/{league_slug}/{season}"
        filepath = self.data_dir / f"league_{league_id}_season_{season_id}.json"
        reader = self._request_api(api_url, filepath, no_cache=no_cache)
        data = json.load(reader)
        return {
            "datesData": data["dates"],
            "playersData": data["players"],
            "teamsData": data["teams"],
        }

    def _read_match(self, match_id: int) -> Optional[Dict[str, Any]]:
        try:
            api_url = UNDERSTAT_URL + f"/getMatchData/{match_id}"
            filepath = self.data_dir / f"match_{match_id}.json"
            reader = self._request_api(api_url, filepath)
            data = json.load(reader)

            home_team_name = self._extract_team_name(data["tmpl"]["home"])
            away_team_name = self._extract_team_name(data["tmpl"]["away"])
            rosters = data["rosters"]
            home_team_id = next(iter(rosters["h"].values()))["team_id"]
            away_team_id = next(iter(rosters["a"].values()))["team_id"]

            match_info = {
                "h": home_team_id,
                "a": away_team_id,
                "team_h": home_team_name,
                "team_a": away_team_name,
            }

            return {
                "match_info": match_info,
                "rostersData": rosters,
                "shotsData": data["shots"],
            }
        except ConnectionError:
            return None

    @staticmethod
    def available_leagues(
        no_cache: bool = False,
        no_store: bool = False,
        proxy: Optional[str] = None,
        data_dir: Path = UNDERSTAT_DATADIR,
    ) -> List[str]:
        reader = UnderstatApi(
            leagues=None,
            seasons=None,
            proxy=proxy,
            no_cache=no_cache,
            no_store=no_store,
            data_dir=data_dir,
        )
        data = reader._read_leagues(no_cache=no_cache)
        leagues = sorted({d["league"] for d in data["statData"]})
        return leagues

    def read_schedule(self, include_matches_without_data: bool = True, force_cache: bool = False) -> pd.DataFrame:
        df_seasons = self.read_seasons()
        matches: List[Dict[str, Any]] = []

        for (league, season), league_season in df_seasons.iterrows():
            league_id = league_season["league_id"]
            season_id = league_season["season_id"]
            url = league_season["url"]

            is_current_season = not self._is_complete(season)
            no_cache = is_current_season and not force_cache

            data = self._read_league_season(url, league_id, season_id, no_cache)
            matches_data = data["datesData"]
            for match in matches_data:
                match_id = _as_int(match["id"])
                has_home_xg = match["xG"]["h"] not in ("0", None)
                has_away_xg = match["xG"]["a"] not in ("0", None)
                has_data = has_home_xg or has_away_xg
                matches.append(
                    {
                        "league_id": league_id,
                        "league": league,
                        "season_id": season_id,
                        "season": season,
                        "game_id": match_id,
                        "date": match["datetime"],
                        "home_team_id": _as_int(match["h"]["id"]),
                        "away_team_id": _as_int(match["a"]["id"]),
                        "home_team": _as_str(match["h"]["title"]),
                        "away_team": _as_str(match["a"]["title"]),
                        "away_team_code": match["a"]["short_title"],
                        "home_team_code": match["h"]["short_title"],
                        "home_goals": _as_int(match["goals"]["h"]),
                        "away_goals": _as_int(match["goals"]["a"]),
                        "home_xg": _as_float(match["xG"]["h"]),
                        "away_xg": _as_float(match["xG"]["a"]),
                        "is_result": _as_bool(match["isResult"]),
                        "has_data": has_data,
                        "url": UNDERSTAT_URL + f"/match/{match_id}",
                    }
                )

        index = ["league", "season", "game"]
        if len(matches) == 0:
            return pd.DataFrame(index=index)

        df = (
            pd.DataFrame.from_records(matches)
            .assign(date=lambda g: pd.to_datetime(g["date"], format="%Y-%m-%d %H:%M:%S"))
            .assign(game=lambda g: g.apply(make_game_id, axis=1))
            .set_index(index)
            .sort_index()
            .convert_dtypes()
        )

        if not include_matches_without_data:
            df = df[df["has_data"]]

        return df

    def read_team_match_stats(self, force_cache: bool = False) -> pd.DataFrame:
        df_seasons = self.read_seasons()
        stats: Dict[int, Dict[str, Any]] = {}

        for (league, season), league_season in df_seasons.iterrows():
            league_id = league_season["league_id"]
            season_id = league_season["season_id"]
            url = league_season["url"]

            is_current_season = not self._is_complete(season)
            no_cache = is_current_season and not force_cache

            data = self._read_league_season(url, league_id, season_id, no_cache)
            schedule: Dict[int, Dict[str, Any]] = {}
            matches: Dict[tuple, int] = {}

            matches_data = data["datesData"]
            for match in matches_data:
                match_id = _as_int(match["id"])
                match_date = match["datetime"]
                schedule[match_id] = {
                    "league_id": league_id,
                    "league": league,
                    "season_id": season_id,
                    "season": season,
                    "game_id": match_id,
                    "date": match["datetime"],
                    "home_team_id": _as_int(match["h"]["id"]),
                    "away_team_id": _as_int(match["a"]["id"]),
                    "home_team": _as_str(match["h"]["title"]),
                    "away_team": _as_str(match["a"]["title"]),
                    "away_team_code": _as_str(match["a"]["short_title"]),
                    "home_team_code": _as_str(match["h"]["short_title"]),
                }
                for side in ("h", "a"):
                    team_id = _as_int(match[side]["id"])
                    matches[(match_date, team_id)] = match_id

            teams_data = data["teamsData"]
            for team in teams_data.values():
                team_id = _as_int(team["id"])
                for match in team["history"]:
                    match_date = match["date"]
                    match_id = matches[(match_date, team_id)]
                    team_side = match["h_a"]
                    prefix = "home" if team_side == "h" else "away"

                    if match_id not in stats:
                        stats[match_id] = schedule[match_id]

                    ppda = match["ppda"]
                    team_ppda = (ppda["att"] / ppda["def"]) if ppda["def"] != 0 else pd.NA

                    stats[match_id].update(
                        {
                            f"{prefix}_points": _as_int(match["pts"]),
                            f"{prefix}_expected_points": _as_float(match["xpts"]),
                            f"{prefix}_goals": _as_int(match["scored"]),
                            f"{prefix}_xg": _as_float(match["xG"]),
                            f"{prefix}_np_xg": _as_float(match["npxG"]),
                            f"{prefix}_np_xg_difference": _as_float(match["npxGD"]),
                            f"{prefix}_ppda": _as_float(team_ppda),
                            f"{prefix}_deep_completions": _as_int(match["deep"]),
                        }
                    )

        index = ["league", "season", "game"]
        if len(stats) == 0:
            return pd.DataFrame(index=index)

        return (
            pd.DataFrame.from_records(list(stats.values()))
            .assign(date=lambda g: pd.to_datetime(g["date"], format="%Y-%m-%d %H:%M:%S"))
            .assign(game=lambda g: g.apply(make_game_id, axis=1))
            .set_index(index)
            .sort_index()
            .convert_dtypes()
        )

    def read_player_season_stats(self, force_cache: bool = False) -> pd.DataFrame:
        df_seasons = self.read_seasons()
        stats: List[Dict[str, Any]] = []

        for (league, season), league_season in df_seasons.iterrows():
            league_id = league_season["league_id"]
            season_id = league_season["season_id"]
            url = league_season["url"]

            is_current_season = not self._is_complete(season)
            no_cache = is_current_season and not force_cache

            data = self._read_league_season(url, league_id, season_id, no_cache)
            teams_data = data["teamsData"]
            team_mapping: Dict[str, int] = {}
            for team in teams_data.values():
                team_name = _as_str(team["title"])
                team_id = _as_int(team["id"])
                team_mapping[str(team_name)] = int(team_id) if team_id is not None else team_id

            players_data = data["playersData"]
            for player in players_data:
                player_team_name = player["team_title"]
                if "," in player_team_name:
                    player_team_name = player_team_name.split(",")[0]
                player_team_name = _as_str(player_team_name)
                player_team_id = team_mapping.get(str(player_team_name))
                stats.append(
                    {
                        "league": league,
                        "league_id": league_id,
                        "season": season,
                        "season_id": season_id,
                        "team": player_team_name,
                        "team_id": player_team_id,
                        "player": _as_str(player["player_name"]),
                        "player_id": _as_int(player["id"]),
                        "position": _as_str(player["position"]),
                        "matches": _as_int(player["games"]),
                        "minutes": _as_int(player["time"]),
                        "goals": _as_int(player["goals"]),
                        "xg": _as_float(player["xG"]),
                        "np_goals": _as_int(player["npg"]),
                        "np_xg": _as_float(player["npxG"]),
                        "assists": _as_int(player["assists"]),
                        "xa": _as_float(player["xA"]),
                        "shots": _as_int(player["shots"]),
                        "key_passes": _as_int(player["key_passes"]),
                        "yellow_cards": _as_int(player["yellow_cards"]),
                        "red_cards": _as_int(player["red_cards"]),
                        "xg_chain": _as_float(player["xGChain"]),
                        "xg_buildup": _as_float(player["xGBuildup"]),
                    }
                )

        index = ["league", "season", "team", "player"]
        if len(stats) == 0:
            return pd.DataFrame(index=index)

        return (
            pd.DataFrame.from_records(stats)
            .set_index(index)
            .sort_index()
            .convert_dtypes()
        )

    def _select_matches(
        self, df_schedule: pd.DataFrame, match_id: Optional[Union[int, List[int]]] = None
    ) -> pd.DataFrame:
        if match_id is not None:
            match_ids = [match_id] if isinstance(match_id, int) else match_id
            df = df_schedule[df_schedule["game_id"].isin(match_ids)]
            if df.empty:
                raise ValueError("No matches found with the given IDs in the selected seasons.")
        else:
            df = df_schedule
        return df

    def read_player_match_stats(self, match_id: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:
        df_schedule = self.read_schedule(include_matches_without_data=False)
        df_results = self._select_matches(df_schedule, match_id)

        stats: List[Dict[str, Any]] = []
        for (league, season, game), league_season_game in df_results.iterrows():
            league_id = league_season_game["league_id"]
            season_id = league_season_game["season_id"]
            game_id = league_season_game["game_id"]

            data = self._read_match(game_id)
            if data is None:
                continue

            match_info = data["match_info"]
            team_id_to_name = {
                match_info[side]: _as_str(match_info[f"team_{side}"]) for side in ("h", "a")
            }

            players_data = data["rostersData"]
            for team_players in players_data.values():
                for player in team_players.values():
                    team_id = player["team_id"]
                    team = team_id_to_name[team_id]
                    stats.append(
                        {
                            "league": league,
                            "league_id": league_id,
                            "season": season,
                            "season_id": season_id,
                            "game_id": game_id,
                            "game": game,
                            "team": team,
                            "team_id": _as_int(team_id),
                            "player": _as_str(player["player"]),
                            "player_id": _as_int(player["player_id"]),
                            "position": _as_str(player["position"]),
                            "position_id": _as_int(player["positionOrder"]),
                            "minutes": _as_int(player["time"]),
                            "goals": _as_int(player["goals"]),
                            "own_goals": _as_int(player["own_goals"]),
                            "shots": _as_int(player["shots"]),
                            "xg": _as_float(player["xG"]),
                            "xg_chain": _as_float(player["xGChain"]),
                            "xg_buildup": _as_float(player["xGBuildup"]),
                            "assists": _as_int(player["assists"]),
                            "xa": _as_float(player["xA"]),
                            "key_passes": _as_int(player["key_passes"]),
                            "yellow_cards": _as_int(player["yellow_card"]),
                            "red_cards": _as_int(player["red_card"]),
                        }
                    )

        index = ["league", "season", "game", "team", "player"]
        if len(stats) == 0:
            return pd.DataFrame(index=index)

        return (
            pd.DataFrame.from_records(stats)
            .set_index(index)
            .sort_index()
            .convert_dtypes()
        )

    def read_shot_events(self, match_id: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:
        df_schedule = self.read_schedule(include_matches_without_data=False)
        df_results = self._select_matches(df_schedule, match_id)

        shots: List[Dict[str, Any]] = []
        for (league, season, game), league_season_game in df_results.iterrows():
            league_id = league_season_game["league_id"]
            season_id = league_season_game["season_id"]
            game_id = league_season_game["game_id"]

            data = self._read_match(game_id)
            if data is None:
                continue

            match_info = data["match_info"]
            team_name_to_id = {
                _as_str(match_info[f"team_{side}"]): _as_int(match_info[side])
                for side in ("h", "a")
            }

            rosters_data = data["rostersData"]
            player_name_to_id: Dict[str, Any] = {}
            for team_data in rosters_data.values():
                for player in team_data.values():
                    player_name = _as_str(player["player"])
                    player_id = _as_int(player["id"])
                    player_name_to_id[str(player_name)] = player_id

            shots_data = data["shotsData"]
            for team_shots in shots_data.values():
                for shot in team_shots:
                    team_side = shot["h_a"]
                    team = _as_str(shot[f"{team_side}_team"])
                    team_id = team_name_to_id[team]
                    assist_player = _as_str(shot["player_assisted"])
                    assist_player_id = player_name_to_id.get(str(assist_player), pd.NA)
                    shots.append(
                        {
                            "league_id": league_id,
                            "league": league,
                            "season_id": season_id,
                            "season": season,
                            "game_id": game_id,
                            "game": game,
                            "date": shot["date"],
                            "shot_id": _as_int(shot["id"]),
                            "team_id": team_id,
                            "team": team,
                            "player_id": _as_int(shot["player_id"]),
                            "player": shot["player"],
                            "assist_player_id": assist_player_id,
                            "assist_player": assist_player,
                            "xg": _as_float(shot["xG"]),
                            "location_x": _as_float(shot["X"]),
                            "location_y": _as_float(shot["Y"]),
                            "minute": _as_int(shot["minute"]),
                            "body_part": SHOT_BODY_PARTS.get(shot["shotType"], pd.NA),
                            "situation": SHOT_SITUATIONS.get(shot["situation"], pd.NA),
                            "result": SHOT_RESULTS.get(shot["result"], pd.NA),
                        }
                    )

        index = ["league", "season", "game", "team", "player"]
        if len(shots) == 0:
            return pd.DataFrame(index=index)

        return (
            pd.DataFrame.from_records(shots)
            .assign(date=lambda g: pd.to_datetime(g["date"], format="%Y-%m-%d %H:%M:%S"))
            .set_index(index)
            .sort_index()
            .convert_dtypes()
        )


def _seasons_from_league(league: str, *, no_cache: bool, proxy: Optional[str], data_dir: Path) -> List[str]:
    us = UnderstatApi(leagues=league, no_cache=no_cache, proxy=proxy, data_dir=data_dir)
    seasons = us.read_seasons()
    if isinstance(seasons, pd.DataFrame) and not seasons.empty:
        if "season" in seasons.columns:
            return seasons["season"].astype(str).unique().tolist()
        try:
            return seasons.index.get_level_values("season").unique().astype(str).tolist()
        except Exception:
            pass
    return []


def scrape_one(
    league: str,
    season: str,
    out_base: Path,
    delay: float,
    *,
    no_cache: bool = False,
    no_store: bool = False,
    force_cache: bool = False,
    max_retries: int = 6,
    backoff_base: float = 3.2,
    periodic_every: int = 8,
    periodic_secs: float = 15.0,
    proxy: Optional[str] = None,
    tables: Sequence[str] = TABLE_CHOICES,
    include_matches_without_data: bool = True,
    match_ids: Optional[List[int]] = None,
    skip_existing: bool = False,
    data_dir: Path = UNDERSTAT_DATADIR,
) -> Dict[str, List[str]]:
    log = logging.getLogger("understat")

    us = UnderstatApi(
        leagues=league,
        seasons=season,
        proxy=proxy,
        no_cache=no_cache,
        no_store=no_store,
        data_dir=data_dir,
    )

    out_dir = out_base / league / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, List[str]] = {
        "skipped_existing": [],
        "scraped_ok": [],
        "failed": [],
    }

    def _maybe_skip(path_base: Path) -> bool:
        if not skip_existing or not _csv_exists(path_base):
            return False
        existing = _load_existing_csv(path_base)
        if existing is not None and not existing.empty:
            return True
        return False

    if "schedule" in tables:
        out_path_base = out_dir / "schedule"
        if _maybe_skip(out_path_base):
            log.info("Skipping schedule for %s %s (existing CSV).", league, season)
            summary["skipped_existing"].append("schedule")
        else:
            try:
                df = _with_backoff(
                    us.read_schedule,
                    max_retries=max_retries,
                    base_delay=backoff_base,
                    kwargs={"include_matches_without_data": include_matches_without_data, "force_cache": force_cache},
                    count_as_network=not force_cache,
                    periodic_every=periodic_every,
                    periodic_secs=periodic_secs,
                )
                safe_write(df, out_path_base)
                summary["scraped_ok"].append("schedule")
            except Exception as e:
                log.warning("Failed to scrape schedule for %s %s: %s", league, season, e)
                summary["failed"].append("schedule")
            _snooze(delay)

    if "team_match" in tables:
        out_path_base = out_dir / "team_match"
        if _maybe_skip(out_path_base):
            log.info("Skipping team_match for %s %s (existing CSV).", league, season)
            summary["skipped_existing"].append("team_match")
        else:
            try:
                df = _with_backoff(
                    us.read_team_match_stats,
                    max_retries=max_retries,
                    base_delay=backoff_base,
                    kwargs={"force_cache": force_cache},
                    count_as_network=not force_cache,
                    periodic_every=periodic_every,
                    periodic_secs=periodic_secs,
                )
                safe_write(df, out_path_base)
                summary["scraped_ok"].append("team_match")
            except Exception as e:
                log.warning("Failed to scrape team_match for %s %s: %s", league, season, e)
                summary["failed"].append("team_match")
            _snooze(delay)

    if "player_season" in tables:
        out_path_base = out_dir / "player_season"
        if _maybe_skip(out_path_base):
            log.info("Skipping player_season for %s %s (existing CSV).", league, season)
            summary["skipped_existing"].append("player_season")
        else:
            try:
                df = _with_backoff(
                    us.read_player_season_stats,
                    max_retries=max_retries,
                    base_delay=backoff_base,
                    kwargs={"force_cache": force_cache},
                    count_as_network=not force_cache,
                    periodic_every=periodic_every,
                    periodic_secs=periodic_secs,
                )
                safe_write(df, out_path_base)
                summary["scraped_ok"].append("player_season")
            except Exception as e:
                log.warning("Failed to scrape player_season for %s %s: %s", league, season, e)
                summary["failed"].append("player_season")
            _snooze(delay)

    if "player_match" in tables:
        out_path_base = out_dir / "player_match"
        if _maybe_skip(out_path_base):
            log.info("Skipping player_match for %s %s (existing CSV).", league, season)
            summary["skipped_existing"].append("player_match")
        else:
            try:
                df = _with_backoff(
                    us.read_player_match_stats,
                    max_retries=max_retries,
                    base_delay=backoff_base,
                    kwargs={"match_id": match_ids} if match_ids else {},
                    count_as_network=True,
                    periodic_every=periodic_every,
                    periodic_secs=periodic_secs,
                )
                safe_write(df, out_path_base)
                summary["scraped_ok"].append("player_match")
            except Exception as e:
                log.warning("Failed to scrape player_match for %s %s: %s", league, season, e)
                summary["failed"].append("player_match")
            _snooze(delay)

    if "shot_events" in tables:
        out_path_base = out_dir / "shot_events"
        if _maybe_skip(out_path_base):
            log.info("Skipping shot_events for %s %s (existing CSV).", league, season)
            summary["skipped_existing"].append("shot_events")
        else:
            try:
                df = _with_backoff(
                    us.read_shot_events,
                    max_retries=max_retries,
                    base_delay=backoff_base,
                    kwargs={"match_id": match_ids} if match_ids else {},
                    count_as_network=True,
                    periodic_every=periodic_every,
                    periodic_secs=periodic_secs,
                )
                safe_write(df, out_path_base)
                summary["scraped_ok"].append("shot_events")
            except Exception as e:
                log.warning("Failed to scrape shot_events for %s %s: %s", league, season, e)
                summary["failed"].append("shot_events")
            _snooze(delay)

    return summary


def main() -> None:
    p = argparse.ArgumentParser("Scrape Understat data")
    p.add_argument(
        "--league",
        nargs="+",
        default=["EPL"],
        help="One or more Understat league IDs (e.g. EPL, La_liga, Serie_A).",
    )
    p.add_argument(
        "--all-known-leagues",
        action="store_true",
        help="Ignore --league and scrape all leagues from Understat API.",
    )
    p.add_argument(
        "--seasons",
        nargs="*",
        help="Specific seasons to scrape. If omitted, seasons are resolved from Understat.",
    )
    p.add_argument("--out-dir", default="data/raw/understat")
    p.add_argument("--no-cache", action="store_true", help="Bypass local API cache for this run.")
    p.add_argument("--no-store", action="store_true", help="Do not store downloaded data to cache.")
    p.add_argument(
        "--force-cache",
        action="store_true",
        help="Read only from cache where supported; never hit the network.",
    )
    p.add_argument("--delay", type=float, default=3.2, help="Base inter-request delay (seconds).")
    p.add_argument("--max-retries", type=int, default=6, help="Max 429 backoff retries per request.")
    p.add_argument("--backoff-base", type=float, default=3.2, help="Initial backoff (seconds) on HTTP 429.")
    p.add_argument("--periodic-every", type=int, default=8, help="Take a longer cool-down every N network calls.")
    p.add_argument("--periodic-secs", type=float, default=15.0, help="Cool-down length in seconds.")
    p.add_argument(
        "--tables",
        nargs="*",
        default=TABLE_CHOICES,
        choices=TABLE_CHOICES,
        help="Subset of tables to scrape.",
    )
    p.add_argument(
        "--include-matches-without-data",
        action="store_true",
        default=True,
        help="Include matches without data in schedule (default: True).",
    )
    p.add_argument(
        "--exclude-matches-without-data",
        action="store_true",
        help="Exclude matches without data in schedule.",
    )
    p.add_argument(
        "--match-ids",
        nargs="*",
        default=None,
        help="Optional match_id or comma-separated list for player_match/shot_events.",
    )
    p.add_argument(
        "--proxy",
        type=str,
        default=None,
        help='Optional proxy (e.g., "tor" or "http://user:pass@host:port").',
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="If set, skip a table if an existing CSV exists and has rows.",
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

    if args.force_cache and not args.seasons:
        raise SystemExit("In --force-cache mode, pass --seasons to avoid network lookups.")

    if args.force_cache and any(t in args.tables for t in ("player_match", "shot_events")):
        raise SystemExit("--force-cache cannot be used with player_match or shot_events.")

    if args.exclude_matches_without_data:
        args.include_matches_without_data = False

    init_logger(args.verbose)
    log = logging.getLogger("understat")

    if args.all_known_leagues:
        leagues = UnderstatApi.available_leagues(
            no_cache=args.no_cache, no_store=args.no_store, proxy=args.proxy, data_dir=UNDERSTAT_DATADIR
        )
    else:
        leagues = args.league

    match_ids = _parse_match_ids(args.match_ids)
    out_base = Path(args.out_dir)
    meta_path = Path(args.meta_path)

    log.info(
        "Understat scrape configuration: leagues=%s seasons=%s tables=%s no_cache=%s "
        "no_store=%s force_cache=%s out_dir=%s skip_existing=%s match_ids=%s proxy=%s",
        leagues,
        args.seasons if args.seasons else "auto (read_seasons)",
        args.tables,
        args.no_cache,
        args.no_store,
        args.force_cache,
        args.out_dir,
        args.skip_existing,
        match_ids,
        args.proxy,
    )

    for league in leagues:
        seasons = args.seasons or _seasons_from_league(
            league, no_cache=args.no_cache, proxy=args.proxy, data_dir=UNDERSTAT_DATADIR
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
                include_matches_without_data=args.include_matches_without_data,
                match_ids=match_ids,
                skip_existing=args.skip_existing,
                data_dir=UNDERSTAT_DATADIR,
            )

            job = ScrapeJobId(
                scraper="understat",
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
        "Understat scrape completed for leagues=%s. Total approximate network calls: %d",
        leagues,
        _GLOBAL["net_calls"],
    )


if __name__ == "__main__":
    main()
