#!/usr/bin/env python3
# scripts/fbref_pipeline/scrape/whoscored_match_stats_scraper.py
"""
WhoScored match-level scraper.

Backends:
- native (default): direct WhoScored HTTP/browser fallback, no soccerdata required
- soccerdata: legacy compatibility path

Tables:
- schedule
- missing_players
- events (default), raw, spadl, atomic-spadl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import time
import warnings
from collections.abc import Iterable as IterableABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from fpl_assistant.platform.paths import CONFIG_ROOT

os.environ.setdefault("SOCCERDATA_LOGLEVEL", "ERROR")
os.environ.setdefault("SOCCERDATA_DIR", str(Path("data/_soccerdata_cache").absolute()))

try:
    import soccerdata as sd
except ImportError:  # pragma: no cover - optional dependency during tests
    sd = None  # type: ignore[assignment]

from scripts.fbref_pipeline.automation.auto_scrape import ScrapeJobId, record_last_run
from scripts.fbref_pipeline.scrape.whoscored_native_backend import (
    NativeWhoScoredBackend,
    extract_report_json_blobs,
    load_native_competitions,
)

if sd is not None:
    warnings.filterwarnings("ignore", category=FutureWarning, module=r".*soccerdata.*")

REPO_ROOT = CONFIG_ROOT.parent
CONFIG_DIR = CONFIG_ROOT
REPO_LEAGUE_DICT_PATH = CONFIG_DIR / "league_dict.json"
REPO_NATIVE_COMPETITIONS_PATH = CONFIG_DIR / "whoscored_competitions.json"
REPO_COMPETITION_GROUPS_PATH = CONFIG_DIR / "whoscored_competition_groups.json"

ALL_KNOWN_LEAGUES: List[str] = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]

TABLE_CHOICES = ["schedule", "missing_players", "events"]
EVENT_FORMATS = ["events", "raw", "spadl", "atomic-spadl", "none"]
BACKEND_CHOICES = ["native", "soccerdata"]
STATS_MODE_CHOICES = ["none", "core", "all-visible"]
DERIVED_TABLE_CHOICES = [
    "match_info",
    "incidents",
    "player_dictionary",
    "lineups",
    "formations",
]

DEFAULT_LEAGUE_DICT: Dict[str, Dict[str, str]] = {
    "ENG-Premier League": {"WhoScored": "England - Premier League"},
    "ESP-La Liga": {"WhoScored": "Spain - LaLiga"},
    "ITA-Serie A": {"WhoScored": "Italy - Serie A"},
    "GER-Bundesliga": {"WhoScored": "Germany - Bundesliga"},
    "FRA-Ligue 1": {"WhoScored": "France - Ligue 1"},
    "INT-UEFA Champions League": {"WhoScored": "INT-UEFA Champions League"},
    "INT-UEFA Europa League": {"WhoScored": "INT-UEFA Europa League"},
    "INT-UEFA Europa Conference League": {"WhoScored": "INT-UEFA Europa Conference League"},
    "INT-UEFA Super Cup": {"WhoScored": "INT-UEFA Super Cup"},
    "INT-World Cup": {"WhoScored": "International - FIFA World Cup"},
    "INT-European Championship": {"WhoScored": "International - European Championship"},
    "INT-UEFA Nations League": {"WhoScored": "INT-UEFA Nations League"},
    "INT-Copa America": {"WhoScored": "INT-Copa America"},
    "INT-Africa Cup of Nations": {"WhoScored": "INT-Africa Cup of Nations"},
    "INT-World Cup qualification": {"WhoScored": "INT-World Cup qualification"},
    "INT-European Championship qualification": {
        "WhoScored": "INT-European Championship qualification"
    },
}

DEFAULT_COMPETITION_GROUPS: Dict[str, List[str]] = {
    "top5-domestic": ALL_KNOWN_LEAGUES,
    "uefa-club": [
        "INT-UEFA Champions League",
        "INT-UEFA Europa League",
        "INT-UEFA Europa Conference League",
        "INT-UEFA Super Cup",
    ],
    "major-international": [
        "INT-World Cup",
        "INT-European Championship",
        "INT-UEFA Nations League",
        "INT-Copa America",
        "INT-Africa Cup of Nations",
    ],
    "major-international-qualifiers": [
        "INT-World Cup qualification",
        "INT-European Championship qualification",
    ],
}

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

COLS_MATCH_INFO = [
    "league",
    "season",
    "game",
    "game_id",
    "date",
    "start_time",
    "status",
    "stage",
    "venue",
    "referee",
    "attendance",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_score",
    "away_score",
    "score",
    "home_formation",
    "away_formation",
    "has_events",
]

COLS_INCIDENTS = [
    "league",
    "season",
    "game",
    "game_id",
    "source",
    "incident_index",
    "minute",
    "type",
    "sub_type",
    "team_id",
    "team",
    "player_id",
    "player",
    "text",
    "payload_json",
]

COLS_PLAYER_DICTIONARY = [
    "league",
    "season",
    "game",
    "game_id",
    "player_id",
    "player",
    "team_id",
    "team",
    "side",
    "lineup_status",
    "position",
    "shirt_no",
]

COLS_LINEUPS = [
    "league",
    "season",
    "game",
    "game_id",
    "team_id",
    "team",
    "side",
    "player_id",
    "player",
    "lineup_status",
    "position",
    "shirt_no",
    "is_first_eleven",
]

COLS_FORMATIONS = [
    "league",
    "season",
    "game",
    "game_id",
    "team_id",
    "team",
    "side",
    "source",
    "event_id",
    "minute",
    "expanded_minute",
    "period",
    "formation",
]

COLS_TEAM_MATCH_STATS = [
    "league",
    "season",
    "game",
    "game_id",
    "team_id",
    "team",
    "stat_group",
    "stat_name",
    "stat_key",
    "value",
    "value_text",
    "source_tab",
]

COLS_PLAYER_MATCH_STATS = [
    "league",
    "season",
    "game",
    "game_id",
    "team_id",
    "team",
    "player_id",
    "player",
    "stat_group",
    "stat_name",
    "stat_key",
    "value",
    "value_text",
    "source_tab",
]

COLS_MATCH_FACT_STATS = [
    "league",
    "season",
    "game",
    "game_id",
    "fact_group",
    "fact_name",
    "fact_key",
    "value",
    "value_text",
    "source_tab",
]


def init_logger(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def polite_sleep(delay: float) -> None:
    if delay > 0:
        time.sleep(delay)


def safe_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=True)


def _require_soccerdata() -> None:
    if sd is None:
        raise RuntimeError(
            "soccerdata is required to scrape WhoScored data. "
            "Install it in the active Python environment to run this scraper."
        )


def _camel_to_snake(name: str) -> str:
    name = name.replace(".", "_")
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.replace("-", "_").lower()


def _display_name(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "displayName" in value:
            return value.get("displayName")
        if "display_name" in value:
            return value.get("display_name")
    return value


def _safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _schema_df(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _default_league_dict() -> Dict[str, Dict[str, str]]:
    return json.loads(json.dumps(DEFAULT_LEAGUE_DICT))


def _default_competition_groups() -> Dict[str, List[str]]:
    return json.loads(json.dumps(DEFAULT_COMPETITION_GROUPS))


def _load_native_competitions() -> Dict[str, Dict[str, Any]]:
    data = _load_json(REPO_NATIVE_COMPETITIONS_PATH, {})
    if not isinstance(data, dict):
        data = {}
    if not data and REPO_LEAGUE_DICT_PATH.is_file():
        legacy = _load_json(REPO_LEAGUE_DICT_PATH, _default_league_dict())
        adapted: Dict[str, Dict[str, Any]] = {}
        for key, value in legacy.items():
            if isinstance(value, Mapping):
                adapted[str(key)] = {
                    "source_name": str(value.get("WhoScored") or key),
                    "competition_type": "international" if str(key).startswith("INT-") else "club",
                    "season_mode": "single-year" if str(key).startswith("INT-") else "split-year",
                }
        return adapted
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            out[str(key)] = {str(k): v for k, v in value.items()}
    return out


def _load_repo_league_dict() -> Dict[str, Dict[str, str]]:
    data = _load_json(REPO_LEAGUE_DICT_PATH, _default_league_dict())
    if not isinstance(data, dict):
        return _default_league_dict()
    out: Dict[str, Dict[str, str]] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            out[str(key)] = {str(k): str(v) for k, v in value.items()}
    return out or _default_league_dict()


def _load_competition_groups() -> Dict[str, List[str]]:
    data = _load_json(REPO_COMPETITION_GROUPS_PATH, _default_competition_groups())
    if not isinstance(data, dict):
        return _default_competition_groups()
    out: Dict[str, List[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            out[str(key)] = [str(x) for x in value]
    return out or _default_competition_groups()


def _soccerdata_config_dir() -> Path:
    return Path(os.environ["SOCCERDATA_DIR"]).resolve() / "config"


def sync_repo_league_dict() -> Path:
    target = _soccerdata_config_dir() / "league_dict.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if REPO_LEAGUE_DICT_PATH.is_file():
        shutil.copyfile(REPO_LEAGUE_DICT_PATH, target)
    else:
        _save_json(target, _default_league_dict())
    return target


def _configured_competitions() -> List[str]:
    return sorted(_load_native_competitions().keys())


def resolve_competitions(
    *,
    explicit_leagues: Optional[Sequence[str]],
    competition_groups: Optional[Sequence[str]],
    all_known_leagues: bool,
    all_configured_competitions: bool,
) -> List[str]:
    groups = _load_competition_groups()
    configured = set(_configured_competitions())
    resolved: List[str] = []

    if explicit_leagues:
        resolved.extend(explicit_leagues)
    if all_known_leagues:
        resolved.extend(groups.get("top5-domestic", ALL_KNOWN_LEAGUES))
    if all_configured_competitions:
        resolved.extend(sorted(configured))
    for group in competition_groups or []:
        if group not in groups:
            raise ValueError(
                f"Unknown competition group {group!r}. Available groups: {sorted(groups)}"
            )
        resolved.extend(groups[group])

    deduped: List[str] = []
    seen: set[str] = set()
    for league in resolved:
        if league not in configured:
            raise ValueError(
                f"Unknown competition {league!r}. Add it to {REPO_NATIVE_COMPETITIONS_PATH} "
                "or use --list-competitions."
            )
        if league not in seen:
            deduped.append(league)
            seen.add(league)
    return deduped


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
    return _schema_df(
        [
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
    )


def _schema_only_missing_players() -> pd.DataFrame:
    return _schema_df(
        [
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
    )


def _schema_only_events() -> pd.DataFrame:
    return _schema_df(["game"] + COLS_EVENTS)


def _schema_only_match_info() -> pd.DataFrame:
    return _schema_df(COLS_MATCH_INFO)


def _schema_only_incidents() -> pd.DataFrame:
    return _schema_df(COLS_INCIDENTS)


def _schema_only_player_dictionary() -> pd.DataFrame:
    return _schema_df(COLS_PLAYER_DICTIONARY)


def _schema_only_lineups() -> pd.DataFrame:
    return _schema_df(COLS_LINEUPS)


def _schema_only_formations() -> pd.DataFrame:
    return _schema_df(COLS_FORMATIONS)


def _schema_only_team_match_stats() -> pd.DataFrame:
    return _schema_df(COLS_TEAM_MATCH_STATS)


def _schema_only_player_match_stats() -> pd.DataFrame:
    return _schema_df(COLS_PLAYER_MATCH_STATS)


def _schema_only_match_fact_stats() -> pd.DataFrame:
    return _schema_df(COLS_MATCH_FACT_STATS)


def _build_ws(
    league: Union[str, Sequence[str]],
    seasons: Optional[Union[int, Sequence[int], Sequence[str], str]] = None,
    *,
    proxy: Optional[Union[str, dict, list]] = None,
    no_cache: bool = False,
    no_store: bool = False,
    path_to_browser: Optional[str] = None,
    headless: bool = True,
) -> Any:
    _require_soccerdata()
    kwargs: Dict[str, Any] = dict(
        leagues=league,
        proxy=proxy,
        no_cache=no_cache,
        no_store=no_store,
        headless=headless,
    )
    if seasons is not None:
        kwargs["seasons"] = seasons
    if path_to_browser:
        kwargs["path_to_browser"] = path_to_browser
    return sd.WhoScored(**kwargs)  # type: ignore[union-attr]


def _close_ws(ws: Any) -> None:
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


def _read_existing_csv(base: Path) -> pd.DataFrame:
    path = base if base.suffix else base.with_suffix(".csv")
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _resolve_seasons_for_league(
    *,
    league: str,
    explicit_seasons: Optional[Sequence[str]],
    proxy: Optional[Union[str, dict, list]],
    no_cache: bool,
    no_store: bool,
    path_to_browser: Optional[str],
    headless: bool,
) -> List[Union[str, int]]:
    if explicit_seasons:
        return list(explicit_seasons)

    ws = _build_ws(
        league,
        None,
        proxy=proxy,
        no_cache=no_cache,
        no_store=no_store,
        path_to_browser=path_to_browser,
        headless=headless,
    )
    try:
        seasons_df = ws.read_seasons()
        if isinstance(seasons_df, pd.DataFrame) and not seasons_df.empty:
            values = seasons_df.index.get_level_values("season").unique().tolist()
            return [str(v) for v in values]
    finally:
        _close_ws(ws)
    raise RuntimeError(f"Could not resolve seasons for {league}")


def _cached_read_leagues(ws: Any) -> pd.DataFrame:
    filepath = Path(getattr(ws, "data_dir")) / "tiers.json"
    if not filepath.is_file():
        reader = ws.get("https://www.whoscored.com", filepath, var="allRegions")
        data = json.load(reader)
    else:
        data = json.loads(filepath.read_text(encoding="utf-8"))

    leagues: List[Dict[str, Any]] = []
    for region in data:
        for league in region.get("tournaments", []):
            leagues.append(
                {
                    "region_id": region.get("id"),
                    "region": region.get("name"),
                    "league_id": league.get("id"),
                    "league": league.get("name"),
                }
            )

    df = pd.DataFrame(leagues)
    if df.empty:
        raise RuntimeError("WhoScored tiers cache is empty or invalid.")

    df = (
        df.assign(league=lambda x: x.region.astype(str) + " - " + x.league.astype(str))
        .pipe(ws._translate_league)
        .set_index("league")
    )
    selected = list(ws._selected_leagues.keys())
    missing = [league for league in selected if league not in df.index]
    if missing:
        raise RuntimeError(
            f"Selected leagues not found in cached WhoScored tiers: {missing}. "
            "Refresh the tiers cache or adjust the configured league mapping."
        )
    return df.loc[selected].sort_index()


def _attach_cached_read_leagues(ws: Any) -> None:
    def _read_leagues_cached() -> pd.DataFrame:
        return _cached_read_leagues(ws)

    try:
        ws.read_leagues = _read_leagues_cached  # type: ignore[assignment]
    except Exception:
        pass


def _resolve_seasons_frame_for_league(
    *,
    league: str,
    explicit_seasons: Optional[Sequence[str]],
    proxy: Optional[Union[str, dict, list]],
    no_cache: bool,
    no_store: bool,
    path_to_browser: Optional[str],
    headless: bool,
) -> pd.DataFrame:
    ws = _build_ws(
        league,
        None,
        proxy=proxy,
        no_cache=no_cache,
        no_store=no_store,
        path_to_browser=path_to_browser,
        headless=headless,
    )
    try:
        _attach_cached_read_leagues(ws)
        seasons_df = ws.read_seasons()
        if not isinstance(seasons_df, pd.DataFrame) or seasons_df.empty:
            raise RuntimeError(f"Could not resolve seasons for {league}")
        if explicit_seasons:
            wanted = {str(_to_ws_season_int(season)) for season in explicit_seasons}
            idx = seasons_df.index.get_level_values("season").astype(str)
            seasons_df = seasons_df[idx.isin(wanted)]
        return seasons_df.copy()
    finally:
        _close_ws(ws)


def _event_cache_dir(ws: Any, league: str, season_int: int) -> Path:
    data_dir = Path(getattr(ws, "data_dir", _soccerdata_config_dir().parent / "WhoScored"))
    return data_dir / "events" / f"{league}_{season_int}"


def _load_cached_raw_payloads(
    *,
    ws: Any,
    league: str,
    season_int: int,
    match_ids: Optional[Sequence[int]],
) -> Dict[int, Dict[str, Any]]:
    cache_dir = _event_cache_dir(ws, league, season_int)
    payloads: Dict[int, Dict[str, Any]] = {}
    if not cache_dir.is_dir():
        return payloads

    allowed = {int(x) for x in match_ids} if match_ids else None
    for path in sorted(cache_dir.glob("*.json")):
        game_id = _coerce_int(path.stem)
        if game_id is None:
            continue
        if allowed is not None and game_id not in allowed:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, Mapping):
            payloads[game_id] = dict(payload)
    return payloads


def _archive_raw_payloads(raw_dir: Path, payloads: Mapping[int, Mapping[str, Any]]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for game_id, payload in payloads.items():
        _save_json(raw_dir / f"{game_id}.json", payload)


def _build_schedule_lookup(schedule_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    if schedule_df.empty:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for _, row in schedule_df.iterrows():
        game_id = _coerce_int(row.get("game_id"))
        if game_id is not None:
            out[game_id] = row.to_dict()
    return out


def _home_away_team_map(payload: Mapping[str, Any]) -> Dict[int, Tuple[str, str]]:
    mapping: Dict[int, Tuple[str, str]] = {}
    for side in ("home", "away"):
        team_obj = payload.get(side)
        if isinstance(team_obj, Mapping):
            team_id = _coerce_int(team_obj.get("teamId") or team_obj.get("id"))
            team_name = str(team_obj.get("name") or team_obj.get("teamName") or "")
            if team_id is not None:
                mapping[team_id] = (team_name, side)
    return mapping


def _iter_player_containers(
    team_obj: Mapping[str, Any], status_hint: str
) -> Iterable[Tuple[Mapping[str, Any], str]]:
    candidate_keys = [
        ("players", status_hint),
        ("lineup", "starter"),
        ("lineUp", "starter"),
        ("subs", "bench"),
        ("substitutes", "bench"),
        ("bench", "bench"),
        ("squad", "squad"),
    ]
    for key, status in candidate_keys:
        value = team_obj.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    yield item, status
        elif isinstance(value, Mapping):
            nested_list = value.get("players")
            if isinstance(nested_list, list):
                for item in nested_list:
                    if isinstance(item, Mapping):
                        yield item, status


def _lineup_status(raw: Mapping[str, Any], hint: str) -> str:
    if raw.get("isFirstEleven") is True:
        return "starter"
    if raw.get("isSub") is True or raw.get("isSubstitute") is True:
        return "bench"
    position = str(raw.get("position") or raw.get("positionText") or "").lower()
    if position == "sub":
        return "bench"
    return hint


def _extract_formation_from_qualifiers(qualifiers: Any) -> Optional[str]:
    if not isinstance(qualifiers, list):
        return None
    for qualifier in qualifiers:
        if not isinstance(qualifier, Mapping):
            continue
        name = _display_name(qualifier.get("type"))
        if name == "TeamPlayerFormation":
            value = qualifier.get("value")
            if value:
                return str(value)
    return None


def _extract_lineups_rows(
    *,
    payload: Mapping[str, Any],
    league: str,
    season_int: int,
    schedule_row: Optional[Mapping[str, Any]],
    game_id: int,
) -> List[Dict[str, Any]]:
    game_label = str(schedule_row.get("game")) if schedule_row else ""
    seen: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for side in ("home", "away"):
        team_obj = payload.get(side)
        if not isinstance(team_obj, Mapping):
            continue
        team_id = _coerce_int(team_obj.get("teamId") or team_obj.get("id"))
        team_name = str(team_obj.get("name") or team_obj.get("teamName") or "")
        for player_obj, hint in _iter_player_containers(team_obj, "squad"):
            player_id = _coerce_int(
                player_obj.get("playerId") or player_obj.get("id") or player_obj.get("personId")
            )
            if player_id is None:
                continue
            row = {
                "league": league,
                "season": season_int,
                "game": game_label,
                "game_id": game_id,
                "team_id": team_id,
                "team": team_name,
                "side": side,
                "player_id": player_id,
                "player": player_obj.get("name") or player_obj.get("playerName"),
                "lineup_status": _lineup_status(player_obj, hint),
                "position": player_obj.get("position") or player_obj.get("positionText"),
                "shirt_no": player_obj.get("shirtNo") or player_obj.get("shirtNumber"),
                "is_first_eleven": bool(player_obj.get("isFirstEleven")),
            }
            key = (team_id or -1, player_id)
            existing = seen.get(key)
            if existing is None:
                seen[key] = row
            else:
                rank = {"starter": 3, "bench": 2, "squad": 1}.get(row["lineup_status"], 0)
                current_rank = {"starter": 3, "bench": 2, "squad": 1}.get(
                    existing["lineup_status"], 0
                )
                if rank > current_rank:
                    seen[key] = row
    return list(seen.values())


def _extract_player_dictionary_rows(
    *,
    payload: Mapping[str, Any],
    league: str,
    season_int: int,
    schedule_row: Optional[Mapping[str, Any]],
    game_id: int,
) -> List[Dict[str, Any]]:
    lineups = _extract_lineups_rows(
        payload=payload,
        league=league,
        season_int=season_int,
        schedule_row=schedule_row,
        game_id=game_id,
    )
    by_player: Dict[int, Dict[str, Any]] = {
        int(row["player_id"]): {
            "league": league,
            "season": season_int,
            "game": row["game"],
            "game_id": game_id,
            "player_id": row["player_id"],
            "player": row.get("player"),
            "team_id": row.get("team_id"),
            "team": row.get("team"),
            "side": row.get("side"),
            "lineup_status": row.get("lineup_status"),
            "position": row.get("position"),
            "shirt_no": row.get("shirt_no"),
        }
        for row in lineups
        if row.get("player_id") is not None
    }

    player_dict = payload.get("playerIdNameDictionary")
    if isinstance(player_dict, Mapping):
        for raw_pid, player_name in player_dict.items():
            player_id = _coerce_int(raw_pid)
            if player_id is None:
                continue
            row = by_player.get(player_id)
            if row is None:
                by_player[player_id] = {
                    "league": league,
                    "season": season_int,
                    "game": str(schedule_row.get("game")) if schedule_row else "",
                    "game_id": game_id,
                    "player_id": player_id,
                    "player": player_name,
                    "team_id": None,
                    "team": None,
                    "side": None,
                    "lineup_status": None,
                    "position": None,
                    "shirt_no": None,
                }
            elif not row.get("player") and player_name:
                row["player"] = player_name
    return list(by_player.values())


def _extract_formations_rows(
    *,
    payload: Mapping[str, Any],
    league: str,
    season_int: int,
    schedule_row: Optional[Mapping[str, Any]],
    game_id: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    game_label = str(schedule_row.get("game")) if schedule_row else ""
    team_map = _home_away_team_map(payload)

    for side in ("home", "away"):
        team_obj = payload.get(side)
        if not isinstance(team_obj, Mapping):
            continue
        formation = team_obj.get("formation") or team_obj.get("teamFormation")
        if formation:
            rows.append(
                {
                    "league": league,
                    "season": season_int,
                    "game": game_label,
                    "game_id": game_id,
                    "team_id": _coerce_int(team_obj.get("teamId") or team_obj.get("id")),
                    "team": team_obj.get("name") or team_obj.get("teamName"),
                    "side": side,
                    "source": "payload_team",
                    "event_id": None,
                    "minute": 0,
                    "expanded_minute": 0,
                    "period": "PreMatch",
                    "formation": str(formation),
                }
            )

    events = payload.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, Mapping):
                continue
            event_type = _display_name(event.get("type"))
            formation = _extract_formation_from_qualifiers(event.get("qualifiers"))
            if event_type != "FormationSet" and not formation:
                continue
            team_id = _coerce_int(event.get("teamId"))
            team_name, side = team_map.get(team_id or -1, ("", ""))
            rows.append(
                {
                    "league": league,
                    "season": season_int,
                    "game": game_label,
                    "game_id": game_id,
                    "team_id": team_id,
                    "team": team_name,
                    "side": side,
                    "source": "raw_event",
                    "event_id": _coerce_int(event.get("id")),
                    "minute": event.get("minute"),
                    "expanded_minute": event.get("expandedMinute"),
                    "period": _display_name(event.get("period")),
                    "formation": formation,
                }
            )
    return rows


def _extract_incidents_rows(
    *,
    payload: Mapping[str, Any],
    schedule_row: Optional[Mapping[str, Any]],
    league: str,
    season_int: int,
    game_id: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    game_label = str(schedule_row.get("game")) if schedule_row else ""
    team_map = _home_away_team_map(payload)
    sources: List[Tuple[str, Any]] = []

    if schedule_row and schedule_row.get("incidents") is not None:
        sources.append(("schedule", schedule_row.get("incidents")))
    for key in ("incidents", "incidentEvents", "matchIncidents"):
        if payload.get(key) is not None:
            sources.append(("raw", payload.get(key)))

    for source, incidents in sources:
        if not isinstance(incidents, list):
            continue
        for idx, incident in enumerate(incidents):
            if not isinstance(incident, Mapping):
                continue
            player_obj = incident.get("player") if isinstance(incident.get("player"), Mapping) else None
            team_id = _coerce_int(incident.get("teamId"))
            team_name, _side = team_map.get(team_id or -1, ("", ""))
            rows.append(
                {
                    "league": league,
                    "season": season_int,
                    "game": game_label,
                    "game_id": game_id,
                    "source": source,
                    "incident_index": idx,
                    "minute": incident.get("minute"),
                    "type": _display_name(incident.get("type")),
                    "sub_type": _display_name(incident.get("subType") or incident.get("sub_type")),
                    "team_id": team_id,
                    "team": team_name or incident.get("teamName"),
                    "player_id": _coerce_int(
                        incident.get("playerId") or (player_obj or {}).get("playerId")
                    ),
                    "player": incident.get("playerName") or (player_obj or {}).get("name"),
                    "text": incident.get("text") or incident.get("comment"),
                    "payload_json": _safe_json_dumps(incident),
                }
            )
    return rows


def _normalize_events_from_payload(
    *,
    payload: Mapping[str, Any],
    schedule_row: Optional[Mapping[str, Any]],
    league: str,
    season_int: int,
    game_id: int,
) -> pd.DataFrame:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        return _normalize(_schema_only_events(), league, season_int)

    player_names: Dict[int, str] = {}
    player_dict = payload.get("playerIdNameDictionary")
    if isinstance(player_dict, Mapping):
        for key, value in player_dict.items():
            pid = _coerce_int(key)
            if pid is not None and value is not None:
                player_names[pid] = str(value)

    team_names = {
        team_id: team_name for team_id, (team_name, _side) in _home_away_team_map(payload).items()
    }
    game_label = str(schedule_row.get("game")) if schedule_row else ""
    rows: List[Dict[str, Any]] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            continue
        row = {_camel_to_snake(str(k)): v for k, v in raw_event.items()}
        player_id = _coerce_int(raw_event.get("playerId"))
        team_id = _coerce_int(raw_event.get("teamId"))
        row["game"] = game_label
        row["game_id"] = game_id
        row["period"] = _display_name(raw_event.get("period"))
        row["type"] = _display_name(raw_event.get("type"))
        row["outcome_type"] = _display_name(raw_event.get("outcomeType"))
        row["card_type"] = _display_name(raw_event.get("cardType"))
        row["player_id"] = player_id
        row["team_id"] = team_id
        row["player"] = player_names.get(player_id or -1)
        row["team"] = team_names.get(team_id or -1)
        row["related_event_id"] = _coerce_int(raw_event.get("relatedEventId"))
        row["related_player_id"] = _coerce_int(raw_event.get("relatedPlayerId"))
        row["expanded_minute"] = raw_event.get("expandedMinute", raw_event.get("expanded_minute"))
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in COLS_EVENTS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    preferred = ["game"] + COLS_EVENTS
    trailing = [col for col in df.columns if col not in preferred]
    return _normalize(df[preferred + trailing], league, season_int)


def _extract_match_info_row(
    *,
    payload: Mapping[str, Any],
    schedule_row: Optional[Mapping[str, Any]],
    league: str,
    season_int: int,
    game_id: int,
) -> Dict[str, Any]:
    home = payload.get("home") if isinstance(payload.get("home"), Mapping) else {}
    away = payload.get("away") if isinstance(payload.get("away"), Mapping) else {}
    return {
        "league": league,
        "season": season_int,
        "game": schedule_row.get("game") if schedule_row else "",
        "game_id": game_id,
        "date": (schedule_row or {}).get("date") or payload.get("startDate"),
        "start_time": (schedule_row or {}).get("start_time") or payload.get("startTime"),
        "status": (schedule_row or {}).get("status") or payload.get("matchStatus"),
        "stage": (schedule_row or {}).get("stage"),
        "venue": payload.get("venueName") or payload.get("venue"),
        "referee": payload.get("refereeName") or payload.get("referee"),
        "attendance": payload.get("attendance"),
        "home_team_id": (schedule_row or {}).get("home_team_id") or _coerce_int(home.get("teamId")),
        "home_team": (schedule_row or {}).get("home_team") or home.get("name"),
        "away_team_id": (schedule_row or {}).get("away_team_id") or _coerce_int(away.get("teamId")),
        "away_team": (schedule_row or {}).get("away_team") or away.get("name"),
        "home_score": (schedule_row or {}).get("home_score") or payload.get("homeScore"),
        "away_score": (schedule_row or {}).get("away_score") or payload.get("awayScore"),
        "score": payload.get("score"),
        "home_formation": home.get("formation") or home.get("teamFormation"),
        "away_formation": away.get("formation") or away.get("teamFormation"),
        "has_events": bool(payload.get("events")),
    }


def build_derived_tables(
    *,
    payloads: Mapping[int, Mapping[str, Any]],
    schedule_df: pd.DataFrame,
    league: str,
    season_int: int,
    derived_tables: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    schedule_lookup = _build_schedule_lookup(schedule_df)
    outputs: Dict[str, pd.DataFrame] = {
        "match_info": _schema_only_match_info(),
        "incidents": _schema_only_incidents(),
        "player_dictionary": _schema_only_player_dictionary(),
        "lineups": _schema_only_lineups(),
        "formations": _schema_only_formations(),
    }
    if not payloads:
        return {k: outputs[k] for k in derived_tables}

    match_info_rows: List[Dict[str, Any]] = []
    incidents_rows: List[Dict[str, Any]] = []
    player_rows: List[Dict[str, Any]] = []
    lineup_rows: List[Dict[str, Any]] = []
    formation_rows: List[Dict[str, Any]] = []
    for game_id, payload in payloads.items():
        schedule_row = schedule_lookup.get(game_id)
        match_info_rows.append(
            _extract_match_info_row(
                payload=payload,
                schedule_row=schedule_row,
                league=league,
                season_int=season_int,
                game_id=game_id,
            )
        )
        incidents_rows.extend(
            _extract_incidents_rows(
                payload=payload,
                schedule_row=schedule_row,
                league=league,
                season_int=season_int,
                game_id=game_id,
            )
        )
        lineup_rows.extend(
            _extract_lineups_rows(
                payload=payload,
                schedule_row=schedule_row,
                league=league,
                season_int=season_int,
                game_id=game_id,
            )
        )
        player_rows.extend(
            _extract_player_dictionary_rows(
                payload=payload,
                schedule_row=schedule_row,
                league=league,
                season_int=season_int,
                game_id=game_id,
            )
        )
        formation_rows.extend(
            _extract_formations_rows(
                payload=payload,
                schedule_row=schedule_row,
                league=league,
                season_int=season_int,
                game_id=game_id,
            )
        )

    outputs["match_info"] = pd.DataFrame(match_info_rows, columns=COLS_MATCH_INFO)
    outputs["incidents"] = pd.DataFrame(incidents_rows, columns=COLS_INCIDENTS)
    outputs["player_dictionary"] = pd.DataFrame(player_rows, columns=COLS_PLAYER_DICTIONARY)
    outputs["lineups"] = pd.DataFrame(lineup_rows, columns=COLS_LINEUPS)
    outputs["formations"] = pd.DataFrame(formation_rows, columns=COLS_FORMATIONS)
    return {k: outputs[k] for k in derived_tables}


def _write_derived_tables(derived_dir: Path, tables: Mapping[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        safe_write(df, derived_dir / name)


def _iter_scalar_stats(
    value: Any,
    *,
    prefix: Tuple[str, ...] = (),
) -> Iterable[Tuple[str, str, Any, Any]]:
    if isinstance(value, Mapping):
        if prefix and value and all(_is_numeric_like(key) for key in value):
            latest_key = max(value.keys(), key=_numeric_key_sort_value)
            latest_value = value[latest_key]
            if isinstance(latest_value, Mapping):
                yield from _iter_scalar_stats(latest_value, prefix=prefix)
                return
            if isinstance(latest_value, list):
                if latest_value and all(
                    not isinstance(item, (Mapping, list, tuple, set)) for item in latest_value
                ):
                    stat_key = ".".join(prefix)
                    stat_name = prefix[-1] if prefix else "value"
                    yield stat_key, stat_name, None, ",".join(str(item) for item in latest_value)
                return
            stat_key = ".".join(prefix)
            stat_name = prefix[-1] if prefix else "value"
            yield stat_key, stat_name, latest_value, latest_value
            return
        for key, child in value.items():
            child_key = _camel_to_snake(str(key))
            yield from _iter_scalar_stats(child, prefix=prefix + (child_key,))
        return
    if isinstance(value, list):
        if value and all(not isinstance(item, (Mapping, list, tuple, set)) for item in value):
            stat_key = ".".join(prefix)
            stat_name = prefix[-1] if prefix else "value"
            yield stat_key, stat_name, None, ",".join(str(item) for item in value)
        return
    if not prefix:
        return
    stat_key = ".".join(prefix)
    stat_name = prefix[-1]
    yield stat_key, stat_name, _coerce_int(value) if isinstance(value, bool) else value, value


def _is_numeric_like(value: Any) -> bool:
    text = str(value).strip()
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", text))


def _numeric_key_sort_value(value: Any) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float("-inf")


def _collect_stats_from_mapping(
    mapping: Mapping[str, Any],
    *,
    default_group: str,
    include_keys: Optional[Sequence[str]] = None,
    exclude_keys: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    include_set = {str(key) for key in include_keys or []}
    exclude_set = {str(key) for key in exclude_keys or []}
    for key, value in mapping.items():
        raw_key = str(key)
        if raw_key in exclude_set:
            continue
        if include_set and raw_key not in include_set:
            continue
        stat_group = _camel_to_snake(raw_key)
        if isinstance(value, Mapping):
            for stat_key, stat_name, numeric_value, raw_value in _iter_scalar_stats(
                value,
                prefix=(stat_group,),
            ):
                rows.append(
                    {
                        "stat_group": stat_group,
                        "stat_name": stat_name,
                        "stat_key": stat_key,
                        "value": numeric_value,
                        "value_text": None if raw_value is None else str(raw_value),
                    }
                )
        elif isinstance(value, list):
            continue
        else:
            rows.append(
                {
                    "stat_group": default_group,
                    "stat_name": _camel_to_snake(raw_key),
                    "stat_key": _camel_to_snake(raw_key),
                    "value": value,
                    "value_text": None if value is None else str(value),
                }
            )
    return rows


def build_match_stats_tables(
    *,
    payloads: Mapping[int, Mapping[str, Any]],
    report_blobs: Optional[Mapping[int, Mapping[str, Any]]],
    schedule_df: pd.DataFrame,
    league: str,
    season_int: int,
    stats_mode: str,
) -> Dict[str, pd.DataFrame]:
    schedule_lookup = _build_schedule_lookup(schedule_df)
    if stats_mode == "none":
        return {
            "team_match_stats": _schema_only_team_match_stats(),
            "player_match_stats": _schema_only_player_match_stats(),
            "match_fact_stats": _schema_only_match_fact_stats(),
        }

    team_rows: List[Dict[str, Any]] = []
    player_rows: List[Dict[str, Any]] = []
    fact_rows: List[Dict[str, Any]] = []
    report_blobs = report_blobs or {}
    team_exclude = {
        "teamId",
        "id",
        "name",
        "teamName",
        "players",
        "lineup",
        "lineUp",
        "subs",
        "substitutes",
        "bench",
        "squad",
        "formation",
        "teamFormation",
    }
    player_exclude = {
        "playerId",
        "id",
        "personId",
        "name",
        "playerName",
        "position",
        "positionText",
        "shirtNo",
        "shirtNumber",
        "isFirstEleven",
        "isSub",
        "isSubstitute",
    }
    top_level_exclude = {
        "events",
        "home",
        "away",
        "playerIdNameDictionary",
        "incidents",
        "incidentEvents",
        "matchIncidents",
    }

    for game_id, payload in payloads.items():
        schedule_row = schedule_lookup.get(game_id, {})
        game_label = str(schedule_row.get("game") or "")
        for side in ("home", "away"):
            team_obj = payload.get(side)
            if not isinstance(team_obj, Mapping):
                continue
            team_id = _coerce_int(team_obj.get("teamId") or team_obj.get("id"))
            team_name = team_obj.get("name") or team_obj.get("teamName")
            stat_candidates: List[Tuple[str, Mapping[str, Any]]] = []
            for key in ("stats", "teamStats", "statistics"):
                value = team_obj.get(key)
                if isinstance(value, Mapping):
                    stat_candidates.append((key, value))
            if stats_mode == "all-visible":
                stat_candidates.append(("team", team_obj))
            for source_tab, stat_mapping in stat_candidates:
                for row in _collect_stats_from_mapping(
                    stat_mapping,
                    default_group=source_tab,
                    exclude_keys=team_exclude,
                ):
                    team_rows.append(
                        {
                            "league": league,
                            "season": season_int,
                            "game": game_label,
                            "game_id": game_id,
                            "team_id": team_id,
                            "team": team_name,
                            "source_tab": source_tab,
                            **row,
                        }
                    )

            for player_obj, _hint in _iter_player_containers(team_obj, "squad"):
                player_id = _coerce_int(
                    player_obj.get("playerId") or player_obj.get("id") or player_obj.get("personId")
                )
                stat_candidates = []
                for key in ("stats", "playerStats", "statistics"):
                    value = player_obj.get(key)
                    if isinstance(value, Mapping):
                        stat_candidates.append((key, value))
                if stats_mode == "all-visible":
                    stat_candidates.append(("player", player_obj))
                for source_tab, stat_mapping in stat_candidates:
                    for row in _collect_stats_from_mapping(
                        stat_mapping,
                        default_group=source_tab,
                        exclude_keys=player_exclude,
                    ):
                        player_rows.append(
                            {
                                "league": league,
                                "season": season_int,
                                "game": game_label,
                                "game_id": game_id,
                                "team_id": team_id,
                                "team": team_name,
                                "player_id": player_id,
                                "player": player_obj.get("name") or player_obj.get("playerName"),
                                "source_tab": source_tab,
                                **row,
                            }
                        )

        fact_candidates: List[Tuple[str, Mapping[str, Any]]] = []
        for key in ("facts", "matchFacts", "statistics"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                fact_candidates.append((key, value))
        if stats_mode == "all-visible":
            fact_candidates.append(("match", payload))
        for source_tab, stat_mapping in fact_candidates:
            for row in _collect_stats_from_mapping(
                stat_mapping,
                default_group=source_tab,
                exclude_keys=top_level_exclude,
            ):
                fact_rows.append(
                    {
                        "league": league,
                        "season": season_int,
                        "game": game_label,
                        "game_id": game_id,
                        "fact_group": row["stat_group"],
                        "fact_name": row["stat_name"],
                        "fact_key": row["stat_key"],
                        "value": row["value"],
                        "value_text": row["value_text"],
                        "source_tab": source_tab,
                    }
                )

        for source_tab, report_blob in report_blobs.get(game_id, {}).items():
            if not isinstance(report_blob, Mapping):
                continue
            for row in _collect_stats_from_mapping(report_blob, default_group=source_tab):
                fact_rows.append(
                    {
                        "league": league,
                        "season": season_int,
                        "game": game_label,
                        "game_id": game_id,
                        "fact_group": row["stat_group"],
                        "fact_name": row["stat_name"],
                        "fact_key": row["stat_key"],
                        "value": row["value"],
                        "value_text": row["value_text"],
                        "source_tab": f"report:{source_tab}",
                    }
                )

    return {
        "team_match_stats": pd.DataFrame(team_rows, columns=COLS_TEAM_MATCH_STATS),
        "player_match_stats": pd.DataFrame(player_rows, columns=COLS_PLAYER_MATCH_STATS),
        "match_fact_stats": pd.DataFrame(fact_rows, columns=COLS_MATCH_FACT_STATS),
    }


def _write_stats_tables(stats_dir: Path, tables: Mapping[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        safe_write(df, stats_dir / name)


def _support_schedule_df(
    *,
    ws: Any,
    out_dir: Path,
    league: str,
    season_int: int,
    force_cache: bool,
    log: logging.Logger,
) -> pd.DataFrame:
    existing = _read_existing_csv(out_dir / "ws_schedule")
    if not existing.empty:
        return _normalize(existing, league, season_int)
    try:
        schedule_df = ws.read_schedule(force_cache=force_cache)
        if isinstance(schedule_df, pd.DataFrame):
            return _normalize(schedule_df, league, season_int)
    except Exception as exc:
        log.debug("Support schedule load failed for %s %s: %s", league, season_int, exc)
    return _normalize(_schema_only_schedule(), league, season_int)


def _attach_cached_read_seasons(ws: Any, seasons_frame: Optional[pd.DataFrame]) -> None:
    if seasons_frame is None or seasons_frame.empty:
        return

    def _cached_read_seasons() -> pd.DataFrame:
        return seasons_frame.copy()

    try:
        ws.read_seasons = _cached_read_seasons  # type: ignore[assignment]
    except Exception:
        pass


def _native_backend(
    *,
    browser_fallback: bool,
    no_cache: bool,
    no_store: bool,
    path_to_browser: Optional[str],
    headless: bool,
) -> NativeWhoScoredBackend:
    competitions = load_native_competitions(REPO_NATIVE_COMPETITIONS_PATH)
    return NativeWhoScoredBackend(
        competitions=competitions,
        cache_dir=REPO_ROOT / "data" / "_whoscored_native_cache",
        no_cache=no_cache,
        no_store=no_store,
        browser_fallback=browser_fallback,
        path_to_browser=path_to_browser,
        headless=headless,
    )


def _resolve_native_seasons_frame_for_league(
    *,
    league: str,
    explicit_seasons: Optional[Sequence[str]],
    browser_fallback: bool,
    no_cache: bool,
    no_store: bool,
    path_to_browser: Optional[str],
    headless: bool,
) -> pd.DataFrame:
    backend = _native_backend(
        browser_fallback=browser_fallback,
        no_cache=no_cache,
        no_store=no_store,
        path_to_browser=path_to_browser,
        headless=headless,
    )
    return backend.resolve_seasons(league, explicit_seasons)


def scrape_one_native(
    *,
    league: str,
    season: Union[str, int],
    out_base: Path,
    delay: float,
    no_cache: bool,
    no_store: bool,
    path_to_browser: Optional[str],
    headless: bool,
    tables: Sequence[str],
    events_format: str,
    match_ids: Optional[Sequence[int]],
    skip_existing: bool,
    derived_tables: Sequence[str],
    archive_raw_events: bool,
    browser_fallback: bool,
    stats_mode: str,
    raw_artifacts: bool,
    retry_failed_matches: bool,
    match_limit: Optional[int],
    seasons_frame: pd.DataFrame,
) -> Dict[str, Any]:
    log = logging.getLogger("whoscored")
    season_int = _to_ws_season_int(season)
    out_dir = out_base / "WhoScored" / league / str(season_int)
    events_dir = out_dir / "events"
    derived_dir = out_dir / "derived"
    stats_dir = out_dir / "stats"
    raw_match_root = out_dir / "raw" / "matches"

    backend = _native_backend(
        browser_fallback=browser_fallback,
        no_cache=no_cache,
        no_store=no_store,
        path_to_browser=path_to_browser,
        headless=headless,
    )

    season_slice = seasons_frame[
        seasons_frame.index.get_level_values("season").astype(str) == str(season_int)
    ]
    if season_slice.empty:
        raise RuntimeError(f"Season {season_int} is not available for {league}")
    season_record = season_slice.reset_index().iloc[0].to_dict()

    summary: Dict[str, Any] = {
        "backend": "native",
        "schedule": None,
        "missing_players": None,
        "events": None,
        "events_format": events_format,
        "raw_archive": None,
        "derived": {},
        "stats": {},
        "counts": {},
        "failed_matches": [],
    }

    schedule_path = out_dir / "ws_schedule"
    schedule_df = pd.DataFrame()
    if "schedule" in tables:
        if skip_existing and _csv_exists(schedule_path):
            schedule_df = _normalize(_read_existing_csv(schedule_path), league, season_int)
            summary["schedule"] = "skipped_existing"
        else:
            schedule_df = backend.read_schedule(
                league=league,
                season_record=season_record,
                match_ids=match_ids,
            )
            schedule_df = _normalize(schedule_df, league, season_int)
            if not schedule_df.empty:
                if match_limit:
                    schedule_df = schedule_df.head(match_limit).reset_index(drop=True)
                safe_write(schedule_df, schedule_path)
                summary["schedule"] = "scraped_ok"
            else:
                empty = _normalize(_schema_only_schedule(), league, season_int)
                safe_write(empty, schedule_path)
                summary["schedule"] = "schema_only"
        polite_sleep(delay)
    else:
        schedule_df = _read_existing_csv(schedule_path)

    if schedule_df.empty:
        schedule_df = _normalize(schedule_df, league, season_int)

    summary["counts"]["matches_discovered"] = int(schedule_df.shape[0])

    if "missing_players" in tables:
        out_path = out_dir / "missing_players"
        if skip_existing and _csv_exists(out_path):
            summary["missing_players"] = "skipped_existing"
        else:
            mp_df, _preview_artifacts = backend.read_missing_players(
                league=league,
                season=str(season_int),
                schedule_df=schedule_df,
                raw_match_root=raw_match_root if raw_artifacts else None,
                match_limit=match_limit,
            )
            mp_df = _normalize(mp_df, league, season_int)
            if not mp_df.empty:
                safe_write(mp_df, out_path)
                summary["missing_players"] = "scraped_ok"
            else:
                safe_write(_normalize(_schema_only_missing_players(), league, season_int), out_path)
                summary["missing_players"] = "schema_only"
        polite_sleep(delay)

    if "events" in tables:
        normalized_events_path = events_dir / "events"
        if skip_existing and _csv_exists(normalized_events_path):
            summary["events"] = "skipped_existing"
        else:
            payloads, report_blobs, report_html, failures = backend.read_match_payloads(
                league=league,
                season=str(season_int),
                schedule_df=schedule_df,
                raw_match_root=raw_match_root if raw_artifacts or archive_raw_events else None,
                match_limit=match_limit,
                retry_failed_matches=retry_failed_matches,
                stats_mode=stats_mode,
            )
            summary["failed_matches"] = failures
            summary["counts"]["raw_matches_archived"] = len(payloads)
            summary["counts"]["matches_with_events"] = len([payload for payload in payloads.values() if payload.get("events")])

            parts = [
                _normalize_events_from_payload(
                    payload=payload,
                    schedule_row=_build_schedule_lookup(schedule_df).get(game_id),
                    league=league,
                    season_int=season_int,
                    game_id=game_id,
                )
                for game_id, payload in payloads.items()
            ]
            events_df = pd.concat(parts, ignore_index=True) if parts else _schema_only_events()
            events_df = _normalize(events_df, league, season_int)
            if events_format != "none":
                safe_write(events_df, normalized_events_path)
            summary["events"] = "scraped_ok" if not events_df.empty else "schema_only"

            if events_format == "raw":
                raw_flat_dir = events_dir / "raw"
                raw_flat_dir.mkdir(parents=True, exist_ok=True)
                for game_id, payload in payloads.items():
                    _save_json(raw_flat_dir / f"{game_id}.json", payload)
            elif events_format in {"spadl", "atomic-spadl"}:
                extra_name = "spadl" if events_format == "spadl" else "atomic_spadl"
                safe_write(_normalize(_schema_only_events(), league, season_int), events_dir / extra_name)

            if archive_raw_events:
                summary["raw_archive"] = "scraped_ok" if payloads else "schema_only"
            else:
                summary["raw_archive"] = "disabled"

            if derived_tables:
                derived_outputs = build_derived_tables(
                    payloads=payloads,
                    schedule_df=schedule_df,
                    league=league,
                    season_int=season_int,
                    derived_tables=derived_tables,
                )
                _write_derived_tables(derived_dir, derived_outputs)
                summary["derived"] = {
                    name: ("scraped_ok" if not derived_outputs[name].empty else "schema_only")
                    for name in derived_tables
                }

            stats_outputs = build_match_stats_tables(
                payloads=payloads,
                report_blobs=report_blobs,
                schedule_df=schedule_df,
                league=league,
                season_int=season_int,
                stats_mode=stats_mode,
            )
            _write_stats_tables(stats_dir, stats_outputs)
            summary["stats"] = {
                name: ("scraped_ok" if not df.empty else "schema_only")
                for name, df in stats_outputs.items()
            }
            summary["counts"]["matches_with_stats"] = len(report_html) or len(payloads)
        polite_sleep(delay)

    return summary


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
    derived_tables: Sequence[str],
    archive_raw_events: bool,
    raw_match_dir_layout: str,
    seasons_frame: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    log = logging.getLogger("whoscored")

    season_int = _to_ws_season_int(season)
    log.info("WhoScored scrape: %s %s", league, season_int)
    sync_repo_league_dict()

    ws = _build_ws(
        league,
        season_int,
        proxy=proxy,
        no_cache=no_cache,
        no_store=no_store,
        path_to_browser=path_to_browser,
        headless=headless,
    )
    _attach_cached_read_seasons(ws, seasons_frame)

    out_dir = out_base / "WhoScored" / league / str(season_int)
    events_dir = out_dir / "events"
    derived_dir = out_dir / "derived"
    raw_dir = events_dir / "raw"

    summary: Dict[str, Any] = {
        "schedule": None,
        "missing_players": None,
        "events": None,
        "events_format": events_format,
        "raw_archive": None,
        "derived": {},
    }
    schedule_df = pd.DataFrame()

    try:
        if "schedule" in tables:
            out_path = out_dir / "ws_schedule"
            if skip_existing and _csv_exists(out_path):
                summary["schedule"] = "skipped_existing"
                schedule_df = _normalize(_read_existing_csv(out_path), league, season_int)
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
            schedule_support = schedule_df
            if schedule_support.empty:
                schedule_support = _support_schedule_df(
                    ws=ws,
                    out_dir=out_dir,
                    league=league,
                    season_int=season_int,
                    force_cache=force_cache,
                    log=log,
                )

            normalized_events_path = events_dir / "events"
            extra_events_path = (
                events_dir / "spadl"
                if events_format == "spadl"
                else events_dir / "atomic_spadl"
                if events_format == "atomic-spadl"
                else None
            )
            normalized_exists = _csv_exists(normalized_events_path)
            raw_exists = raw_dir.is_dir() and any(raw_dir.glob("*.json"))
            derived_exists = all(_csv_exists(derived_dir / name) for name in derived_tables)

            if (
                skip_existing
                and normalized_exists
                and (not archive_raw_events or raw_exists)
                and (not derived_tables or derived_exists)
                and (
                    events_format not in {"spadl", "atomic-spadl"}
                    or (extra_events_path is not None and _csv_exists(extra_events_path))
                )
            ):
                summary["events"] = "skipped_existing"
                summary["raw_archive"] = "skipped_existing" if archive_raw_events else "disabled"
                summary["derived"] = {name: "skipped_existing" for name in derived_tables}
            else:
                try:
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
                        normalized = ws.read_events(
                            match_id=match_ids,
                            force_cache=force_cache,
                            live=live,
                            output_fmt="events",
                            retry_missing=retry_missing,
                            on_error=on_error,
                        )
                        normalized_df = normalized if isinstance(normalized, pd.DataFrame) else pd.DataFrame()
                        if normalized_df.empty:
                            raw_payloads = _load_cached_raw_payloads(
                                ws=ws,
                                league=league,
                                season_int=season_int,
                                match_ids=match_ids,
                            )
                            schedule_lookup = _build_schedule_lookup(schedule_support)
                            parts = [
                                _normalize_events_from_payload(
                                    payload=payload,
                                    schedule_row=schedule_lookup.get(game_id),
                                    league=league,
                                    season_int=season_int,
                                    game_id=game_id,
                                )
                                for game_id, payload in raw_payloads.items()
                            ]
                            normalized_df = (
                                pd.concat(parts, ignore_index=True) if parts else _schema_only_events()
                            )
                        normalized_df = _normalize(normalized_df, league, season_int)
                        safe_write(normalized_df, normalized_events_path)
                        summary["events"] = "scraped_ok"

                        if events_format in {"spadl", "atomic-spadl"} and extra_events_path is not None:
                            extra_df = ws.read_events(
                                match_id=match_ids,
                                force_cache=force_cache,
                                live=live,
                                output_fmt=events_format,
                                retry_missing=retry_missing,
                                on_error=on_error,
                            )
                            extra_df = extra_df if isinstance(extra_df, pd.DataFrame) else pd.DataFrame()
                            extra_df = _normalize(extra_df, league, season_int)
                            safe_write(extra_df, extra_events_path)

                    raw_payloads = _load_cached_raw_payloads(
                        ws=ws,
                        league=league,
                        season_int=season_int,
                        match_ids=match_ids,
                    )
                    if archive_raw_events and raw_match_dir_layout == "per-match":
                        _archive_raw_payloads(raw_dir, raw_payloads)
                        summary["raw_archive"] = "scraped_ok"
                    else:
                        summary["raw_archive"] = "disabled"

                    if derived_tables:
                        derived_outputs = build_derived_tables(
                            payloads=raw_payloads,
                            schedule_df=schedule_support,
                            league=league,
                            season_int=season_int,
                            derived_tables=derived_tables,
                        )
                        _write_derived_tables(derived_dir, derived_outputs)
                        summary["derived"] = {name: "scraped_ok" for name in derived_tables}
                    polite_sleep(delay)
                except Exception as e:
                    log.warning(
                        "Failed to scrape events for %s %s: %s",
                        league,
                        season_int,
                        e,
                    )
                    if events_format != "none":
                        safe_write(_normalize(_schema_only_events(), league, season_int), normalized_events_path)
                    if archive_raw_events:
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        summary["raw_archive"] = "schema_only"
                    else:
                        summary["raw_archive"] = "disabled"
                    for name in derived_tables:
                        empty = {
                            "match_info": _schema_only_match_info(),
                            "incidents": _schema_only_incidents(),
                            "player_dictionary": _schema_only_player_dictionary(),
                            "lineups": _schema_only_lineups(),
                            "formations": _schema_only_formations(),
                        }[name]
                        safe_write(empty, derived_dir / name)
                    summary["events"] = "schema_only"
                    summary["derived"] = {name: "schema_only" for name in derived_tables}

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Scrape WhoScored match-level data")
    p.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="native",
        help="Scrape backend. Native is requests-first with browser fallback; soccerdata is kept as a legacy path.",
    )
    p.add_argument(
        "--league",
        nargs="+",
        default=None,
        help="One or more configured competition identifiers.",
    )
    p.add_argument(
        "--competition-group",
        action="append",
        default=[],
        help="Repeatable group selector from config/whoscored_competition_groups.json.",
    )
    p.add_argument(
        "--all-known-leagues",
        action="store_true",
        help="Backward-compatible alias for the top5-domestic group.",
    )
    p.add_argument(
        "--all-configured-competitions",
        action="store_true",
        help="Ignore --league and scrape every configured competition.",
    )
    p.add_argument(
        "--list-competitions",
        action="store_true",
        help="Print configured competitions and exit.",
    )
    p.add_argument(
        "--list-groups",
        action="store_true",
        help="Print configured competition groups and exit.",
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
        help="Optional extra event output format. Normalized events CSV is always written when events are scraped.",
    )
    p.add_argument(
        "--derived-tables",
        nargs="+",
        choices=DERIVED_TABLE_CHOICES,
        default=DERIVED_TABLE_CHOICES,
        help="Derived raw-payload tables to persist.",
    )
    p.add_argument(
        "--no-derived-tables",
        action="store_true",
        help="Disable derived table output.",
    )
    p.add_argument(
        "--archive-raw-events",
        dest="archive_raw_events",
        action="store_true",
        default=True,
        help="Archive per-match full raw event payloads.",
    )
    p.add_argument(
        "--no-archive-raw-events",
        dest="archive_raw_events",
        action="store_false",
        help="Disable raw event payload archive.",
    )
    p.add_argument(
        "--raw-match-dir-layout",
        choices=["per-match"],
        default="per-match",
        help="Raw archive directory strategy.",
    )
    p.add_argument(
        "--browser-fallback",
        dest="browser_fallback",
        action="store_true",
        default=True,
        help="Allow Selenium browser fallback for native backend when direct HTTP extraction fails.",
    )
    p.add_argument(
        "--no-browser-fallback",
        dest="browser_fallback",
        action="store_false",
        help="Disable Selenium browser fallback for native backend.",
    )
    p.add_argument(
        "--stats-mode",
        choices=STATS_MODE_CHOICES,
        default="all-visible",
        help="Controls how much match statistics are normalized from match-centre and report payloads.",
    )
    p.add_argument(
        "--raw-artifacts",
        dest="raw_artifacts",
        action="store_true",
        default=True,
        help="Persist raw preview/match/report artifacts under raw/matches/<game_id>/.",
    )
    p.add_argument(
        "--no-raw-artifacts",
        dest="raw_artifacts",
        action="store_false",
        help="Disable raw artifact persistence.",
    )
    p.add_argument(
        "--retry-failed-matches",
        action="store_true",
        help="Retry failed native match payload scrapes once.",
    )
    p.add_argument(
        "--match-limit",
        type=int,
        default=None,
        help="Optional limit for matches per league-season, useful for safe testing.",
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
    return p


def _print_listings(args: argparse.Namespace) -> bool:
    if args.list_competitions:
        print(json.dumps(_load_native_competitions(), ensure_ascii=True, indent=2, sort_keys=True))
        return True
    if args.list_groups:
        print(json.dumps(_load_competition_groups(), ensure_ascii=True, indent=2, sort_keys=True))
        return True
    return False


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = build_parser()
    args = p.parse_args(list(argv) if argv is not None else None)

    init_logger(args.verbose)
    log = logging.getLogger("whoscored")
    if _print_listings(args):
        return

    if args.no_cache and args.force_cache:
        raise SystemExit("--no-cache and --force-cache are mutually exclusive.")

    sync_repo_league_dict()
    leagues = resolve_competitions(
        explicit_leagues=args.league,
        competition_groups=args.competition_group,
        all_known_leagues=args.all_known_leagues,
        all_configured_competitions=args.all_configured_competitions,
    )
    if not leagues:
        leagues = resolve_competitions(
            explicit_leagues=None,
            competition_groups=["top5-domestic"],
            all_known_leagues=False,
            all_configured_competitions=False,
        )
    match_ids_raw = _parse_list(args.match_ids)
    match_ids = [int(x) for x in match_ids_raw] if match_ids_raw else None
    derived_tables = [] if args.no_derived_tables else list(args.derived_tables)

    out_base = Path(args.out_dir)
    meta_path = Path(args.meta_path)

    for league in leagues:
        if args.backend == "native":
            seasons_frame = _resolve_native_seasons_frame_for_league(
                league=league,
                explicit_seasons=args.seasons,
                browser_fallback=args.browser_fallback,
                no_cache=args.no_cache,
                no_store=args.no_store,
                path_to_browser=args.path_to_browser,
                headless=args.headless,
            )
        else:
            seasons_frame = _resolve_seasons_frame_for_league(
                league=league,
                explicit_seasons=args.seasons,
                proxy=args.proxy,
                no_cache=args.no_cache,
                no_store=args.no_store,
                path_to_browser=args.path_to_browser,
                headless=args.headless,
            )
        seasons = seasons_frame.index.get_level_values("season").astype(str).unique().tolist()
        log.info("Resolved seasons for %s: %s", league, seasons)
        for s in seasons:
            season_frame = seasons_frame[
                seasons_frame.index.get_level_values("season").astype(str) == str(s)
            ].copy()
            if args.backend == "native":
                summary = scrape_one_native(
                    league=league,
                    season=s,
                    out_base=out_base,
                    delay=args.delay,
                    no_cache=args.no_cache,
                    no_store=args.no_store,
                    path_to_browser=args.path_to_browser,
                    headless=args.headless,
                    tables=args.tables,
                    events_format=args.events_format,
                    match_ids=match_ids,
                    skip_existing=args.skip_existing,
                    derived_tables=derived_tables,
                    archive_raw_events=args.archive_raw_events and "events" in args.tables,
                    browser_fallback=args.browser_fallback,
                    stats_mode=args.stats_mode,
                    raw_artifacts=args.raw_artifacts,
                    retry_failed_matches=args.retry_failed_matches,
                    match_limit=args.match_limit,
                    seasons_frame=season_frame,
                )
            else:
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
                    derived_tables=derived_tables,
                    archive_raw_events=args.archive_raw_events and "events" in args.tables,
                    raw_match_dir_layout=args.raw_match_dir_layout,
                    seasons_frame=season_frame,
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
                "backend": args.backend,
                "stats_summary": summary,
                "tables": list(args.tables),
                "events_format": args.events_format,
                "derived_tables": derived_tables,
                "archive_raw_events": bool(args.archive_raw_events and "events" in args.tables),
                "stats_mode": args.stats_mode,
            }
            record_last_run(meta_path, job, run_info=run_info)
            log.info("Recorded last-run meta for %s", job.key())


if __name__ == "__main__":
    main()
