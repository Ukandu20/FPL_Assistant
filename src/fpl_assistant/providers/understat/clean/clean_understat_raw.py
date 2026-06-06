#!/usr/bin/env python3
"""
Clean raw Understat CSV files into normalized team/player-friendly outputs.

Output contract:
  data/processed/understat/<LEAGUE_STD>/<SEASON_LONG>/<same_filename>.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

LEAGUE_MAP: Dict[str, str] = {
    "EPL": "ENG-Premier League",
    "Bundesliga": "GER-Bundesliga",
    "La liga": "ESP-La Liga",
    "Ligue 1": "FRA-Ligue 1",
    "Serie A": "ITA-Serie A",
    "RFPL": "RUS-Russian PL",
}

MODERN_LEAGUE_FOLDERS = list(LEAGUE_MAP.keys())

POS_MAP_PLAYER_MATCH: Dict[str, str] = {
    "GK": "GK",
    "DR": "DEF",
    "DC": "DEF",
    "DL": "DEF",
    "DMR": "DEF",
    "DML": "DEF",
    "DMC": "MID",
    "MC": "MID",
    "AMC": "MID",
    "FWL": "MID",
    "FWR": "MID",
    "AML": "MID",
    "AMR": "MID",
    "ML": "MID",
    "MR": "MID",
    "FW": "FWD",
    "Sub": "SUB",
}

FPL_POS_PRECEDENCE: Dict[str, int] = {
    "GK": 0,
    "DEF": 1,
    "MID": 2,
    "FWD": 3,
    "SUB": 4,
    "UNK": 5,
}

LOG = logging.getLogger("understat_clean")


def init_logger(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def season_to_long(season_value: Any) -> str:
    text = str(season_value).strip()
    m = re.fullmatch(r"(\d{4})-(\d{4})", text)
    if m:
        return text
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
        return f"{year}-{year + 1}"
    m = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if m:
        return f"{m.group(1)}-20{m.group(2)}"
    raise ValueError(f"Unsupported season format: {season_value!r}")


def season_long_to_short(season_long: str) -> str:
    m = re.fullmatch(r"(\d{4})-(\d{4})", str(season_long).strip())
    if not m:
        raise ValueError(f"Expected season long format YYYY-YYYY, got: {season_long!r}")
    return f"{m.group(1)}-{m.group(2)[2:]}"


def map_league_value(raw_league: str) -> str:
    if raw_league in LEAGUE_MAP:
        return LEAGUE_MAP[raw_league]
    # fallback by case-insensitive comparison
    low = raw_league.strip().lower()
    for k, v in LEAGUE_MAP.items():
        if k.lower() == low:
            return v
    return raw_league


def normalize_player_key(value: Any) -> str:
    text = unescape(str(value))
    text = text.replace("\u2019", "'").replace("`", "'")
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def parse_game_row(row: pd.Series) -> Tuple[str, str, str]:
    game_raw = _safe_str(row.get("game")).strip()
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})_(.+)_vs_(.+)", game_raw)
    if m:
        return m.group(1), m.group(2), m.group(3)

    date_raw = _safe_str(row.get("date")).strip()
    date_only = date_raw.split(" ")[0] if " " in date_raw else date_raw
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_only):
        date_only = ""
    home = _safe_str(row.get("home_team")).strip()
    away = _safe_str(row.get("away_team")).strip()
    return date_only, home, away


def _extract_game_time(date_value: Any) -> str:
    text = _safe_str(date_value).strip()
    if not text:
        return ""
    try:
        ts = pd.to_datetime(text)
        return ts.strftime("%H:%M:%S")
    except Exception:
        if " " in text:
            return text.split(" ", 1)[1].strip()
    return ""


def build_player_lookup(player_lookup_raw: Dict[str, str]) -> Tuple[Dict[str, str], set[str]]:
    bucket: Dict[str, set[str]] = defaultdict(set)
    for raw_name, pid in player_lookup_raw.items():
        key = normalize_player_key(raw_name)
        if not key:
            continue
        bucket[key].add(str(pid))

    lookup: Dict[str, str] = {}
    ambiguous: set[str] = set()
    for key, ids in bucket.items():
        if len(ids) == 1:
            lookup[key] = next(iter(ids))
        else:
            ambiguous.add(key)
    return lookup, ambiguous


def normalize_team_lookup(team_lookup_raw: Dict[str, str]) -> Dict[str, str]:
    return {str(k).strip().lower(): str(v).strip() for k, v in team_lookup_raw.items()}


def normalize_team_name(value: Any) -> str:
    text = unescape(str(value))
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _pick_canonical_team_name(candidates: List[str]) -> str:
    stop_words = {
        "fc",
        "f.c.",
        "football club",
        "club",
        "afc",
        "a.f.c.",
        "cf",
        "c.f.",
    }

    def _score(name: str) -> Tuple[int, int, int, str]:
        low = name.lower()
        has_suffix = 1 if any(sw in low for sw in stop_words) else 0
        is_lower = 1 if name == low else 0
        return (has_suffix, is_lower, len(name), name)

    return sorted(candidates, key=_score)[0]


def build_teams_config_maps(teams_config_raw: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    name_to_code: Dict[str, str] = {}
    name_to_display_candidates: Dict[str, set[str]] = defaultdict(set)

    for raw_name, raw_code in teams_config_raw.items():
        code = str(raw_code).strip().upper()
        if not code:
            continue
        normalized_name = normalize_team_name(raw_name)
        if not normalized_name:
            continue
        if normalized_name not in name_to_code:
            name_to_code[normalized_name] = code
        name_to_display_candidates[normalized_name].add(str(raw_name).strip())

    name_to_display: Dict[str, str] = {}
    for normalized_name, names in name_to_display_candidates.items():
        cleaned = sorted(n for n in names if n)
        if not cleaned:
            continue
        name_to_display[normalized_name] = _pick_canonical_team_name(cleaned)

    return name_to_code, name_to_display


def _canonical_team_code(
    raw_code: Any,
    raw_name: Any,
    *,
    teams_name_to_code: Dict[str, str],
) -> str:
    name_key = normalize_team_name(raw_name)
    from_name = teams_name_to_code.get(name_key)
    if from_name:
        return from_name
    return _safe_str(raw_code).strip().upper()


def _canonical_team_name(
    raw_name: Any,
    *,
    teams_name_to_display: Dict[str, str],
) -> str:
    name_key = normalize_team_name(raw_name)
    from_name = teams_name_to_display.get(name_key)
    if from_name:
        return from_name
    return _safe_str(raw_name).strip()


def _to_upper_code(value: Any) -> str:
    text = _safe_str(value).strip().upper()
    return text


def _clean_result_value(value: Any) -> str:
    text = _safe_str(value).strip().upper()
    if text in {"W", "L", "D"}:
        return text
    return ""


def _build_game_from_codes(home_code: Any, away_code: Any, fallback: Any = "") -> str:
    h = _to_upper_code(home_code)
    a = _to_upper_code(away_code)
    if h and a:
        return f"{h} - {a}"
    return _safe_str(fallback).strip()


def standardize_team_and_game_codes(
    df: pd.DataFrame,
    *,
    teams_name_to_code: Dict[str, str],
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    if "team" in out.columns:
        out["team"] = out["team"].apply(
            lambda v: _canonical_team_code(v, v, teams_name_to_code=teams_name_to_code)
        )
    if "opp" in out.columns:
        out["opp"] = out["opp"].apply(
            lambda v: _canonical_team_code(v, v, teams_name_to_code=teams_name_to_code)
        )

    home_code_col: Optional[pd.Series] = None
    away_code_col: Optional[pd.Series] = None

    if "home_team_code" in out.columns:
        home_code_col = out["home_team_code"].apply(
            lambda v: _canonical_team_code(v, v, teams_name_to_code=teams_name_to_code)
        )
    elif "home_team_name" in out.columns:
        home_code_col = out["home_team_name"].apply(
            lambda v: _canonical_team_code("", v, teams_name_to_code=teams_name_to_code)
        )

    if "away_team_code" in out.columns:
        away_code_col = out["away_team_code"].apply(
            lambda v: _canonical_team_code(v, v, teams_name_to_code=teams_name_to_code)
        )
    elif "away_team_name" in out.columns:
        away_code_col = out["away_team_name"].apply(
            lambda v: _canonical_team_code("", v, teams_name_to_code=teams_name_to_code)
        )

    if home_code_col is None or away_code_col is None:
        if "game" in out.columns:
            parsed = out.apply(parse_game_row, axis=1, result_type="expand")
            parsed.columns = ["__game_date", "__game_home", "__game_away"]
            if home_code_col is None:
                home_code_col = parsed["__game_home"].apply(
                    lambda v: _canonical_team_code("", v, teams_name_to_code=teams_name_to_code)
                )
            if away_code_col is None:
                away_code_col = parsed["__game_away"].apply(
                    lambda v: _canonical_team_code("", v, teams_name_to_code=teams_name_to_code)
                )

    if "game" in out.columns and home_code_col is not None and away_code_col is not None:
        out["game"] = [
            _build_game_from_codes(h, a, fallback=g)
            for h, a, g in zip(home_code_col, away_code_col, out["game"])
        ]

    # Drop redundant team-name columns once codes are available.
    for col in ["home_team_name", "away_team_name"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    return out


def _normalize_override_side(text: str) -> str:
    return normalize_player_key(text)


def normalize_override_key(raw_key: str) -> str:
    text = str(raw_key)
    if "|" not in text:
        return normalize_player_key(text)
    left, right = text.split("|", 1)
    left_n = _normalize_override_side(left)
    right_n = _normalize_override_side(right)
    return f"{left_n} | {right_n}"


def build_override_lookup(override_raw: Dict[str, str]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = defaultdict(set)
    for k, v in override_raw.items():
        nk = normalize_override_key(k)
        if not nk:
            continue
        out[nk].add(str(v))
    return out


def override_candidate_keys(player_name: str) -> List[str]:
    norm_name = normalize_player_key(player_name)
    if not norm_name:
        return []
    toks = norm_name.split()
    if len(toks) < 2:
        return []
    keys: set[str] = set()
    # Every contiguous left/right split
    for i in range(1, len(toks)):
        left = " ".join(toks[:i])
        right = " ".join(toks[i:])
        keys.add(f"{left} | {right}")
    # Common "first | last" short form
    keys.add(f"{toks[0]} | {toks[-1]}")
    return sorted(keys)


def annotate_missing_with_overrides(
    missing_df: pd.DataFrame,
    override_lookup: Dict[str, set[str]],
) -> pd.DataFrame:
    if missing_df.empty:
        return missing_df.copy()
    out = missing_df.copy()

    def _resolve(player: str) -> Tuple[bool, str, str, Optional[str], bool]:
        keys = override_candidate_keys(player)
        matched_keys = [k for k in keys if k in override_lookup]
        if not matched_keys:
            return False, "", "", None, False
        ids: set[str] = set()
        for k in matched_keys:
            ids.update(override_lookup[k])
        ids_sorted = sorted(ids)
        keys_str = "; ".join(matched_keys)
        ids_str = "; ".join(ids_sorted)
        is_unique = len(ids_sorted) == 1
        resolved_id = ids_sorted[0] if is_unique else None
        return True, keys_str, ids_str, resolved_id, is_unique

    resolved = out["player"].astype(str).apply(_resolve)
    out["override_match_found"] = resolved.apply(lambda x: x[0])
    out["override_candidate_keys"] = resolved.apply(lambda x: x[1])
    out["override_candidate_ids"] = resolved.apply(lambda x: x[2])
    out["override_resolved_player_id"] = resolved.apply(lambda x: x[3])
    out["override_unique_match"] = resolved.apply(lambda x: x[4])
    return out


def map_player_match_pos(raw_position: Any) -> str:
    raw = _safe_str(raw_position).strip()
    return POS_MAP_PLAYER_MATCH.get(raw, "UNK")


def apply_common_league_season(df: pd.DataFrame, league_std: str, season_long: str) -> pd.DataFrame:
    out = df.copy()
    if "league" in out.columns:
        out["league"] = league_std
    if "season" in out.columns:
        out["season"] = season_long
    return out


def _result_from_pov(row: pd.Series, is_home: bool) -> str:
    if not to_bool(row.get("is_result")):
        return ""
    hg = row.get("home_goals")
    ag = row.get("away_goals")
    if pd.isna(hg) or pd.isna(ag):
        return ""
    h = float(hg)
    a = float(ag)
    if is_home:
        if h > a:
            return "W"
        if h < a:
            return "L"
        return "D"
    if a > h:
        return "W"
    if a < h:
        return "L"
    return "D"


def transform_schedule(
    schedule_df: pd.DataFrame,
    *,
    league_std: str,
    season_long: str,
    team_lookup: Dict[str, str],
    teams_name_to_code: Optional[Dict[str, str]] = None,
    teams_name_to_display: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = apply_common_league_season(schedule_df, league_std=league_std, season_long=season_long)
    teams_name_to_code = teams_name_to_code or {}
    teams_name_to_display = teams_name_to_display or {}

    if "home_team_code" in df.columns and "home_team" in df.columns:
        df["home_team_code"] = df.apply(
            lambda r: _canonical_team_code(
                r.get("home_team_code"),
                r.get("home_team"),
                teams_name_to_code=teams_name_to_code,
            ),
            axis=1,
        )
    if "away_team_code" in df.columns and "away_team" in df.columns:
        df["away_team_code"] = df.apply(
            lambda r: _canonical_team_code(
                r.get("away_team_code"),
                r.get("away_team"),
                teams_name_to_code=teams_name_to_code,
            ),
            axis=1,
        )
    if "home_team" in df.columns and "home_team_code" in df.columns:
        df["home_team"] = df.apply(
            lambda r: _canonical_team_name(
                r.get("home_team"),
                teams_name_to_display=teams_name_to_display,
            ),
            axis=1,
        )
    if "away_team" in df.columns and "away_team_code" in df.columns:
        df["away_team"] = df.apply(
            lambda r: _canonical_team_name(
                r.get("away_team"),
                teams_name_to_display=teams_name_to_display,
            ),
            axis=1,
        )

    parsed = df.apply(parse_game_row, axis=1, result_type="expand")
    parsed.columns = ["game_date", "home_name_from_game", "away_name_from_game"]

    df["game_date"] = parsed["game_date"]
    home_code = df.get("home_team_code", pd.Series("", index=df.index)).astype(str).str.strip().str.upper()
    away_code = df.get("away_team_code", pd.Series("", index=df.index)).astype(str).str.strip().str.upper()
    df["game"] = home_code + " - " + away_code
    df["game_time"] = df["date"].apply(_extract_game_time) if "date" in df.columns else ""

    home = df.copy()
    home["team"] = home.get("home_team_code")
    home["opp"] = home.get("away_team_code")
    home["team_code"] = home.get("home_team_code")
    home["opp_code"] = home.get("away_team_code")
    home["venue"] = "H"
    home["understat_team_id"] = home.get("home_team_id")
    home["understat_opp_id"] = home.get("away_team_id")
    home["result"] = home.apply(lambda r: _result_from_pov(r, True), axis=1)
    home["team_goals"] = home.get("home_goals")
    home["opp_goals"] = home.get("away_goals")
    home["team_xg"] = home.get("home_xg")
    home["opp_xg"] = home.get("away_xg")
    home["forecast_win"] = home.get("forecast_home_win")
    home["forecast_draw"] = home.get("forecast_draw")
    home["forecast_loss"] = home.get("forecast_away_win")

    away = df.copy()
    away["team"] = away.get("away_team_code")
    away["opp"] = away.get("home_team_code")
    away["team_code"] = away.get("away_team_code")
    away["opp_code"] = away.get("home_team_code")
    away["venue"] = "A"
    away["understat_team_id"] = away.get("away_team_id")
    away["understat_opp_id"] = away.get("home_team_id")
    away["result"] = away.apply(lambda r: _result_from_pov(r, False), axis=1)
    away["team_goals"] = away.get("away_goals")
    away["opp_goals"] = away.get("home_goals")
    away["team_xg"] = away.get("away_xg")
    away["opp_xg"] = away.get("home_xg")
    away["forecast_win"] = away.get("forecast_away_win")
    away["forecast_draw"] = away.get("forecast_draw")
    away["forecast_loss"] = away.get("forecast_home_win")

    out = pd.concat([home, away], ignore_index=True)
    out["team_id"] = out["team_code"].astype(str).str.lower().map(team_lookup)
    out["opp_id"] = out["opp_code"].astype(str).str.lower().map(team_lookup)
    out["team_id_missing"] = out["team_id"].isna()
    out["opp_id_missing"] = out["opp_id"].isna()

    sort_keys = [c for c in ["team", "game_date", "game_time", "game_id"] if c in out.columns]
    out = out.sort_values(sort_keys).reset_index(drop=True)
    out["round"] = (
        out.groupby(["league", "season", "team"], dropna=False).cumcount() + 1
    ).astype("Int64")
    out = out.sort_values([c for c in ["team", "round", "game_date", "game_time", "game_id"] if c in out.columns]).reset_index(drop=True)

    team_missing = pd.concat(
        [
            out.loc[out["team_id_missing"], ["league", "season", "team_code"]],
            out.loc[out["opp_id_missing"], ["league", "season", "opp_code"]].rename(columns={"opp_code": "team_code"}),
        ],
        ignore_index=True,
    )
    team_missing = team_missing.dropna(subset=["team_code"])

    # Streamlined team-POV schema: metadata first, then team/opp descriptors, then match metrics.
    ordered_cols = [
        "league",
        "season",
        "league_id",
        "season_id",
        "game_id",
        "game_date",
        "game_time",
        "game",
        "team",
        "result",
        "round",
        "team_id",
        "understat_team_id",
        "opp",
        "opp_id",
        "understat_opp_id",
        "venue",
        "team_goals",
        "opp_goals",
        "team_xg",
        "opp_xg",
        "forecast_win",
        "forecast_draw",
        "forecast_loss",
        "url",
        "is_result",
        "has_data",
    ]
    out = out[[c for c in ordered_cols if c in out.columns]]
    return out, team_missing


def transform_team_match(
    team_match_df: pd.DataFrame,
    *,
    league_std: str,
    season_long: str,
    team_lookup: Dict[str, str],
    teams_name_to_code: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = apply_common_league_season(team_match_df, league_std=league_std, season_long=season_long)

    if "home_team_code" in df.columns and "home_team" in df.columns:
        df["home_team_code"] = df.apply(
            lambda r: _canonical_team_code(
                r.get("home_team_code"),
                r.get("home_team"),
                teams_name_to_code=teams_name_to_code,
            ),
            axis=1,
        )
    if "away_team_code" in df.columns and "away_team" in df.columns:
        df["away_team_code"] = df.apply(
            lambda r: _canonical_team_code(
                r.get("away_team_code"),
                r.get("away_team"),
                teams_name_to_code=teams_name_to_code,
            ),
            axis=1,
        )

    parsed = df.apply(parse_game_row, axis=1, result_type="expand")
    parsed.columns = ["game_date", "home_name_from_game", "away_name_from_game"]
    df["game_date"] = parsed["game_date"]
    df["game_time"] = df["date"].apply(_extract_game_time) if "date" in df.columns else ""
    df["game"] = [
        _build_game_from_codes(h, a, fallback=g)
        for h, a, g in zip(df.get("home_team_code", ""), df.get("away_team_code", ""), df.get("game", ""))
    ]

    home = df.copy()
    home["team"] = home.get("home_team_code")
    home["opp"] = home.get("away_team_code")
    home["venue"] = "H"
    home["understat_team_id"] = home.get("home_team_id")
    home["understat_opp_id"] = home.get("away_team_id")
    home["result"] = home.get("home_result", pd.Series("", index=home.index)).apply(_clean_result_value)
    home["points"] = home.get("home_points")
    home["expected_points"] = home.get("home_expected_points")
    home["goals"] = home.get("home_goals")
    home["goals_against"] = home.get("home_goals_against")
    home["xg"] = home.get("home_xg")
    home["xga"] = home.get("home_xga")
    home["np_xg"] = home.get("home_np_xg")
    home["np_xga"] = home.get("home_np_xga")
    home["np_xg_difference"] = home.get("home_np_xg_difference")
    home["wins"] = home.get("home_wins")
    home["draws"] = home.get("home_draws")
    home["losses"] = home.get("home_losses")
    home["ppda"] = home.get("home_ppda")
    home["ppda_allowed"] = home.get("home_ppda_allowed")
    home["deep_completions"] = home.get("home_deep_completions")
    home["deep_completions_allowed"] = home.get("home_deep_completions_allowed")

    away = df.copy()
    away["team"] = away.get("away_team_code")
    away["opp"] = away.get("home_team_code")
    away["venue"] = "A"
    away["understat_team_id"] = away.get("away_team_id")
    away["understat_opp_id"] = away.get("home_team_id")
    away["result"] = away.get("away_result", pd.Series("", index=away.index)).apply(_clean_result_value)
    away["points"] = away.get("away_points")
    away["expected_points"] = away.get("away_expected_points")
    away["goals"] = away.get("away_goals")
    away["goals_against"] = away.get("away_goals_against")
    away["xg"] = away.get("away_xg")
    away["xga"] = away.get("away_xga")
    away["np_xg"] = away.get("away_np_xg")
    away["np_xga"] = away.get("away_np_xga")
    away["np_xg_difference"] = away.get("away_np_xg_difference")
    away["wins"] = away.get("away_wins")
    away["draws"] = away.get("away_draws")
    away["losses"] = away.get("away_losses")
    away["ppda"] = away.get("away_ppda")
    away["ppda_allowed"] = away.get("away_ppda_allowed")
    away["deep_completions"] = away.get("away_deep_completions")
    away["deep_completions_allowed"] = away.get("away_deep_completions_allowed")

    out = pd.concat([home, away], ignore_index=True)
    out["team_id"] = out["team"].astype(str).str.lower().map(team_lookup)
    out["opp_id"] = out["opp"].astype(str).str.lower().map(team_lookup)
    out["team_id_missing"] = out["team_id"].isna()
    out["opp_id_missing"] = out["opp_id"].isna()

    sort_keys = [c for c in ["team", "game_date", "game_time", "game_id"] if c in out.columns]
    out = out.sort_values(sort_keys).reset_index(drop=True)
    out["round"] = (
        out.groupby(["league", "season", "team"], dropna=False).cumcount() + 1
    ).astype("Int64")
    out = out.sort_values([c for c in ["team", "round", "game_date", "game_time", "game_id"] if c in out.columns]).reset_index(drop=True)

    team_missing = pd.concat(
        [
            out.loc[out["team_id_missing"], ["league", "season", "team"]].rename(columns={"team": "team_code"}),
            out.loc[out["opp_id_missing"], ["league", "season", "opp"]].rename(columns={"opp": "team_code"}),
        ],
        ignore_index=True,
    )
    team_missing = team_missing.dropna(subset=["team_code"])

    ordered_cols = [
        "league",
        "season",
        "league_id",
        "season_id",
        "game_id",
        "game_date",
        "game_time",
        "game",
        "team",
        "result",
        "round",
        "team_id",
        "understat_team_id",
        "opp",
        "opp_id",
        "understat_opp_id",
        "venue",
        "points",
        "expected_points",
        "goals",
        "goals_against",
        "xg",
        "xga",
        "np_xg",
        "np_xga",
        "np_xg_difference",
        "wins",
        "draws",
        "losses",
        "ppda",
        "ppda_allowed",
        "deep_completions",
        "deep_completions_allowed",
    ]
    out = out[[c for c in ordered_cols if c in out.columns]]
    return out, team_missing


def transform_team_season(
    team_season_df: pd.DataFrame,
    *,
    league_std: str,
    season_long: str,
    team_lookup: Dict[str, str],
    teams_name_to_code: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = apply_common_league_season(team_season_df, league_std=league_std, season_long=season_long)
    if "team_code" in out.columns:
        out["team"] = out.apply(
            lambda r: _canonical_team_code(
                r.get("team_code"),
                r.get("team"),
                teams_name_to_code=teams_name_to_code,
            ),
            axis=1,
        )
    elif "team" in out.columns:
        out["team"] = out["team"].apply(
            lambda v: _canonical_team_code(v, v, teams_name_to_code=teams_name_to_code)
        )

    if "team_id" in out.columns:
        out["understat_team_id"] = out["team_id"]
    else:
        out["understat_team_id"] = pd.NA

    out["team_id"] = out["team"].astype(str).str.lower().map(team_lookup)
    out["team_id_missing"] = out["team_id"].isna()

    team_missing = out.loc[out["team_id_missing"], ["league", "season", "team"]].rename(
        columns={"team": "team_code"}
    )

    ordered_cols = [
        "league",
        "season",
        "league_id",
        "season_id",
        "team",
        "team_id",
        "understat_team_id",
        "matches",
        "home_matches",
        "away_matches",
        "wins",
        "draws",
        "losses",
        "points",
        "expected_points",
        "goals_for",
        "goals_against",
        "goal_difference",
        "xg",
        "xga",
        "xg_difference",
        "np_xg",
        "np_xga",
        "np_xg_against",
        "np_xg_difference",
        "deep_completions",
        "deep_completions_allowed",
        "ppda",
        "ppda_allowed",
    ]
    out = out[[c for c in ordered_cols if c in out.columns]]
    sort_cols = [c for c in ["points", "goal_difference", "goals_for", "team"] if c in out.columns]
    if sort_cols:
        sort_asc = {"points": False, "goal_difference": False, "goals_for": False, "team": True}
        out = out.sort_values(sort_cols, ascending=[sort_asc[c] for c in sort_cols]).reset_index(drop=True)
    return out, team_missing


def apply_player_id_cleaning(
    df: pd.DataFrame,
    *,
    player_lookup: Dict[str, str],
    ambiguous_lookup_keys: set[str],
    override_lookup: Optional[Dict[str, set[str]]] = None,
) -> pd.DataFrame:
    out = df.copy()
    if "player" not in out.columns:
        return out

    out["__player_key"] = out["player"].apply(normalize_player_key)
    if "player_id" in out.columns:
        out["understat_player_id"] = out["player_id"]
    else:
        out["understat_player_id"] = pd.NA

    def _map_player_id(key: str) -> Optional[str]:
        if not key or key in ambiguous_lookup_keys:
            return None
        return player_lookup.get(key)

    out["player_id"] = out["__player_key"].map(_map_player_id)
    if override_lookup:
        def _map_override_player_id(player_name: Any) -> Optional[str]:
            keys = override_candidate_keys(_safe_str(player_name))
            matched_keys = [k for k in keys if k in override_lookup]
            if not matched_keys:
                return None
            ids: set[str] = set()
            for mk in matched_keys:
                ids.update(override_lookup[mk])
            ids_sorted = sorted(ids)
            if len(ids_sorted) == 1:
                return ids_sorted[0]
            return None

        out["__override_player_id"] = out["player"].apply(_map_override_player_id)
        use_override = out["player_id"].isna() & out["__override_player_id"].notna()
        out.loc[use_override, "player_id"] = out.loc[use_override, "__override_player_id"]

    out["player_id_missing"] = out["player_id"].isna()
    return out


def add_master_fpl_compare(
    df: pd.DataFrame,
    *,
    master_fpl: Dict[str, Any],
    season_long: str,
) -> pd.DataFrame:
    out = df.copy()
    if "fpl_pos" not in out.columns:
        return out
    if "player_id" not in out.columns:
        out["master_fpl_pos"] = pd.NA
        out["fpl_pos_master_match"] = pd.NA
        return out

    season_short = season_long_to_short(season_long)

    def _master_pos(pid: Any) -> Optional[str]:
        if pd.isna(pid):
            return None
        rec = master_fpl.get(str(pid))
        if not rec:
            return None
        career = rec.get("career", {})
        season_blob = career.get(season_short, {})
        pos = season_blob.get("fpl_position")
        if pos in {"GK", "DEF", "MID", "FWD", "SUB"}:
            return pos
        return None

    out["master_fpl_pos"] = out["player_id"].apply(_master_pos)
    both = out["fpl_pos"].notna() & out["master_fpl_pos"].notna()
    out["fpl_pos_master_match"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[both, "fpl_pos_master_match"] = (
        out.loc[both, "fpl_pos"].astype(str) == out.loc[both, "master_fpl_pos"].astype(str)
    )
    return out


def build_fpl_mode_maps(player_match_df: pd.DataFrame) -> Tuple[Dict[Tuple[str, str, str], str], Dict[Tuple[str, str, str], str]]:
    if player_match_df.empty or "fpl_pos" not in player_match_df.columns:
        return {}, {}

    work = player_match_df.copy()
    work["minutes"] = pd.to_numeric(work.get("minutes"), errors="coerce").fillna(0.0)
    work["fpl_pos"] = work["fpl_pos"].fillna("UNK").astype(str)
    work["__pos_rank"] = work["fpl_pos"].map(FPL_POS_PRECEDENCE).fillna(999).astype(int)

    by_id: Dict[Tuple[str, str, str], str] = {}
    id_work = work[work["player_id"].notna()].copy()
    if not id_work.empty:
        id_agg = (
            id_work.groupby(["league", "season", "player_id", "fpl_pos"], dropna=False, as_index=False)
            .agg(count=("fpl_pos", "size"), minutes=("minutes", "sum"))
        )
        id_agg["rank"] = id_agg["fpl_pos"].map(FPL_POS_PRECEDENCE).fillna(999).astype(int)
        id_agg = id_agg.sort_values(
            ["league", "season", "player_id", "count", "minutes", "rank"],
            ascending=[True, True, True, False, False, True],
        )
        id_top = id_agg.drop_duplicates(["league", "season", "player_id"], keep="first")
        by_id = {
            (str(r["league"]), str(r["season"]), str(r["player_id"])): str(r["fpl_pos"])
            for _, r in id_top.iterrows()
        }

    by_name: Dict[Tuple[str, str, str], str] = {}
    name_work = work[work["player_id"].isna()].copy()
    if not name_work.empty:
        name_agg = (
            name_work.groupby(["league", "season", "__player_key", "fpl_pos"], dropna=False, as_index=False)
            .agg(count=("fpl_pos", "size"), minutes=("minutes", "sum"))
        )
        name_agg["rank"] = name_agg["fpl_pos"].map(FPL_POS_PRECEDENCE).fillna(999).astype(int)
        name_agg = name_agg.sort_values(
            ["league", "season", "__player_key", "count", "minutes", "rank"],
            ascending=[True, True, True, False, False, True],
        )
        name_top = name_agg.drop_duplicates(["league", "season", "__player_key"], keep="first")
        by_name = {
            (str(r["league"]), str(r["season"]), str(r["__player_key"])): str(r["fpl_pos"])
            for _, r in name_top.iterrows()
        }

    return by_id, by_name


def process_player_match(
    df: pd.DataFrame,
    *,
    league_std: str,
    season_long: str,
    player_lookup: Dict[str, str],
    ambiguous_lookup_keys: set[str],
    override_lookup: Dict[str, set[str]],
    master_fpl: Dict[str, Any],
    teams_name_to_code: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = apply_common_league_season(df, league_std=league_std, season_long=season_long)
    out = apply_player_id_cleaning(
        out,
        player_lookup=player_lookup,
        ambiguous_lookup_keys=ambiguous_lookup_keys,
        override_lookup=override_lookup,
    )
    if "home_away" in out.columns:
        out["venue"] = (
            out["home_away"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"HOME": "H", "AWAY": "A"})
        )
    else:
        out["venue"] = pd.NA
    out["fpl_pos"] = out.get("position", pd.Series(pd.NA, index=out.index)).apply(map_player_match_pos)
    out["position_unmapped"] = out["fpl_pos"].eq("UNK")
    out = add_master_fpl_compare(out, master_fpl=master_fpl, season_long=season_long)
    out = standardize_team_and_game_codes(out, teams_name_to_code=teams_name_to_code)

    ordered_cols = [
        "league",
        "season",
        "league_id",
        "season_id",
        "game_id",
        "game",
        "team",
        "player",
        "player_id",
        "understat_player_id",
        "venue",
        "position",
        "fpl_pos",
        "master_fpl_pos",
        "fpl_pos_master_match",
        "minutes",
        "roster_id",
        "position_id",
        "roster_in",
        "roster_out",
        "goals",
        "own_goals",
        "assists",
        "xa",
        "shots",
        "key_passes",
        "xg",
        "xg_chain",
        "xg_buildup",
        "yellow_cards",
        "red_cards",
        "player_id_missing",
        "position_unmapped",
        "__player_key",
    ]
    out = out[[c for c in ordered_cols if c in out.columns]]

    unknown_pos = out.loc[out["position_unmapped"], ["league", "season", "position"]].copy()
    mism = out.loc[out["fpl_pos_master_match"] == False, ["league", "season", "player", "player_id", "fpl_pos", "master_fpl_pos"]].copy()  # noqa: E712
    return out, unknown_pos, mism


def process_player_season(
    df: pd.DataFrame,
    *,
    league_std: str,
    season_long: str,
    player_lookup: Dict[str, str],
    ambiguous_lookup_keys: set[str],
    override_lookup: Dict[str, set[str]],
    master_fpl: Dict[str, Any],
    mode_by_id: Dict[Tuple[str, str, str], str],
    mode_by_name: Dict[Tuple[str, str, str], str],
    teams_name_to_code: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = apply_common_league_season(df, league_std=league_std, season_long=season_long)
    out = apply_player_id_cleaning(
        out,
        player_lookup=player_lookup,
        ambiguous_lookup_keys=ambiguous_lookup_keys,
        override_lookup=override_lookup,
    )

    def _derive_fpl_pos(row: pd.Series) -> Optional[str]:
        lg = str(row.get("league", ""))
        ss = str(row.get("season", ""))
        pid = row.get("player_id")
        if pd.notna(pid):
            pos = mode_by_id.get((lg, ss, str(pid)))
            if pos:
                return pos
        key = str(row.get("__player_key", ""))
        if key:
            return mode_by_name.get((lg, ss, key))
        return None

    out["fpl_pos"] = out.apply(_derive_fpl_pos, axis=1)
    out = add_master_fpl_compare(out, master_fpl=master_fpl, season_long=season_long)
    out = standardize_team_and_game_codes(out, teams_name_to_code=teams_name_to_code)

    ordered_cols = [
        "league",
        "season",
        "league_id",
        "season_id",
        "team",
        "player",
        "player_id",
        "understat_player_id",
        "position",
        "fpl_pos",
        "master_fpl_pos",
        "fpl_pos_master_match",
        "matches",
        "minutes",
        "goals",
        "xg",
        "np_goals",
        "np_xg",
        "assists",
        "xa",
        "shots",
        "key_passes",
        "yellow_cards",
        "red_cards",
        "xg_chain",
        "xg_buildup",
        "player_id_missing",
    ]
    out = out[[c for c in ordered_cols if c in out.columns]]
    mism = out.loc[out["fpl_pos_master_match"] == False, ["league", "season", "player", "player_id", "fpl_pos", "master_fpl_pos"]].copy()  # noqa: E712
    return out, mism


def process_generic_player_file(
    df: pd.DataFrame,
    *,
    league_std: str,
    season_long: str,
    player_lookup: Dict[str, str],
    ambiguous_lookup_keys: set[str],
    override_lookup: Dict[str, set[str]],
    teams_name_to_code: Dict[str, str],
    team_lookup: Dict[str, str],
) -> pd.DataFrame:
    out = apply_common_league_season(df, league_std=league_std, season_long=season_long)
    out = apply_player_id_cleaning(
        out,
        player_lookup=player_lookup,
        ambiguous_lookup_keys=ambiguous_lookup_keys,
        override_lookup=override_lookup,
    )
    if "home_away" in out.columns:
        out["venue"] = (
            out["home_away"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"HOME": "H", "AWAY": "A"})
        )
        out = out.drop(columns=["home_away"])
    out = standardize_team_and_game_codes(out, teams_name_to_code=teams_name_to_code)
    if "team" in out.columns and "team_id" in out.columns:
        out["understat_team_id"] = out["team_id"]
        mapped = out["team"].astype(str).str.lower().map(team_lookup)
        out["team_id"] = mapped.where(mapped.notna(), out["team_id"])

    ordered_front = [
        "league",
        "season",
        "league_id",
        "season_id",
        "game_id",
        "understat_match_id",
        "understat_season",
        "date",
        "minute",
        "game",
        "team",
        "team_id",
        "understat_team_id",
        "venue",
        "player",
        "player_id",
        "understat_player_id",
    ]
    ordered_end = ["player_id_missing"]
    middle = [c for c in out.columns if c not in set(ordered_front + ordered_end)]
    out = out[[c for c in ordered_front if c in out.columns] + middle + [c for c in ordered_end if c in out.columns]]
    return out


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    drop_cols = [c for c in ["__player_key", "__override_player_id"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df.to_csv(out_path, index=False)


def _group_count(df: pd.DataFrame, by_cols: List[str], count_col_name: str = "count") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*by_cols, count_col_name])
    out = df.groupby(by_cols, dropna=False, as_index=False).size().rename(columns={"size": count_col_name})
    return out.sort_values(by_cols).reset_index(drop=True)


def run_clean(
    *,
    in_root: Path,
    out_root: Path,
    team_lookup_path: Path,
    teams_config_path: Path,
    player_lookup_path: Path,
    master_fpl_path: Path,
    overrides_path: Path,
) -> Dict[str, Any]:
    team_lookup_raw = json.loads(team_lookup_path.read_text(encoding="utf-8"))
    teams_config_raw: Dict[str, str] = {}
    if teams_config_path.exists():
        teams_config_raw = json.loads(teams_config_path.read_text(encoding="utf-8"))
    else:
        LOG.warning("teams config not found: %s", teams_config_path)
    player_lookup_raw = json.loads(player_lookup_path.read_text(encoding="utf-8"))
    master_fpl = json.loads(master_fpl_path.read_text(encoding="utf-8"))
    override_raw: Dict[str, str] = {}
    if overrides_path.exists():
        override_raw = json.loads(overrides_path.read_text(encoding="utf-8"))

    team_lookup = normalize_team_lookup(team_lookup_raw)
    teams_name_to_code, teams_name_to_display = build_teams_config_maps(teams_config_raw)
    player_lookup, ambiguous_player_keys = build_player_lookup(player_lookup_raw)
    override_lookup = build_override_lookup(override_raw)

    files_written = 0
    total_rows_in = 0
    total_rows_out = 0
    season_count = 0

    team_missing_all: List[pd.DataFrame] = []
    player_missing_all: List[pd.DataFrame] = []
    unknown_pos_all: List[pd.DataFrame] = []
    mismatch_all: List[pd.DataFrame] = []

    for src_league in MODERN_LEAGUE_FOLDERS:
        league_dir = in_root / src_league
        if not league_dir.exists():
            LOG.info("Skipping missing league dir: %s", league_dir)
            continue

        league_std = map_league_value(src_league)
        season_dirs = sorted([p for p in league_dir.iterdir() if p.is_dir()])
        for season_dir in season_dirs:
            season_long = season_to_long(season_dir.name)
            season_count += 1
            out_season_dir = out_root / league_std / season_long
            out_season_dir.mkdir(parents=True, exist_ok=True)

            file_map = {p.name: p for p in season_dir.glob("*.csv")}
            process_order = ["schedule.csv", "player_match.csv", "player_season.csv"]
            remaining = sorted([n for n in file_map.keys() if n not in set(process_order)])
            ordered_files = [n for n in process_order if n in file_map] + remaining

            mode_by_id: Dict[Tuple[str, str, str], str] = {}
            mode_by_name: Dict[Tuple[str, str, str], str] = {}

            for filename in ordered_files:
                src_path = file_map[filename]
                dst_path = out_season_dir / filename
                df = pd.read_csv(src_path)
                total_rows_in += len(df)

                if filename == "schedule.csv":
                    clean_df, team_missing = transform_schedule(
                        df,
                        league_std=league_std,
                        season_long=season_long,
                        team_lookup=team_lookup,
                        teams_name_to_code=teams_name_to_code,
                        teams_name_to_display=teams_name_to_display,
                    )
                    team_missing_all.append(team_missing.assign(file=filename))
                elif filename == "player_match.csv":
                    clean_df, unknown_pos, mism = process_player_match(
                        df,
                        league_std=league_std,
                        season_long=season_long,
                        player_lookup=player_lookup,
                        ambiguous_lookup_keys=ambiguous_player_keys,
                        override_lookup=override_lookup,
                        master_fpl=master_fpl,
                        teams_name_to_code=teams_name_to_code,
                    )
                    mode_by_id, mode_by_name = build_fpl_mode_maps(clean_df)
                    if not unknown_pos.empty:
                        unknown_pos_all.append(unknown_pos.assign(file=filename))
                    if not mism.empty:
                        mismatch_all.append(mism.assign(file=filename))
                elif filename == "player_season.csv":
                    clean_df, mism = process_player_season(
                        df,
                        league_std=league_std,
                        season_long=season_long,
                        player_lookup=player_lookup,
                        ambiguous_lookup_keys=ambiguous_player_keys,
                        override_lookup=override_lookup,
                        master_fpl=master_fpl,
                        mode_by_id=mode_by_id,
                        mode_by_name=mode_by_name,
                        teams_name_to_code=teams_name_to_code,
                    )
                    if not mism.empty:
                        mismatch_all.append(mism.assign(file=filename))
                elif filename == "team_match.csv":
                    clean_df, team_missing = transform_team_match(
                        df,
                        league_std=league_std,
                        season_long=season_long,
                        team_lookup=team_lookup,
                        teams_name_to_code=teams_name_to_code,
                    )
                    team_missing_all.append(team_missing.assign(file=filename))
                elif filename == "team_season.csv":
                    clean_df, team_missing = transform_team_season(
                        df,
                        league_std=league_std,
                        season_long=season_long,
                        team_lookup=team_lookup,
                        teams_name_to_code=teams_name_to_code,
                    )
                    team_missing_all.append(team_missing.assign(file=filename))
                else:
                    base_df = apply_common_league_season(df, league_std=league_std, season_long=season_long)
                    if "player" in base_df.columns:
                        clean_df = process_generic_player_file(
                            base_df,
                            league_std=league_std,
                            season_long=season_long,
                            player_lookup=player_lookup,
                            ambiguous_lookup_keys=ambiguous_player_keys,
                            override_lookup=override_lookup,
                            teams_name_to_code=teams_name_to_code,
                            team_lookup=team_lookup,
                        )
                    else:
                        clean_df = base_df

                if "player" in clean_df.columns and "player_id_missing" in clean_df.columns:
                    miss = clean_df.loc[clean_df["player_id_missing"], ["league", "season", "player"]].copy()
                    if not miss.empty:
                        player_missing_all.append(miss.assign(file=filename))

                save_csv(clean_df, dst_path)
                files_written += 1
                total_rows_out += len(clean_df)

    audit_dir = out_root / "_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    team_missing_df = (
        pd.concat(team_missing_all, ignore_index=True)
        if team_missing_all
        else pd.DataFrame(columns=["league", "season", "team_code", "file"])
    )
    team_missing_agg = _group_count(team_missing_df, ["league", "season", "team_code"], "missing_rows")
    team_missing_agg.to_csv(audit_dir / "team_code_lookup_missing.csv", index=False)

    player_missing_df = (
        pd.concat(player_missing_all, ignore_index=True)
        if player_missing_all
        else pd.DataFrame(columns=["league", "season", "file", "player"])
    )
    player_missing_agg = _group_count(player_missing_df, ["league", "season", "file", "player"], "missing_rows")
    if len(override_lookup) > 0 and not player_missing_agg.empty:
        player_missing_agg = annotate_missing_with_overrides(player_missing_agg, override_lookup)
    player_missing_agg.to_csv(audit_dir / "player_lookup_missing.csv", index=False)

    unknown_pos_df = (
        pd.concat(unknown_pos_all, ignore_index=True)
        if unknown_pos_all
        else pd.DataFrame(columns=["league", "season", "position", "file"])
    )
    unknown_pos_agg = _group_count(unknown_pos_df, ["league", "season", "file", "position"], "rows")
    unknown_pos_agg.to_csv(audit_dir / "unknown_position_codes.csv", index=False)

    mismatch_df = (
        pd.concat(mismatch_all, ignore_index=True)
        if mismatch_all
        else pd.DataFrame(columns=["league", "season", "player", "player_id", "fpl_pos", "master_fpl_pos", "file"])
    )
    mismatch_df.to_csv(audit_dir / "fpl_pos_compare_mismatch.csv", index=False)

    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "in_root": str(in_root),
        "out_root": str(out_root),
        "files_written": files_written,
        "seasons_processed": season_count,
        "rows_in": int(total_rows_in),
        "rows_out": int(total_rows_out),
        "team_code_missing_rows": int(len(team_missing_df)),
        "team_code_missing_groups": int(len(team_missing_agg)),
        "player_missing_rows": int(len(player_missing_df)),
        "player_missing_groups": int(len(player_missing_agg)),
        "player_missing_override_match_groups": int(
            player_missing_agg["override_match_found"].sum()
            if "override_match_found" in player_missing_agg.columns
            else 0
        ),
        "player_missing_override_unique_groups": int(
            player_missing_agg["override_unique_match"].sum()
            if "override_unique_match" in player_missing_agg.columns
            else 0
        ),
        "unknown_position_rows": int(len(unknown_pos_df)),
        "unknown_position_groups": int(len(unknown_pos_agg)),
        "fpl_pos_mismatch_rows": int(len(mismatch_df)),
        "ambiguous_player_lookup_keys": int(len(ambiguous_player_keys)),
    }
    (audit_dir / "clean_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser("Clean raw Understat CSV files")
    parser.add_argument("--in-root", type=Path, default=Path("data/raw/understat"))
    parser.add_argument("--out-root", type=Path, default=Path("data/processed/understat"))
    parser.add_argument("--team-lookup", type=Path, default=Path("data/processed/registry/_id_lookup_teams.json"))
    parser.add_argument("--teams-config", type=Path, default=Path("data/config/teams.json"))
    parser.add_argument("--player-lookup", type=Path, default=Path("data/processed/registry/_id_lookup_players.json"))
    parser.add_argument("--master-fpl", type=Path, default=Path("data/processed/registry/master_fpl.json"))
    parser.add_argument("--overrides", type=Path, default=Path("data/processed/registry/overrides.json"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    init_logger(verbose=args.verbose)
    summary = run_clean(
        in_root=args.in_root,
        out_root=args.out_root,
        team_lookup_path=args.team_lookup,
        teams_config_path=args.teams_config,
        player_lookup_path=args.player_lookup,
        master_fpl_path=args.master_fpl,
        overrides_path=args.overrides,
    )
    LOG.info("Clean complete: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
