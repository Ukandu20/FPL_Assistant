# clean_and_enrich.py
from __future__ import annotations

import argparse
import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from unidecode import unidecode

from fpl_assistant.canonical.identity import stable_canonical_id
from fpl_assistant.providers.fpl.paths import DEFAULT_FPL_LEAGUE, league_scoped_root

# Prefer rapidfuzz; fallback to fuzzywuzzy
try:
    from rapidfuzz import fuzz as rf_fuzz
    _USE_RAPIDFUZZ = True
except Exception:
    try:
        from fuzzywuzzy import fuzz as fw_fuzz  # type: ignore
        _USE_RAPIDFUZZ = False
    except Exception:
        fw_fuzz = None
        _USE_RAPIDFUZZ = False

# ───────────────────────── Canonicalisation & helpers ─────────────────────────

NAME_SUFFIX_RE = re.compile(r"_[0-9]+$")  # foo_123 → foo
FPL_POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
FBREF_TO_FPL_POS = {"GK": "GKP", "DF": "DEF", "MF": "MID", "FW": "FWD"}
FPL_POS_ALIASES = {
    "1": "GKP", "GK": "GKP", "GKP": "GKP",
    "2": "DEF", "DF": "DEF", "DEF": "DEF",
    "3": "MID", "MF": "MID", "MID": "MID",
    "4": "FWD", "FW": "FWD", "FWD": "FWD",
}
FPL_TO_FBREF_POS = {"GKP": "GK", "DEF": "DF", "MID": "MF", "FWD": "FW"}
SEASON_CUMULATIVE_COLUMNS = {
    "assists", "bonus", "bps", "clean_sheets", "creativity",
    "expected_assists", "expected_goal_involvements", "expected_goals",
    "expected_goals_conceded", "goals_conceded", "goals_scored",
    "ict_index", "influence", "minutes", "own_goals", "penalties_missed",
    "penalties_saved", "red_cards", "saves", "starts", "threat",
    "total_points", "yellow_cards",
}
PLAYER_SEASON_STAT_COLUMNS = [
    "blocks",
    "interceptions",
    "clearances",
    "tackles_won",
    "recoveries",
    "defcon",
    "xg",
    "xa",
]
PLAYER_SEASON_PROVENANCE_COLUMNS = [
    "defensive_stats_source",
    "expected_stats_source",
    "stats_coverage_status",
]
GOALKEEPER_SEASON_STAT_COLUMNS = [
    "shots_on_target_against",
    "saves",
    "goals_against",
    "save_pct",
    "penalties_faced",
    "penalties_allowed",
    "penalties_saved",
    "penalties_missed",
    "penalty_save_pct",
]
GOALKEEPER_PROVENANCE_COLUMNS = [
    "goalkeeper_stats_source",
    "goalkeeper_stats_coverage",
]
MODERN_PLAYER_STATS_START_SEASON = 2025
PLAYER_BRIDGE_COLUMNS = [
    "entity_type",
    "provider",
    "provider_id",
    "provider_name",
    "canonical_id",
    "valid_from",
    "valid_to",
    "match_method",
    "match_confidence",
    "review_status",
]

def canonical(s: str) -> str:
    """Lowercase, accent-fold, treat separators as spaces, strip punctuation, squeeze."""
    s = s or ""
    s = (
        s.replace("|", " ")
         .replace("_", " ")
         .replace("-", " ")
    )
    s = NAME_SUFFIX_RE.sub("", s)
    s = unidecode(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)     # punctuation → space
    s = " ".join(s.split())            # squeeze
    return s

def season_longform(s: str) -> str:
    """'2019-20' → '2019-2020'; '19-20' → '2019-2020'; pass-through if already long."""
    s = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}", s):
        start = int(s[:4])
        end   = int(str(start)[:2] + s[-2:])
        return f"{start}-{end}"
    if re.fullmatch(r"\d{2}-\d{2}", s):
        start = 2000 + int(s[:2])
        end   = 2000 + int(s[-2:])
        return f"{start}-{end}"
    return s

def season_shortform(long_s: str) -> str:
    """'2019-2020' → '2019-20'."""
    if re.fullmatch(r"\d{4}-\d{4}", long_s):
        start = long_s[:4]
        end   = long_s[-2:]
        return f"{start}-{end}"
    return long_s

def read_json_flex(p: Path) -> dict:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return json.loads(p.read_text(encoding=enc))
        except Exception:
            pass
    return json.loads(p.read_text())

def read_csv_flex(p: Path) -> pd.DataFrame:
    last: Optional[Exception] = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception as e:
            last = e
    raise last or RuntimeError(f"Failed to read {p}")

def write_json_utf8(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

def write_csv_utf8(p: Path, df: pd.DataFrame) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8")


def atomic_write_json_utf8(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_csv_utf8(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def normalized_provider_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def normalise_fpl_position(value) -> Optional[str]:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return FPL_POS_ALIASES.get(str(value).strip().upper())


def load_team_id_lookup(path: Optional[Path]) -> Dict[str, str]:
    if not path or not path.is_file():
        return {}
    raw = read_json_flex(path)
    return {
        str(code).strip().upper(): str(team_id)
        for code, team_id in raw.items()
        if team_id
    }


def attach_fpl_context(df: pd.DataFrame, season_dir: Path) -> pd.DataFrame:
    """Attach official element/team/position fields omitted by cleaned_players.csv."""
    players_path = season_dir / "players_raw.csv"
    teams_path = season_dir / "season" / "teams.csv"
    if not players_path.is_file():
        logging.warning("[%s] missing companion player file: %s", season_dir.name, players_path)
        return df

    players = read_csv_flex(players_path)
    required = {"first_name", "second_name", "id", "team", "element_type"}
    if not required.issubset(players.columns):
        logging.warning(
            "[%s] players_raw.csv lacks required context columns: %s",
            season_dir.name,
            sorted(required - set(players.columns)),
        )
        return df

    team_codes: Dict[int, str] = {}
    if teams_path.is_file():
        teams = read_csv_flex(teams_path)
        if {"id", "short_name"}.issubset(teams.columns):
            team_codes = {
                int(team_id): str(short_name).strip().upper()
                for team_id, short_name in zip(teams["id"], teams["short_name"])
                if pd.notna(team_id) and pd.notna(short_name)
            }

    optional = [
        c
        for c in (
            "web_name",
            "status",
            "can_select",
            "can_transact",
            "news",
            "code",
            "opta_code",
        )
        if c in players.columns
    ]
    context = players[
        ["first_name", "second_name", "id", "team", "element_type", *optional]
    ].copy()
    context["_fpl_join_key"] = context.apply(
        lambda row: canonical(f"{row['first_name']} {row['second_name']}"), axis=1
    )
    duplicate_keys = context["_fpl_join_key"].duplicated(keep=False)
    if duplicate_keys.any():
        logging.warning(
            "[%s] duplicate official player names=%d; keeping first context row",
            season_dir.name,
            int(duplicate_keys.sum()),
        )
        context = context.drop_duplicates("_fpl_join_key", keep="first")

    context = context.rename(
        columns={
            "id": "fpl_element_id",
            "team": "fpl_team_numeric_id",
            "element_type": "fpl_element_type",
            "code": "fpl_code",
            "opta_code": "fpl_opta_code",
        }
    )
    context["fpl_team"] = context["fpl_team_numeric_id"].map(team_codes)

    result = df.copy()
    result["_fpl_join_key"] = result.apply(
        lambda row: canonical(
            f"{row.get('first_name', '')} {row.get('second_name', '')}"
        ),
        axis=1,
    )
    result = result.merge(
        context.drop(columns=["first_name", "second_name"]),
        on="_fpl_join_key",
        how="left",
        validate="many_to_one",
    )
    return result.drop(columns=["_fpl_join_key"])


def reset_preseason_carryover(
    df: pd.DataFrame, season_dir: Path
) -> tuple[pd.DataFrame, list[str]]:
    """Reset prior-season totals when every target-season fixture is unstarted."""
    fixtures_path = season_dir / "season" / "fixtures.csv"
    if not fixtures_path.is_file():
        return df, []

    fixtures = read_csv_flex(fixtures_path)
    state_columns = [
        column
        for column in ("started", "finished", "finished_provisional")
        if column in fixtures.columns
    ]
    if fixtures.empty or not state_columns:
        return df, []

    has_started = any(
        fixtures[column].astype("string").str.lower().eq("true").any()
        for column in state_columns
    )
    if has_started:
        return df, []

    candidates = sorted(SEASON_CUMULATIVE_COLUMNS.intersection(df.columns))
    carried = [
        column
        for column in candidates
        if pd.to_numeric(df[column], errors="coerce").fillna(0).ne(0).any()
    ]
    if not carried:
        return df, []

    result = df.copy()
    result.loc[:, candidates] = 0
    result["season_data_status"] = "preseason_roster"
    result["performance_data_status"] = "prior_season_carryover_reset"
    return result, carried


def _provider_league_root(root: Optional[Path], league: str) -> Optional[Path]:
    if root is None:
        return None
    path = Path(root)
    return path if path.name == league else path / league


def _normalise_join_value(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _source_lookup(
    source: pd.DataFrame,
    key_columns: list[str],
    metric_mapping: Dict[str, str],
    source_name: str,
) -> tuple[Dict[tuple, dict], list[dict]]:
    """Build a unique provider lookup without allowing row multiplication."""
    required = {*key_columns, *metric_mapping}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")

    prepared = source[[*key_columns, *metric_mapping]].copy()
    for column in key_columns:
        prepared[column] = prepared[column].map(_normalise_join_value)
        if column == "team":
            prepared[column] = prepared[column].str.upper()

    duplicate_mask = prepared.duplicated(key_columns, keep=False)
    duplicate_keys = (
        prepared.loc[duplicate_mask, key_columns]
        .drop_duplicates()
        .to_dict("records")
    )
    if duplicate_keys:
        raise ValueError(
            f"{source_name} has duplicate join keys; refusing a row-multiplying "
            f"join: {duplicate_keys[:10]}"
        )

    lookup: Dict[tuple, dict] = {}
    for record in prepared.to_dict("records"):
        key = tuple(record[column] for column in key_columns)
        if any(value is None for value in key):
            continue
        lookup[key] = {
            output: pd.to_numeric(record[source_column], errors="coerce")
            for source_column, output in metric_mapping.items()
        }
    return lookup, duplicate_keys


def _aggregate_modern_goalkeepers(
    path: Path,
) -> tuple[pd.DataFrame, int, list[str]]:
    """Aggregate valid WhoScored keeper appearances from player-match rows."""
    matches = read_csv_flex(path)
    required = {
        "player_id",
        "fpl_pos",
        "minutes",
        "shots_on_target_against",
        "goals_against",
        "saves",
        "penalties_faced",
    }
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(
            "whoscored.player_match.keepers is missing required columns: "
            f"{missing}"
        )

    minutes = pd.to_numeric(matches["minutes"], errors="coerce")
    keepers = matches.loc[
        matches["fpl_pos"].astype("string").eq("GKP") & minutes.gt(0)
    ].copy()
    keepers["minutes"] = minutes.loc[keepers.index]
    match_key = next(
        (
            columns
            for columns in (
                ["provider_match_id", "team_id"],
                ["match_id", "team_id"],
                ["game", "team"],
            )
            if set(columns).issubset(keepers.columns)
        ),
        None,
    )
    shared_appearance_rows = 0
    shared_player_ids: list[str] = []
    if match_key:
        shared = keepers.groupby(match_key, dropna=False)["player_id"].transform("nunique")
        shared_appearance_rows = int(shared.gt(1).sum())
        shared_player_ids = sorted(
            keepers.loc[shared.gt(1), "player_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    additive = [
        "shots_on_target_against",
        "goals_against",
        "saves",
        "penalties_faced",
    ]
    for column in additive:
        keepers[column] = pd.to_numeric(keepers[column], errors="coerce")
    aggregated = (
        keepers.groupby("player_id", as_index=False, dropna=False)[additive]
        .sum(min_count=1)
    )
    aggregated["save_pct"] = (
        100
        * aggregated["saves"]
        / aggregated["shots_on_target_against"].where(
            aggregated["shots_on_target_against"].gt(0)
        )
    )
    return aggregated, shared_appearance_rows, shared_player_ids


def enrich_player_season_stats(
    players: pd.DataFrame,
    season: str,
    league: str = DEFAULT_FPL_LEAGUE,
    fbref_root: Optional[Path] = None,
    whoscored_root: Optional[Path] = None,
    understat_root: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict]:
    """Attach position-aware CBIT/DEFCON ingredients and xG/xA."""
    result = players.copy()
    row_count_before = len(result)
    season_full = season_longform(season)
    start_year = int(season_full.split("-", 1)[0])

    for column in PLAYER_SEASON_STAT_COLUMNS:
        result[column] = pd.Series(pd.NA, index=result.index, dtype="Float64")
    for column in PLAYER_SEASON_PROVENANCE_COLUMNS:
        result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
    for column in GOALKEEPER_SEASON_STAT_COLUMNS:
        result[column] = pd.Series(pd.NA, index=result.index, dtype="Float64")
    for column in GOALKEEPER_PROVENANCE_COLUMNS:
        result[column] = pd.Series(pd.NA, index=result.index, dtype="string")

    fbref_league = _provider_league_root(fbref_root, league)
    whoscored_league = _provider_league_root(whoscored_root, league)
    understat_league = _provider_league_root(understat_root, league)

    if start_year >= MODERN_PLAYER_STATS_START_SEASON:
        defense_path = (
            whoscored_league / season_full / "player_season" / "defense.csv"
            if whoscored_league is not None
            else None
        )
        expected_path = (
            understat_league / season_full / "player_season.csv"
            if understat_league is not None
            else None
        )
        defense_keys = ["player_id"]
        defense_mapping = {
            "blocks": "blocks",
            "interceptions": "interceptions",
            "clearances": "clearances",
            "tackles_won": "tackles_won",
            "recoveries": "recoveries",
        }
        recoveries_path = None
        expected_keys = ["player_id"]
        expected_mapping = {"xg": "xg", "xa": "xa"}
        goalkeeper_path = (
            whoscored_league / season_full / "player_match" / "keepers.csv"
            if whoscored_league is not None
            else None
        )
        goalkeeper_keys = ["player_id"]
        goalkeeper_mapping = {
            "shots_on_target_against": "shots_on_target_against",
            "saves": "saves",
            "goals_against": "goals_against",
            "save_pct": "save_pct",
            "penalties_faced": "penalties_faced",
        }
        defense_source = "whoscored.player_season.defense"
        expected_source = "understat.player_season"
        goalkeeper_source = "whoscored.player_match.keepers.filtered_appearances"
    else:
        defense_path = (
            fbref_league / season_full / "player_season" / "defense.csv"
            if fbref_league is not None
            else None
        )
        expected_path = (
            fbref_league / season_full / "player_season" / "standard.csv"
            if fbref_league is not None
            else None
        )
        defense_keys = ["player_id", "team"]
        defense_mapping = {
            "blocks": "blocks",
            "int": "interceptions",
            "clr": "clearances",
            "tklw": "tackles_won",
        }
        recoveries_path = (
            fbref_league / season_full / "player_season" / "misc.csv"
            if fbref_league is not None
            else None
        )
        expected_keys = ["player_id", "team"]
        expected_mapping = {"xg": "xg", "xag": "xa"}
        goalkeeper_path = (
            fbref_league / season_full / "player_season" / "keeper.csv"
            if fbref_league is not None
            else None
        )
        goalkeeper_keys = ["player_id", "team"]
        goalkeeper_mapping = {
            "sota": "shots_on_target_against",
            "saves": "saves",
            "ga": "goals_against",
            "save": "save_pct",
            "pkatt": "penalties_faced",
            "pka": "penalties_allowed",
            "pksv": "penalties_saved",
            "pkm": "penalties_missed",
            "save_save": "penalty_save_pct",
        }
        defense_source = "fbref.player_season.defense"
        expected_source = "fbref.player_season.standard"
        goalkeeper_source = "fbref.player_season.keeper"

    source_paths = {
        "defensive": str(defense_path) if defense_path is not None else None,
        "expected": str(expected_path) if expected_path is not None else None,
        "goalkeeper": str(goalkeeper_path) if goalkeeper_path is not None else None,
    }
    source_available = {
        "defensive": bool(defense_path and defense_path.is_file()),
        "expected": bool(expected_path and expected_path.is_file()),
        "goalkeeper": bool(goalkeeper_path and goalkeeper_path.is_file()),
    }

    def _player_keys(columns: list[str]) -> list[tuple]:
        keys = []
        for record in result.to_dict("records"):
            values = []
            for column in columns:
                value = _normalise_join_value(record.get(column))
                if column == "team" and value is not None:
                    value = value.upper()
                values.append(value)
            keys.append(tuple(values))
        return keys

    matched = {"defensive": 0, "expected": 0, "goalkeeper": 0}
    shared_goalkeeper_appearance_rows = 0
    shared_goalkeeper_player_ids: list[str] = []
    if source_available["defensive"]:
        defense = read_csv_flex(defense_path)
        lookup, _ = _source_lookup(
            defense, defense_keys, defense_mapping, defense_source
        )
        records = [lookup.get(key) for key in _player_keys(defense_keys)]
        matched["defensive"] = sum(record is not None for record in records)
        for index, record in zip(result.index, records):
            if record is None:
                continue
            for column, value in record.items():
                result.at[index, column] = value
            result.at[index, "defensive_stats_source"] = defense_source

        if recoveries_path is not None and recoveries_path.is_file():
            recoveries = read_csv_flex(recoveries_path)
            recoveries_lookup, _ = _source_lookup(
                recoveries,
                defense_keys,
                {"recov": "recoveries"},
                "fbref.player_season.misc",
            )
            recovery_records = [
                recoveries_lookup.get(key) for key in _player_keys(defense_keys)
            ]
            for index, record in zip(result.index, recovery_records):
                if record is not None:
                    result.at[index, "recoveries"] = record["recoveries"]

    if source_available["expected"]:
        expected = read_csv_flex(expected_path)
        lookup, _ = _source_lookup(
            expected, expected_keys, expected_mapping, expected_source
        )
        records = [lookup.get(key) for key in _player_keys(expected_keys)]
        matched["expected"] = sum(record is not None for record in records)
        for index, record in zip(result.index, records):
            if record is None:
                continue
            for column, value in record.items():
                result.at[index, column] = value
            result.at[index, "expected_stats_source"] = expected_source

    if source_available["goalkeeper"]:
        if start_year >= MODERN_PLAYER_STATS_START_SEASON:
            (
                goalkeeper,
                shared_goalkeeper_appearance_rows,
                shared_goalkeeper_player_ids,
            ) = (
                _aggregate_modern_goalkeepers(goalkeeper_path)
            )
        else:
            goalkeeper = read_csv_flex(goalkeeper_path)
        lookup, _ = _source_lookup(
            goalkeeper,
            goalkeeper_keys,
            goalkeeper_mapping,
            goalkeeper_source,
        )
        records = [lookup.get(key) for key in _player_keys(goalkeeper_keys)]
        matched["goalkeeper"] = sum(record is not None for record in records)
        for index, record in zip(result.index, records):
            if record is None:
                continue
            for column, value in record.items():
                result.at[index, column] = value
            result.at[index, "goalkeeper_stats_source"] = goalkeeper_source

    minutes = pd.to_numeric(
        result.get("minutes", pd.Series(index=result.index, dtype="float64")),
        errors="coerce",
    )
    zero_minutes = minutes.eq(0)
    base_defcon_columns = [
        "clearances",
        "blocks",
        "interceptions",
        "tackles_won",
    ]
    defensive_columns = [*base_defcon_columns, "recoveries"]
    expected_columns = ["xg", "xa"]
    goalkeeper_core_columns = [
        "shots_on_target_against",
        "saves",
        "goals_against",
        "save_pct",
        "penalties_faced",
    ]
    normalized_positions = result.get(
        "fpl_pos", pd.Series(index=result.index, dtype="string")
    ).map(normalise_fpl_position).astype("string")
    goalkeeper_mask = normalized_positions.eq("GKP").fillna(False) | result[
        "goalkeeper_stats_source"
    ].notna()
    known_position = normalized_positions.notna()

    if source_available["defensive"]:
        defensive_zero_fill = zero_minutes & result[defensive_columns].isna().all(axis=1)
        result.loc[defensive_zero_fill, defensive_columns] = 0.0
        result.loc[defensive_zero_fill, "defensive_stats_source"] = (
            "derived.zero_minutes"
        )
    if source_available["expected"]:
        expected_zero_fill = zero_minutes & result[expected_columns].isna().all(axis=1)
        result.loc[expected_zero_fill, expected_columns] = 0.0
        result.loc[expected_zero_fill, "expected_stats_source"] = (
            "derived.zero_minutes"
        )
    if source_available["goalkeeper"]:
        goalkeeper_zero_fill = (
            goalkeeper_mask
            & zero_minutes
            & result[goalkeeper_core_columns].isna().all(axis=1)
        )
        goalkeeper_count_columns = [
            column
            for column in GOALKEEPER_SEASON_STAT_COLUMNS
            if column not in {"save_pct", "penalty_save_pct"}
        ]
        result.loc[goalkeeper_zero_fill, goalkeeper_count_columns] = 0.0
        result.loc[goalkeeper_zero_fill, "goalkeeper_stats_source"] = (
            "derived.zero_minutes"
        )

    base_defcon = result[base_defcon_columns].sum(axis=1, min_count=4)
    midfield_forward = normalized_positions.isin(["MID", "FWD"])
    result["defcon"] = base_defcon
    result.loc[midfield_forward, "defcon"] = (
        base_defcon + result["recoveries"]
    )
    defense_complete = result[base_defcon_columns].notna().all(axis=1) & (
        ~midfield_forward | result["recoveries"].notna()
    )
    expected_complete = result[expected_columns].notna().all(axis=1)
    any_metric = result[PLAYER_SEASON_STAT_COLUMNS].notna().any(axis=1)
    both_sources_missing = not (
        source_available["defensive"] or source_available["expected"]
    )

    result.loc[defense_complete & expected_complete, "stats_coverage_status"] = (
        "complete"
    )
    result.loc[
        ~(defense_complete & expected_complete) & any_metric,
        "stats_coverage_status",
    ] = "partial"
    if both_sources_missing:
        result.loc[:, "stats_coverage_status"] = "provider_data_unavailable"
    else:
        unresolved = result["stats_coverage_status"].isna()
        result.loc[unresolved & minutes.gt(0), "stats_coverage_status"] = (
            "unmatched_active"
        )
        result.loc[
            result["stats_coverage_status"].isna(), "stats_coverage_status"
        ] = "unavailable"

    goalkeeper_complete = result[goalkeeper_core_columns].notna().all(axis=1)
    goalkeeper_any = result[GOALKEEPER_SEASON_STAT_COLUMNS].notna().any(axis=1)
    result.loc[
        ~goalkeeper_mask & known_position, "goalkeeper_stats_coverage"
    ] = "not_applicable"
    result.loc[
        ~goalkeeper_mask & ~known_position, "goalkeeper_stats_coverage"
    ] = "unknown_position"
    result.loc[
        goalkeeper_mask & zero_minutes & source_available["goalkeeper"],
        "goalkeeper_stats_coverage",
    ] = "zero_minutes"
    result.loc[
        goalkeeper_mask & ~zero_minutes & goalkeeper_complete,
        "goalkeeper_stats_coverage",
    ] = "complete"
    shared_goalkeeper_mask = result.get(
        "player_id", pd.Series(index=result.index, dtype="string")
    ).astype("string").isin(shared_goalkeeper_player_ids)
    result.loc[
        goalkeeper_mask & shared_goalkeeper_mask & ~zero_minutes,
        "goalkeeper_stats_coverage",
    ] = "shared_match_warning"
    result.loc[
        goalkeeper_mask & ~zero_minutes & ~goalkeeper_complete & goalkeeper_any,
        "goalkeeper_stats_coverage",
    ] = "partial"
    if not source_available["goalkeeper"]:
        result.loc[goalkeeper_mask, "goalkeeper_stats_coverage"] = (
            "provider_data_unavailable"
        )
    else:
        result.loc[
            goalkeeper_mask
            & result["goalkeeper_stats_coverage"].isna()
            & minutes.gt(0),
            "goalkeeper_stats_coverage",
        ] = "unmatched_active"
        result.loc[
            goalkeeper_mask & result["goalkeeper_stats_coverage"].isna(),
            "goalkeeper_stats_coverage",
        ] = "unavailable"

    incomplete_active = minutes.gt(0) & ~(
        defense_complete & expected_complete
    )
    audit_columns = [
        column
        for column in ("player_id", "name", "team", "minutes")
        if column in result.columns
    ]
    audit = {
        "season": season_full,
        "league": league,
        "row_count_before": row_count_before,
        "row_count_after": len(result),
        "row_count_preserved": len(result) == row_count_before,
        "source_paths": source_paths,
        "source_available": source_available,
        "matched_rows": matched,
        "shared_goalkeeper_appearance_rows": shared_goalkeeper_appearance_rows,
        "shared_goalkeeper_player_ids": shared_goalkeeper_player_ids,
        "coverage_status_counts": {
            str(status): int(count)
            for status, count in result["stats_coverage_status"]
            .value_counts(dropna=False)
            .items()
        },
        "incomplete_active_count": int(incomplete_active.sum()),
        "incomplete_active_players": json.loads(
            result.loc[incomplete_active, audit_columns].to_json(
                orient="records"
            )
        ),
        "goalkeeper_coverage_counts": {
            str(status): int(count)
            for status, count in result["goalkeeper_stats_coverage"]
            .value_counts(dropna=False)
            .items()
        },
    }
    if len(result) != row_count_before:
        raise RuntimeError(
            f"[{season_full}] player-season enrichment changed row count from "
            f"{row_count_before} to {len(result)}"
        )
    return result, audit

# ───────────────────────── Season selection helpers ─────────────────────────

def _season_sort_key(p: Path) -> tuple[int, str]:
    """
    Sort directories by starting year when they look like seasons.
    Accepts 'YYYY-YY' or 'YYYY-YYYY'. Unknown formats come first.
    """
    name = p.name
    longf = season_longform(name)
    m = re.match(r"^(\d{4})-(\d{4})$", longf)
    if m:
        return (int(m.group(1)), longf)
    return (-1, name)

def _looks_like_season(name: str) -> bool:
    # allow 2-2, 4-2, 4-4 patterns
    return re.fullmatch(r"\d{2,4}-\d{2,4}", name) is not None

def _match_season_dir(all_dirs: List[Path], sel: str) -> Optional[Path]:
    """
    Try to match 'sel' against directory names using both short/long forms.
    """
    target_long = season_longform(sel)
    target_short = season_shortform(target_long)
    # direct name match first
    for d in all_dirs:
        if d.name == target_long or d.name == target_short:
            return d
    # compare longform of dir names
    for d in all_dirs:
        if season_longform(d.name) == target_long:
            return d
    return None

# ───────────────────────── Master & overrides ─────────────────────────

def load_fbref_master(master_path: Path) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """
    Return:
      pid2rec: player_id -> full master record
      key2pid: canonical(full name) -> player_id
    """
    data = read_json_flex(master_path)
    if isinstance(data, list):
        records = data
    else:
        records = []
        for pid, rec in data.items():
            rec = dict(rec)
            rec["player_id"] = rec.get("player_id") or pid
            records.append(rec)

    pid2rec: Dict[str, dict] = {}
    key2pid: Dict[str, str] = {}
    for rec in records:
        pid = rec.get("player_id") or rec.get("id")
        if not pid:
            continue
        pid2rec[pid] = rec
        master_name = rec.get("name") or ""
        if master_name:
            key2pid[canonical(master_name)] = pid
    logging.info("FBref master: %d players, %d canonical keys", len(pid2rec), len(key2pid))
    return pid2rec, key2pid

def load_overrides(path: Optional[Path]) -> Dict[str, str]:
    """
    Overrides map: canonical("first | last") -> player_id
    Keys may include spaces around '|'; we normalise them away.
    """
    if not path or not path.is_file():
        return {}
    raw = read_json_flex(path)
    out: Dict[str, str] = {}
    for k, v in raw.items():
        key = canonical(k.replace(" | ", " ").replace("|", " "))
        if isinstance(v, str):
            out[key] = v
        elif isinstance(v, dict) and v.get("id"):
            out[key] = str(v["id"])
    logging.info("Overrides loaded: %d entries", len(out))
    return out


def extend_player_name_lookup(
    key2pid: Dict[str, str],
    registry_root: Path,
) -> None:
    """Add unambiguous canonical names from the other player registries."""
    candidates: Dict[str, set[str]] = {}
    for key, player_id in key2pid.items():
        candidates.setdefault(key, set()).add(str(player_id))

    lookup_path = registry_root / "_id_lookup_players.json"
    if lookup_path.is_file():
        for name, player_id in read_json_flex(lookup_path).items():
            candidates.setdefault(canonical(name), set()).add(str(player_id))

    master_fpl_path = registry_root / "master_fpl.json"
    if master_fpl_path.is_file():
        master_fpl = read_json_flex(master_fpl_path)
        for player_id, record in master_fpl.items():
            if isinstance(record, Mapping) and record.get("name"):
                candidates.setdefault(canonical(record["name"]), set()).add(
                    str(player_id)
                )

    ambiguous = 0
    for name_key, player_ids in candidates.items():
        if len(player_ids) == 1:
            key2pid[name_key] = next(iter(player_ids))
        else:
            ambiguous += 1
    if ambiguous:
        logging.warning(
            "Player registry contains %d ambiguous canonical names; provider "
            "codes or overrides are required for those names.",
            ambiguous,
        )


def load_fpl_code_registry(
    registry_root: Path,
    processed_fpl_root: Path,
    *,
    target_season: str,
) -> tuple[Dict[str, str], set[str]]:
    """Load stable FPL-code identities from bridges and earlier seasons.

    The target season is deliberately excluded so a prior erroneous generated
    ID in that season cannot override the current 8-character policy.
    """
    code_to_pid: Dict[str, str] = {}
    generated_codes: set[str] = set()

    def add(code: Any, player_id: Any, *, generated: bool, source: str) -> None:
        provider_code = normalized_provider_code(code)
        canonical_id = "" if pd.isna(player_id) else str(player_id).strip()
        if not provider_code or not canonical_id:
            return
        prior = code_to_pid.get(provider_code)
        if prior and prior != canonical_id:
            raise ValueError(
                f"FPL provider code {provider_code} maps to both {prior} and "
                f"{canonical_id} ({source})."
            )
        code_to_pid[provider_code] = canonical_id
        if generated:
            generated_codes.add(provider_code)

    bridge_path = registry_root / "bridges" / "player_ids.csv"
    if bridge_path.is_file():
        bridges = pd.read_csv(bridge_path, dtype=str, keep_default_na=False)
        fpl = (
            bridges.loc[bridges["provider"].astype(str).str.lower().eq("fpl")]
            if "provider" in bridges
            else pd.DataFrame()
        )
        for row in fpl.to_dict("records"):
            add(
                row.get("provider_id"),
                row.get("canonical_id"),
                generated="generated" in str(row.get("match_method", "")).lower(),
                source=str(bridge_path),
            )

    target_start = int(season_longform(target_season)[:4])
    if processed_fpl_root.is_dir():
        for season_dir in sorted(path for path in processed_fpl_root.iterdir() if path.is_dir()):
            season_text = season_longform(season_dir.name)
            if not re.fullmatch(r"\d{4}-\d{4}", season_text):
                continue
            if int(season_text[:4]) >= target_start:
                continue
            roster_path = season_dir / "season" / "cleaned_players.csv"
            if not roster_path.is_file():
                continue
            roster = pd.read_csv(roster_path, dtype=str, low_memory=False)
            if not {"fpl_code", "player_id"} <= set(roster.columns):
                continue
            for row in roster.to_dict("records"):
                add(
                    row.get("fpl_code"),
                    row.get("player_id"),
                    generated=str(row.get("player_id_source", "")).startswith(
                        "generated_from_fpl_code"
                    ),
                    source=str(roster_path),
                )
    return code_to_pid, generated_codes


def register_generated_players(
    players: pd.DataFrame,
    registration_mask: pd.Series,
    *,
    season: str,
    league: str,
    registry_root: Path,
    compatibility_master_path: Path | None = None,
    pid2rec: Dict[str, dict] | None = None,
    key2pid: Dict[str, str] | None = None,
) -> pd.DataFrame:
    """Promote generated FPL identities into every maintained player registry."""
    audit_columns = [
        "player_id", "name", "fpl_code", "fpl_element_id", "team", "team_id",
        "fpl_pos", "season", "registry_status",
    ]
    selected = players.loc[registration_mask].copy()
    if selected.empty:
        return pd.DataFrame(columns=audit_columns)
    required = {"player_id", "name", "fpl_code", "team", "team_id", "fpl_pos"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"Generated player registration lacks columns: {missing}")
    selected["fpl_code"] = selected["fpl_code"].map(normalized_provider_code)
    selected = selected.drop_duplicates(["player_id", "fpl_code"])
    if selected["player_id"].duplicated().any():
        raise ValueError("A generated player ID is associated with multiple FPL codes.")
    if selected["fpl_code"].duplicated().any():
        raise ValueError("An FPL code is associated with multiple generated player IDs.")

    master_path = registry_root / "master_players.json"
    lookup_path = registry_root / "_id_lookup_players.json"
    master_fpl_path = registry_root / "master_fpl.json"
    bridge_path = registry_root / "bridges" / "player_ids.csv"
    master = read_json_flex(master_path) if master_path.is_file() else {}
    lookup = read_json_flex(lookup_path) if lookup_path.is_file() else {}
    master_fpl = read_json_flex(master_fpl_path) if master_fpl_path.is_file() else {}
    compatibility_master = (
        read_json_flex(compatibility_master_path)
        if compatibility_master_path and compatibility_master_path.is_file()
        else None
    )
    bridges = (
        pd.read_csv(bridge_path, dtype=str, keep_default_na=False)
        if bridge_path.is_file()
        else pd.DataFrame(columns=PLAYER_BRIDGE_COLUMNS)
    )
    for column in PLAYER_BRIDGE_COLUMNS:
        if column not in bridges:
            bridges[column] = ""
    bridges = bridges[PLAYER_BRIDGE_COLUMNS].copy()

    long_season = season_longform(season)
    short_season = season_shortform(long_season)
    audit_rows: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        player_id = str(row["player_id"]).strip()
        name = str(row["name"]).strip()
        name_key = canonical(name)
        fpl_code = normalized_provider_code(row["fpl_code"])
        existing_record = master.get(player_id)
        if existing_record and canonical(existing_record.get("name", "")) not in {"", name_key}:
            raise ValueError(
                f"Generated ID collision: {player_id} belongs to "
                f"{existing_record.get('name')!r}, not {name!r}."
            )
        existing_lookup_id = lookup.get(name_key)
        if existing_lookup_id and str(existing_lookup_id) != player_id:
            raise ValueError(
                f"Generated name collision: {name!r} already maps to "
                f"{existing_lookup_id}, not {player_id}."
            )
        provider_rows = bridges.loc[
            bridges["provider"].str.lower().eq("fpl")
            & bridges["provider_id"].eq(fpl_code)
        ]
        if not provider_rows.empty and provider_rows["canonical_id"].ne(player_id).any():
            raise ValueError(
                f"FPL code {fpl_code} already maps to another canonical player ID."
            )

        fpl_pos = str(row.get("fpl_pos", "") or "").strip().upper()
        team = str(row.get("team", "") or "").strip().upper()
        team_id = str(row.get("team_id", "") or "").strip()
        season_record = {
            "team": team,
            "team_id": team_id,
            "position": fpl_pos,
            "fpl_position": fpl_pos,
            "position_detail": "UNK",
            "league": league,
        }
        record = dict(existing_record or {})
        record["name"] = record.get("name") or name
        record.setdefault("nation", None)
        record.setdefault("born", None)
        record.setdefault("career", {})[long_season] = season_record
        master[player_id] = record
        lookup[name_key] = player_id

        fpl_record = dict(master_fpl.get(player_id) or {})
        fpl_record.update(
            {
                "first_name": row.get("first_name") if pd.notna(row.get("first_name")) else None,
                "second_name": row.get("second_name") if pd.notna(row.get("second_name")) else None,
                "name": fpl_record.get("name") or name,
                "player_id": player_id,
                "nation": fpl_record.get("nation"),
                "born": fpl_record.get("born"),
            }
        )
        fpl_record.setdefault("career", {})[short_season] = season_record
        master_fpl[player_id] = fpl_record
        if isinstance(compatibility_master, dict):
            compatibility_master[player_id] = dict(fpl_record)

        if provider_rows.empty:
            bridges.loc[len(bridges)] = {
                "entity_type": "player",
                "provider": "fpl",
                "provider_id": fpl_code,
                "provider_name": name,
                "canonical_id": player_id,
                "valid_from": long_season,
                "valid_to": "",
                "match_method": "generated_from_fpl_code",
                "match_confidence": "1.0",
                "review_status": "needs_review",
            }
        audit_rows.append(
            {
                "player_id": player_id,
                "name": name,
                "fpl_code": fpl_code,
                "fpl_element_id": row.get("fpl_element_id"),
                "team": team,
                "team_id": team_id,
                "fpl_pos": fpl_pos,
                "season": long_season,
                "registry_status": "registered",
            }
        )

    bridge_conflicts = bridges.groupby(["provider", "provider_id"])["canonical_id"].nunique()
    if (bridge_conflicts > 1).any():
        raise ValueError("Player provider bridge contains conflicting canonical IDs.")
    atomic_write_json_utf8(master_path, master)
    atomic_write_json_utf8(lookup_path, lookup)
    atomic_write_json_utf8(master_fpl_path, master_fpl)
    atomic_write_csv_utf8(
        bridge_path,
        bridges.sort_values(["provider", "provider_id"], kind="stable"),
    )
    if compatibility_master_path and isinstance(compatibility_master, dict):
        atomic_write_json_utf8(compatibility_master_path, compatibility_master)

    if pid2rec is not None:
        for row in audit_rows:
            pid2rec[row["player_id"]] = master[row["player_id"]]
    if key2pid is not None:
        for row in audit_rows:
            key2pid[canonical(row["name"])] = row["player_id"]
    return pd.DataFrame(audit_rows, columns=audit_columns)

# ───────────────────────── Matching ─────────────────────────

def _token_variants(key: str) -> List[str]:
    toks = key.split()
    n = len(toks)
    vs: List[str] = []
    if n >= 2:
        if n > 2:
            for i in range(n - 1):
                vs.append(" ".join(toks[i:i+2]))      # sliding bigrams
        vs.append(f"{toks[0]} {toks[-1]}")            # first + last
        vs.append(" ".join(toks[1:]))                 # drop first
        vs.append(" ".join(toks[:-1]))                # drop last
    return list(dict.fromkeys(vs))

def _fuzzy_best(name: str, keys: List[str], threshold: int) -> Optional[str]:
    if not keys:
        return None
    if _USE_RAPIDFUZZ:
        best_key, best_score = None, -1
        for k in keys:
            sc = rf_fuzz.token_set_ratio(name, k)
            if sc > best_score:
                best_key, best_score = k, sc
        return best_key if best_score >= threshold else None
    if fw_fuzz is None:
        best_key, best_score = None, -1.0
        for k in keys:
            score = 100.0 * SequenceMatcher(None, name, k).ratio()
            if score > best_score:
                best_key, best_score = k, score
        return best_key if best_score >= threshold else None
    best_key, best_score = None, -1
    for k in keys:
        sc = fw_fuzz.token_set_ratio(name, k)
        if sc > best_score:
            best_key, best_score = k, sc
    return best_key if best_score >= threshold else None

def resolve_player_id(raw_name: str,
                      key2pid: Dict[str, str],
                      overrides: Dict[str, str],
                      threshold: int = 85) -> Optional[str]:
    """Return player_id or None."""
    key = canonical(raw_name)

    # 1) overrides (strongest)
    if key in overrides:
        return overrides[key]

    # 2) exact canonical hit
    if key in key2pid:
        return key2pid[key]

    # 3) token variants
    for v in _token_variants(key):
        if v in overrides:
            return overrides[v]
        if v in key2pid:
            return key2pid[v]

    # 4) fuzzy last resort
    best = _fuzzy_best(key, list(key2pid.keys()), threshold)
    if best:
        return key2pid[best]
    return None

# ───────────────────────── Enrichment ─────────────────────────

def build_display_name(row: pd.Series) -> str:
    # Prefer concatenated first+second; fallback to web_name; else any existing 'name'
    fn = str(row.get("first_name") or "").strip()
    sn = str(row.get("second_name") or "").strip()
    if fn or sn:
        return f"{fn} {sn}".strip()
    wn = str(row.get("web_name") or "").strip()
    if wn:
        return wn
    for c in ["name", "player", "player_name", "Player Name"]:
        if c in row and str(row[c]).strip():
            return str(row[c]).strip()
    return ""

def get_career_season(rec: dict, season_long: str) -> Optional[dict]:
    """
    Try long form first (e.g., '2019-2020'); then short ('2019-20').
    """
    career = rec.get("career") or {}
    if season_long in career:
        return career[season_long]
    short = season_shortform(season_long)
    if short in career:
        return career[short]
    return None

def enrich_season(season_dir: Path,
                  out_root: Path,
                  pid2rec: Dict[str, dict],
                  key2pid: Dict[str, str],
                  overrides: Dict[str, str],
                  team_ids: Dict[str, str],
                  generate_missing_ids: bool,
                  threshold: int,
                  fail_if_unmatched_pct: float,
                  league: str = DEFAULT_FPL_LEAGUE,
                  fbref_root: Optional[Path] = None,
                  whoscored_root: Optional[Path] = None,
                  understat_root: Optional[Path] = None,
                  registry_root: Optional[Path] = None,
                  fpl_code_to_pid: Optional[Mapping[str, str]] = None,
                  historically_generated_codes: Optional[set[str]] = None,
                  compatibility_master_path: Optional[Path] = None) -> None:
    season = season_dir.name
    season_full = season_longform(season)

    in_csv  = season_dir / "season" / "cleaned_players.csv"
    if not in_csv.is_file():
        logging.warning("[%s] missing input: %s", season, in_csv)
        return

    df = read_csv_flex(in_csv)
    if df.empty:
        logging.warning("[%s] empty CSV: %s", season, in_csv)
        return

    df = attach_fpl_context(df, season_dir)
    df, reset_columns = reset_preseason_carryover(df, season_dir)
    if reset_columns:
        logging.warning(
            "[%s] all fixtures are unstarted; reset carried season totals: %s",
            season,
            ", ".join(reset_columns),
        )

    # Normalise "name"
    if "name" not in df.columns:
        df["name"] = df.apply(build_display_name, axis=1)
    else:
        df["name"] = df["name"].astype(str)

    # Approved provider bridges are stronger identity evidence than names.
    # Historical generated IDs are only fallbacks: a player may subsequently
    # have acquired an established canonical identity in the main registry.
    code_registry = dict(fpl_code_to_pid or {})
    generated_code_registry = set(historically_generated_codes or set())
    provider_codes = df.get(
        "fpl_code", pd.Series("", index=df.index, dtype="string")
    ).map(normalized_provider_code)
    approved_code_registry = {
        code: player_id
        for code, player_id in code_registry.items()
        if code not in generated_code_registry
    }
    df["player_id"] = provider_codes.map(approved_code_registry).astype("object")
    df["player_id_source"] = pd.Series(
        np.where(df["player_id"].notna(), "registry_fpl_code_bridge", None),
        index=df.index,
        dtype="object",
    )
    unresolved = df["player_id"].isna()
    df.loc[unresolved, "player_id"] = df.loc[unresolved, "name"].apply(
        lambda name: resolve_player_id(name, key2pid, overrides, threshold)
    )
    name_resolved = unresolved & df["player_id"].notna()
    df.loc[name_resolved, "player_id_source"] = "master_or_override"

    if "web_name" in df.columns:
        for idx in df.index[df["player_id"].isna()]:
            web_name = df.at[idx, "web_name"]
            if pd.isna(web_name) or not str(web_name).strip():
                continue
            web_key = canonical(str(web_name))
            player_id = overrides.get(web_key) or key2pid.get(web_key)
            if player_id:
                df.at[idx, "player_id"] = player_id
                df.at[idx, "player_id_source"] = "master_or_override_web_name"

    unresolved = df["player_id"].isna()
    historical_ids = provider_codes.map(
        {
            code: player_id
            for code, player_id in code_registry.items()
            if code in generated_code_registry
        }
    )
    historical_resolved = unresolved & historical_ids.notna()
    df.loc[historical_resolved, "player_id"] = historical_ids.loc[
        historical_resolved
    ]
    df.loc[historical_resolved, "player_id_source"] = (
        "historical_generated_fpl_code"
    )

    generated_mask = df["player_id_source"].eq("historical_generated_fpl_code")
    if generate_missing_ids and "fpl_code" in df.columns:
        for idx in df.index[df["player_id"].isna()]:
            provider_code = df.at[idx, "fpl_code"]
            if pd.isna(provider_code) or not str(provider_code).strip():
                continue
            df.at[idx, "player_id"] = stable_canonical_id(
                "player", "fpl", str(provider_code), length=8
            )
            df.at[idx, "player_id_source"] = "generated_from_fpl_code"
            generated_mask.at[idx] = True

    duplicate_ids = df.loc[
        df["player_id"].notna() & df["player_id"].duplicated(keep=False),
        "player_id",
    ].unique()
    for duplicate_id in duplicate_ids:
        duplicate_rows = df.index[df["player_id"] == duplicate_id].tolist()

        def _identity_confidence(idx: int) -> tuple[int, int]:
            full_key = canonical(df.at[idx, "name"])
            original_key = canonical(
                f"{df.at[idx, 'first_name']} {df.at[idx, 'second_name']}"
            )
            web_key = (
                canonical(df.at[idx, "web_name"])
                if "web_name" in df.columns and pd.notna(df.at[idx, "web_name"])
                else ""
            )
            master_key = canonical(pid2rec.get(str(duplicate_id), {}).get("name", ""))
            exact_override = int(
                overrides.get(original_key) == duplicate_id
                or overrides.get(full_key) == duplicate_id
            )
            exact_master = int(bool(master_key) and master_key in {full_key, original_key})
            exact_web = int(bool(master_key) and master_key == web_key)
            return (3 * exact_override + 2 * exact_master + exact_web, -idx)

        keeper = max(duplicate_rows, key=_identity_confidence)
        for idx in duplicate_rows:
            if idx == keeper or not generate_missing_ids:
                continue
            provider_code = df.at[idx, "fpl_code"]
            df.at[idx, "player_id"] = stable_canonical_id(
                "player", "fpl", str(provider_code), length=8
            )
            df.at[idx, "player_id_source"] = "generated_from_fpl_code_duplicate"
            generated_mask.at[idx] = True

    if df.loc[df["player_id"].notna(), "player_id"].duplicated().any():
        duplicates = sorted(
            df.loc[
                df["player_id"].notna() & df["player_id"].duplicated(keep=False),
                "player_id",
            ].astype(str).unique()
        )
        raise ValueError(
            "Player IDs remain duplicated after FPL-code resolution; possible "
            f"8-character hash collision or ambiguous identity: {duplicates}"
        )

    # Join FBref truth
    nations, borns, fb_positions, fb_teams, fplpos_from_master, master_names, has_career = [], [], [], [], [], [], []
    for pid in df["player_id"]:
        if not pid or pid not in pid2rec:
            nations.append(None); borns.append(None)
            fb_positions.append(None); fb_teams.append(None)
            fplpos_from_master.append(None); master_names.append(None); has_career.append(False)
            continue

        rec = pid2rec[pid]
        nations.append(rec.get("nation"))
        borns.append(rec.get("born"))
        master_names.append(rec.get("name") or None)

        srec = get_career_season(rec, season_full)
        if srec:
            has_career.append(True)
            fb_teams.append(srec.get("team") or srec.get("short") or srec.get("team_short"))
            fb_positions.append(srec.get("position") or srec.get("pos"))
            fplpos_from_master.append(srec.get("fpl_position") or srec.get("fpl_pos"))
        else:
            has_career.append(False)
            fb_teams.append(None)
            fb_positions.append(None)
            fplpos_from_master.append(None)

    df["nation"] = nations
    df["born"] = borns

    official_position_source = (
        df["fpl_element_type"]
        if "fpl_element_type" in df.columns
        else df.get("element_type", pd.Series([None] * len(df), index=df.index))
    )
    official_fpl_pos = official_position_source.map(normalise_fpl_position).astype("string")
    official_team = (
        df["fpl_team"].astype("string")
        if "fpl_team" in df.columns
        else pd.Series([None] * len(df), index=df.index, dtype="string")
    )

    df["team"] = official_team.fillna(pd.Series(fb_teams, dtype="string"))
    df["team_id"] = df["team"].map(
        lambda code: (
            team_ids.get(str(code).strip().upper())
            or stable_canonical_id(
                "team", "fpl", str(code).strip().upper(), length=12
            )
        )
        if pd.notna(code)
        else None
    )
    df["team_id_source"] = df["team"].map(
        lambda code: (
            "registry"
            if pd.notna(code) and str(code).strip().upper() in team_ids
            else "generated_from_fpl_code"
        )
        if pd.notna(code)
        else None
    )

    # Official FPL position is authoritative; master/FBref fill historical gaps.
    fbref_position = pd.Series(fb_positions, dtype="string")
    fbref_to_fpl = fbref_position.map(
        lambda p: FBREF_TO_FPL_POS.get(str(p).upper(), None)
        if pd.notna(p)
        else None
    ).astype("string")
    df["fpl_pos"] = official_fpl_pos.fillna(
        pd.Series(fplpos_from_master, dtype="string")
    ).fillna(fbref_to_fpl)
    df["position"] = fbref_position.fillna(df["fpl_pos"].map(FPL_TO_FBREF_POS))

    # Names should be FBref canonical names where matched
    df["name"] = pd.Series(master_names, dtype="string").fillna(df["name"].astype("string"))

    df, player_stats_audit = enrich_player_season_stats(
        df,
        season=season_full,
        league=league,
        fbref_root=fbref_root,
        whoscored_root=whoscored_root,
        understat_root=understat_root,
    )

    registry_audit = pd.DataFrame()
    if registry_root is not None and generated_mask.any():
        registry_audit = register_generated_players(
            df,
            generated_mask,
            season=season_full,
            league=league,
            registry_root=registry_root,
            compatibility_master_path=compatibility_master_path,
            pid2rec=pid2rec,
            key2pid=key2pid,
        )

    # Review partitions
    unmatched_ids = df[df["player_id"].isna()].copy()
    # "no season entry" = player_id matched but FBref has no career record for this season
    no_season_mask = (
        df["player_id"].notna()
        & (~pd.Series(has_career, index=df.index))
        & ~generated_mask
    )
    no_season_rows = df[no_season_mask].copy()

    # Write enriched file (keep all rows; downstream can filter if needed)
    out_csv = out_root / season / "season" / "cleaned_players.csv"
    write_csv_utf8(out_csv, df)
    logging.info("[%s] wrote enriched players: %s (rows=%d)", season, out_csv, len(df))

    review_dir = out_root / season / "_manual_review"
    write_json_utf8(
        review_dir / f"player_season_stat_enrichment_{season_full}.json",
        player_stats_audit,
    )

    if reset_columns:
        review_dir = out_root / season / "_manual_review"
        write_json_utf8(
            review_dir / f"preseason_carryover_reset_{season}.json",
            {
                "season": season,
                "status": "preseason_roster",
                "reason": "all fixtures unstarted and cumulative values were non-zero",
                "reset_columns": reset_columns,
            },
        )

    # Write unmatched IDs (CSV)
    if len(unmatched_ids):
        review_dir = out_root / season / "_manual_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        miss_csv = review_dir / f"missing_ids_{season}.csv"
        tmp = unmatched_ids[["name"]].copy()
        tmp["canonical"]   = tmp["name"].map(canonical)
        tmp["suggestions"] = tmp["canonical"].map(lambda c: " | ".join(_token_variants(c)[:4]))
        write_csv_utf8(miss_csv, tmp)
        logging.warning("[%s] unmatched rows=%d → %s", season, len(unmatched_ids), miss_csv)

    # Write missing season career entries (CSV)
    if len(no_season_rows):
        review_dir = out_root / season / "_manual_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        miss_season_csv = review_dir / f"missing_season_{season}.csv"
        cols = ["player_id", "name", "nation", "born"]
        write_csv_utf8(miss_season_csv, no_season_rows[cols])
        logging.warning("[%s] no-career-entry rows=%d → %s", season, len(no_season_rows), miss_season_csv)

    if generated_mask.any():
        review_dir = out_root / season / "_manual_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        generated_csv = review_dir / f"generated_ids_{season}.csv"
        generated_cols = [
            "player_id",
            "name",
            "fpl_element_id",
            "fpl_code",
            "fpl_opta_code",
            "team",
        ]
        write_csv_utf8(
            generated_csv,
            df.loc[generated_mask, [c for c in generated_cols if c in df.columns]],
        )
        logging.warning(
            "[%s] generated stable IDs=%d -> %s",
            season,
            int(generated_mask.sum()),
            generated_csv,
        )
        if not registry_audit.empty:
            registry_csv = review_dir / f"registered_generated_ids_{season}.csv"
            write_csv_utf8(registry_csv, registry_audit)
            logging.warning(
                "[%s] registered generated IDs=%d -> %s",
                season,
                len(registry_audit),
                registry_csv,
            )

    # Guardrail for unmatched percentage
    total = len(df)
    pct_unmatched = 100.0 * len(unmatched_ids) / max(total, 1)
    if pct_unmatched > fail_if_unmatched_pct:
        raise SystemExit(f"[{season}] Unmatched {pct_unmatched:.2f}% > {fail_if_unmatched_pct:.2f}% threshold")


def backfill_published_season_stats(
    season_dir: Path,
    league: str = DEFAULT_FPL_LEAGUE,
    fbref_root: Optional[Path] = None,
    whoscored_root: Optional[Path] = None,
    understat_root: Optional[Path] = None,
) -> dict:
    """Enrich an already-published roster in place while preserving its rows."""
    season = season_dir.name
    roster_path = season_dir / "season" / "cleaned_players.csv"
    if not roster_path.is_file():
        raise FileNotFoundError(f"Missing published roster: {roster_path}")

    players = read_csv_flex(roster_path)
    enriched, audit = enrich_player_season_stats(
        players,
        season=season,
        league=league,
        fbref_root=fbref_root,
        whoscored_root=whoscored_root,
        understat_root=understat_root,
    )
    write_csv_utf8(roster_path, enriched)
    write_json_utf8(
        season_dir / "_manual_review" / f"player_season_stat_enrichment_{season}.json",
        audit,
    )
    logging.info(
        "[%s] backfilled player-season stats: %s (rows=%d)",
        season,
        roster_path,
        len(enriched),
    )
    return audit

# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Enrich FPL season players with FBref metadata (names are FBref canonical)."
    )
    ap.add_argument("--raw-root", type=Path, default=None,
                    help="Raw FPL provider root or already league-scoped root")
    ap.add_argument("--proc-root", type=Path, required=True,
                    help="Processed FPL provider root or already league-scoped root")
    ap.add_argument("--league", default=DEFAULT_FPL_LEAGUE,
                    help="League folder beneath the provider roots")
    ap.add_argument("--fbref-master", type=Path, default=None,
                    help="FBref master players JSON (source of truth)")
    ap.add_argument("--fbref-root", type=Path, default=Path("data/processed/fbref"),
                    help="Processed FBref provider root")
    ap.add_argument("--whoscored-root", type=Path,
                    default=Path("data/processed/whoscored"),
                    help="Processed WhoScored provider root")
    ap.add_argument("--understat-root", type=Path,
                    default=Path("data/processed/understat"),
                    help="Processed Understat provider root")
    ap.add_argument("--overrides", type=Path, default=None,
                    help="Manual overrides JSON (e.g., 'first | last': 'pid')")
    ap.add_argument("--team-map", type=Path, default=None,
                    help="Optional team-code to canonical team_id JSON")
    ap.add_argument(
        "--registry-root",
        type=Path,
        default=None,
        help=(
            "Player registry root. Defaults to the parent of --fbref-master; "
            "generated FPL identities are promoted here."
        ),
    )
    ap.add_argument(
        "--register-generated-players",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Promote generated FPL identities into canonical player registries.",
    )
    ap.add_argument("--generate-missing-ids", action="store_true",
                    help="Generate deterministic canonical IDs from stable FPL codes")
    ap.add_argument("--threshold", type=int, default=85,
                    help="Fuzzy minimum (0-100)")
    ap.add_argument("--fail-if-unmatched", type=float, default=10.0,
                    help="Fail run if unmatched percentage exceeds this value")
    ap.add_argument("--season", type=str, default="all",
                    help="Which season to process: 'all' (default), 'latest', or a specific season like '2025-26'/'2025-2026'")
    ap.add_argument("--stats-only", action="store_true",
                    help="Backfill player-season stats into already-published rosters")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = ap.parse_args()

    args.proc_root = league_scoped_root(args.proc_root, args.league)
    if args.raw_root is not None:
        args.raw_root = league_scoped_root(args.raw_root, args.league)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    discovery_root = args.proc_root if args.stats_only else args.raw_root
    if discovery_root is None or not discovery_root.exists():
        option = "--proc-root" if args.stats_only else "--raw-root"
        raise SystemExit(f"{option} does not exist: {discovery_root}")

    all_dirs = [d for d in discovery_root.iterdir() if d.is_dir()]
    season_dirs = [d for d in all_dirs if _looks_like_season(d.name)]
    season_dirs = sorted(season_dirs, key=_season_sort_key)

    if not season_dirs:
        logging.warning("No season-like folders under %s", discovery_root)
        return

    sel = (args.season or "all").strip().lower()
    if sel == "all":
        seasons = season_dirs
    elif sel == "latest":
        seasons = [season_dirs[-1]]
    else:
        match = _match_season_dir(season_dirs, args.season.strip())
        if not match:
            raise SystemExit(
                f"Requested season {args.season!r} not found under {discovery_root}. "
                f"Available: {[d.name for d in season_dirs]}"
            )
        seasons = [match]

    logging.info("Processing season folder(s): %s", [d.name for d in seasons])

    if args.stats_only:
        for season_dir in seasons:
            backfill_published_season_stats(
                season_dir=season_dir,
                league=args.league,
                fbref_root=args.fbref_root,
                whoscored_root=args.whoscored_root,
                understat_root=args.understat_root,
            )
        return

    if args.fbref_master is None or not args.fbref_master.is_file():
        raise SystemExit("--fbref-master is required for full enrichment")

    pid2rec, key2pid = load_fbref_master(args.fbref_master)
    overrides = load_overrides(args.overrides)
    team_ids = load_team_id_lookup(args.team_map)
    registry_root = args.registry_root or args.fbref_master.parent
    extend_player_name_lookup(key2pid, registry_root)

    for season_dir in seasons:
        fpl_code_to_pid, historically_generated_codes = load_fpl_code_registry(
            registry_root,
            args.proc_root,
            target_season=season_dir.name,
        )
        logging.info("Season %s …", season_dir.name)
        enrich_season(
            season_dir=season_dir,
            out_root=args.proc_root,
            pid2rec=pid2rec,
            key2pid=key2pid,
            overrides=overrides,
            team_ids=team_ids,
            generate_missing_ids=args.generate_missing_ids,
            threshold=args.threshold,
            fail_if_unmatched_pct=args.fail_if_unmatched,
            league=args.league,
            fbref_root=args.fbref_root,
            whoscored_root=args.whoscored_root,
            understat_root=args.understat_root,
            registry_root=(registry_root if args.register_generated_players else None),
            fpl_code_to_pid=fpl_code_to_pid,
            historically_generated_codes=historically_generated_codes,
            compatibility_master_path=args.proc_root / "master_fpl_players.json",
        )

if __name__ == "__main__":
    main()
