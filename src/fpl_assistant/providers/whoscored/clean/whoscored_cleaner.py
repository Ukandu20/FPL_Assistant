"""Normalize native WhoScored data into registry-keyed analytical tables.

The output deliberately remains provider-owned.  It mirrors the useful
FBref table-family layout without claiming that similarly named metrics have
identical provider definitions.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from fpl_assistant.canonical.identity import normalize_identity_text, stable_canonical_id


LOG = logging.getLogger("whoscored.clean")
PROVIDER = "whoscored"
PROCESSING_VERSION = "1.2.0"

BUILTIN_PLAYER_ALIASES = {
    "andy robertson": "andrew robertson",
    "charly alcaraz": "carlos alcaraz",
    "savinho": "savio",
}

ID_COLUMNS = [
    "league",
    "season",
    "provider_season",
    "game",
    "game_date",
    "kickoff_utc",
    "gameweek",
    "match_id",
    "provider_match_id",
    "team_id",
    "provider_team_id",
    "team",
    "provider_team_name",
    "opponent_id",
    "opponent",
    "home",
    "away",
    "is_home",
]
PLAYER_ID_COLUMNS = ID_COLUMNS + [
    "player_id",
    "provider_player_id",
    "player",
    "provider_player_name",
    "nation",
    "born",
    "age",
    "provider_position_match",
    "is_starter",
    "position_detail_match",
    "position_group_match",
    "position",
    "primary_position",
    "fpl_pos",
    "position_detail_match_imputed",
    "position_source",
    "position_confidence",
    "position_imputation_source",
    "fpl_position_source",
    "fpl_position_confidence",
    "shirt_no",
    "minutes",
]

EVENT_BINS: dict[str, str] = {
    "Pass": "passing_possession",
    "BallTouch": "passing_possession",
    "ShieldBallOpp": "passing_possession",
    "TakeOn": "ball_progression",
    "GoodSkill": "ball_progression",
    "Dispossessed": "possession_loss",
    "Error": "possession_loss",
    "Tackle": "defensive_action",
    "Interception": "defensive_action",
    "Clearance": "defensive_action",
    "BlockedPass": "defensive_action",
    "BallRecovery": "defensive_action",
    "Aerial": "duel",
    "Challenge": "duel",
    "Goal": "shot",
    "MissedShots": "shot",
    "ShotOnPost": "shot",
    "ChanceMissed": "shot",
    "Save": "goalkeeping",
    "SavedShot": "shot",
    "KeeperPickup": "goalkeeping",
    "Claim": "goalkeeping",
    "Punch": "goalkeeping",
    "KeeperSweeper": "goalkeeping",
    "Smother": "goalkeeping",
    "PenaltyFaced": "goalkeeping",
    "CrossNotClaimed": "goalkeeping",
    "Foul": "discipline",
    "Card": "discipline",
    "OffsideGiven": "offside",
    "OffsidePass": "offside",
    "OffsideProvoked": "offside",
    "CornerAwarded": "set_piece",
    "SubstitutionOn": "tactical",
    "SubstitutionOff": "tactical",
    "FormationSet": "tactical",
    "FormationChange": "tactical",
    "Start": "administration",
    "End": "administration",
}

POSITION_MAP = {
    "GK": ("GK", "GK"),
    "DC": ("CB", "DEF"),
    "DL": ("FB", "DEF"),
    "DR": ("FB", "DEF"),
    "DML": ("LWB", "DEF"),
    "DMR": ("RWB", "DEF"),
    "WBL": ("LWB", "DEF"),
    "WBR": ("RWB", "DEF"),
    "DMC": ("DM", "MID"),
    "MC": ("CM", "MID"),
    "ML": ("WM", "MID"),
    "MR": ("WM", "MID"),
    "AMC": ("AM", "MID"),
    "AML": ("W", "MID"),
    "AMR": ("W", "MID"),
    "FW": ("FW", "FWD"),
    "FWL": ("FW", "FWD"),
    "FWR": ("FW", "FWD"),
    "Sub": ("UNK", "UNK"),
}

FPL_POSITION_ALIASES = {
    "GK": "GKP", "GKP": "GKP", "GOALKEEPER": "GKP", "1": "GKP",
    "DEF": "DEF", "DF": "DEF", "DEFENDER": "DEF", "2": "DEF",
    "MID": "MID", "MF": "MID", "MIDFIELDER": "MID", "3": "MID",
    "FWD": "FWD", "FW": "FWD", "FORWARD": "FWD", "4": "FWD",
}

REGISTRY_DETAIL_MAP = {
    "GK": "GKP", "GKP": "GKP",
    "DF": "DEF", "DEF": "DEF", "CB": "DEF", "FB": "DEF",
    "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "MF": "MID", "MID": "MID", "DM": "MID", "CM": "MID",
    "AM": "MID", "WM": "MID", "W": "MID",
    "FW": "FWD", "FWD": "FWD", "CF": "FWD", "ST": "FWD",
}

PLAYER_TABLES: dict[str, list[str]] = {
    "summary": [
        "goals", "assists", "shots_total", "shots_on_target", "yellow_cards",
        "red_cards", "touches", "tackles", "interceptions", "blocks",
        "passes_completed", "passes_attempted", "pass_completion_pct", "key_passes",
        "takeons_attempted", "takeons_successful", "rating", "is_man_of_the_match",
    ],
    "defense": [
        "tackles", "tackles_won", "tackles_lost", "dribbled_past", "blocks",
        "interceptions", "clearances", "errors", "recoveries", "defensive_aerials",
    ],
    "keepers": [
        "shots_on_target_against", "goals_against", "saves", "save_pct", "claims_high",
        "collected", "parried_danger", "parried_safe", "punches", "keeper_pickups",
        "keeper_sweeper_actions", "smothers", "penalties_faced", "crosses_not_claimed",
    ],
    "passing": [
        "passes_completed", "passes_attempted", "pass_completion_pct", "key_passes",
        "assists", "progressive_passes", "passes_final_third", "passes_penalty_area",
        "crosses_penalty_area",
    ],
    "passing_types": [
        "passes_attempted", "passes_live", "passes_dead", "passes_free_kick",
        "through_balls", "switches", "crosses", "throw_ins", "corners",
        "corners_inswinging", "corners_outswinging", "corners_straight",
        "passes_completed", "passes_offside", "passes_blocked",
    ],
    "possession": [
        "touches", "touches_defensive_third", "touches_middle_third",
        "touches_attacking_third", "touches_penalty_area", "takeons_attempted",
        "takeons_successful", "takeon_success_pct", "dribbled_past", "dispossessed",
        "recoveries", "carries", "progressive_carries",
    ],
    "misc": [
        "yellow_cards", "red_cards", "fouls_committed", "fouls_drawn", "offsides",
        "crosses", "interceptions", "tackles_won", "penalties_won",
        "penalties_conceded", "own_goals", "recoveries", "aerials_won", "aerials_lost",
    ],
    "shooting": [
        "goals", "shots_total", "shots_on_target", "shots_off_target", "shots_blocked",
        "shots_on_post", "shot_on_target_pct", "goals_per_shot", "average_shot_distance",
        "shots_box", "shots_outside_box", "headed_shots", "penalty_goals", "penalty_attempts",
    ],
    "ratings": ["rating", "is_man_of_the_match"],
    "duels": [
        "aerials_total", "aerials_won", "aerials_lost", "aerial_success_pct",
        "offensive_aerials", "defensive_aerials", "challenges", "tackles",
        "tackles_won", "tackle_success_pct",
    ],
}

TEAM_EXTRA_TABLES: dict[str, list[str]] = {
    "schedule": [
        "home_team_id", "away_team_id", "status", "home_score", "away_score",
        "score", "venue", "referee", "attendance", "formation", "opponent_formation",
    ],
    "shot_zones": [],
    "goal_shot_creation": [
        "key_passes", "assists", "shots_total", "fouls_drawn",
        "takeons_successful", "shot_creating_actions", "goal_creating_actions",
    ],
}

NON_SUM_METRICS = {
    "rating", "is_man_of_the_match", "pass_completion_pct", "save_pct",
    "takeon_success_pct", "shot_on_target_pct", "goals_per_shot",
    "average_shot_distance", "aerial_success_pct", "tackle_success_pct",
    "possession", "average_age",
}


@dataclass(frozen=True)
class CleanResult:
    output_dir: Path
    table_manifest: pd.DataFrame
    coverage: pd.DataFrame
    player_identity: pd.DataFrame
    team_identity: pd.DataFrame
    match_identity: pd.DataFrame


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, **kwargs)


def _drop_export_index(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ~frame.columns.astype(str).str.match(r"^Unnamed:|^level_0$|^index$")]


def _repair_text(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if not any(marker in value for marker in ("Ã", "Â", "â", "ð")):
        return value
    try:
        repaired = value.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    return repaired


def _identity_key(value: Any) -> str:
    repaired = _repair_text(value)
    transliterated = (
        repaired.translate(str.maketrans({"ı": "i", "ł": "l", "đ": "d", "ø": "o"}))
        if isinstance(repaired, str)
        else repaired
    )
    return normalize_identity_text(transliterated)


def _provider_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def normalize_season(value: str) -> tuple[str, str]:
    text = str(value).strip()
    match = re.fullmatch(r"(\d{4})(?:[-/](\d{2}|\d{4}))?", text)
    if not match:
        raise ValueError(f"Unsupported season value: {value!r}")
    start = int(match.group(1))
    end_raw = match.group(2)
    if end_raw is None:
        end = start + 1
        provider = str(start)
    else:
        end = int(end_raw) if len(end_raw) == 4 else (start // 100) * 100 + int(end_raw)
        provider = str(start)
    if end != start + 1:
        raise ValueError(f"Expected a split-year domestic season, got {value!r}")
    return f"{start:04d}-{end:04d}", provider


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_bridge(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=list(columns))
    frame = _read_csv(path, dtype="string")
    for column in columns:
        if column not in frame:
            frame[column] = pd.NA
    return frame.loc[:, list(columns)]


def _team_resolution(
    teams: pd.DataFrame,
    *,
    teams_config: Mapping[str, str],
    team_lookup: Mapping[str, str],
    existing: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = {_identity_key(name): str(code).upper() for name, code in teams_config.items()}
    lookup = {_identity_key(code): str(cid) for code, cid in team_lookup.items()}
    bridge_lookup = {
        _provider_id(row.provider_id): str(row.canonical_id)
        for row in existing.itertuples(index=False)
        if str(row.provider).lower() == PROVIDER and pd.notna(row.canonical_id)
    }
    rows: list[dict[str, Any]] = []
    for row in teams.drop_duplicates("provider_team_id").itertuples(index=False):
        provider_id = _provider_id(row.provider_team_id)
        name = str(_repair_text(row.provider_team_name))
        canonical_id = bridge_lookup.get(provider_id)
        method = "existing_bridge" if canonical_id else ""
        confidence = 1.0 if canonical_id else 0.0
        code = config.get(_identity_key(name))
        if canonical_id is None and code:
            canonical_id = lookup.get(_identity_key(code))
            if canonical_id:
                method, confidence = "configured_name_to_code", 1.0
        rows.append(
            {
                "entity_type": "team",
                "provider": PROVIDER,
                "provider_id": provider_id,
                "provider_name": name,
                "canonical_id": canonical_id,
                "valid_from": pd.NA,
                "valid_to": pd.NA,
                "match_method": method or "unresolved",
                "match_confidence": confidence,
                "review_status": "approved" if canonical_id else "needs_review",
            }
        )
    audit = pd.DataFrame(rows)
    resolved = audit[audit["canonical_id"].notna()].loc[:, list(existing.columns)].copy()
    combined = pd.concat([existing, resolved], ignore_index=True)
    combined = combined.drop_duplicates(["entity_type", "provider", "provider_id"], keep="last")
    conflicts = combined.groupby(["entity_type", "provider", "provider_id"])["canonical_id"].nunique()
    if (conflicts > 1).any():
        raise ValueError("A WhoScored team ID maps to multiple canonical team IDs.")
    return audit, combined


def _player_resolution(
    players: pd.DataFrame,
    *,
    player_lookup: Mapping[str, str],
    master_players: Mapping[str, Any],
    official_fpl_players: pd.DataFrame | None = None,
    player_aliases: Mapping[str, Any],
    team_map: Mapping[str, str],
    season: str,
    existing: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = {_identity_key(name): str(cid) for name, cid in player_lookup.items()}
    fullname_aliases = {
        _identity_key(source): _identity_key(target)
        for source, target in {
            **BUILTIN_PLAYER_ALIASES,
            **player_aliases.get("fullname_alias_map", {}),
        }.items()
    }
    global_candidates = list(lookup.items())
    team_candidates: dict[str, list[tuple[str, str]]] = {}
    for canonical_id, record in master_players.items():
        name_key = _identity_key(record.get("name", ""))
        career = record.get("career", {})
        season_record = career.get(season, {}) if isinstance(career, Mapping) else {}
        canonical_team_id = str(season_record.get("team_id", "")).strip()
        if name_key and canonical_team_id:
            team_candidates.setdefault(canonical_team_id, []).append((name_key, str(canonical_id)))

    # The current FPL roster is authoritative for in-season membership and can
    # contain newly registered players that have not reached master_players yet.
    # Use all official name variants, but only permit globally unique exact-name
    # keys. Team-scoped matching remains safe when the same short name occurs in
    # more than one squad.
    official_by_name: dict[str, set[str]] = {}
    if official_fpl_players is not None and not official_fpl_players.empty:
        for row in official_fpl_players.to_dict("records"):
            canonical_id = _provider_id(row.get("player_id"))
            canonical_team_id = _provider_id(row.get("team_id"))
            if not canonical_id:
                continue
            full_name = " ".join(
                part for part in (
                    str(row.get("first_name", "")).strip(),
                    str(row.get("second_name", "")).strip(),
                )
                if part and part.lower() not in {"nan", "<na>"}
            )
            variants = {
                _identity_key(value)
                for value in (row.get("name"), full_name, row.get("web_name"))
                if pd.notna(value) and _identity_key(value)
            }
            prefix_variants = {
                " ".join(tokens[:length])
                for name_key in variants
                for tokens in [name_key.split()]
                for length in range(2, len(tokens))
            }
            variants.update(prefix_variants)
            for name_key in variants:
                official_by_name.setdefault(name_key, set()).add(canonical_id)
                global_candidates.append((name_key, canonical_id))
                if canonical_team_id:
                    team_candidates.setdefault(canonical_team_id, []).append(
                        (name_key, canonical_id)
                    )
        for name_key, canonical_ids in official_by_name.items():
            if len(canonical_ids) == 1:
                # The element-linked current-season FPL roster supersedes stale
                # historical registry aliases for current Premier League players.
                lookup[name_key] = next(iter(canonical_ids))

    global_candidates = list(dict.fromkeys(global_candidates))
    team_candidates = {
        team_id: list(dict.fromkeys(candidates))
        for team_id, candidates in team_candidates.items()
    }
    bridge_lookup = {
        _provider_id(row.provider_id): str(row.canonical_id)
        for row in existing.itertuples(index=False)
        if str(row.provider).lower() == PROVIDER and pd.notna(row.canonical_id)
    }
    rows: list[dict[str, Any]] = []
    for provider_value, group in players.groupby("provider_player_id", dropna=False):
        provider_id = _provider_id(provider_value)
        name = str(_repair_text(group["provider_player_name"].dropna().iloc[0]))
        provider_team_ids = {
            _provider_id(value) for value in group.get("provider_team_id", pd.Series(dtype="object"))
            if _provider_id(value)
        }
        canonical_team_ids = {team_map[value] for value in provider_team_ids if value in team_map}
        canonical_id = bridge_lookup.get(provider_id)
        method = "existing_bridge" if canonical_id else ""
        confidence = 1.0 if canonical_id else 0.0
        if canonical_id is None:
            name_key = _identity_key(name)
            alias_key = fullname_aliases.get(name_key, name_key)
            canonical_id = lookup.get(alias_key)
            if canonical_id:
                method = "configured_alias" if alias_key != name_key else "exact_normalized_name"
                confidence = 0.99 if alias_key != name_key else 0.98
        if canonical_id is None:
            name_key = _identity_key(name)
            containment = [
                (candidate_name, candidate_id)
                for candidate_name, candidate_id in global_candidates
                if min(len(name_key), len(candidate_name)) >= 5
                and (name_key in candidate_name or candidate_name in name_key)
            ]
            unique_targets = list(dict.fromkeys(candidate_id for _, candidate_id in containment))
            if len(unique_targets) == 1:
                canonical_id = unique_targets[0]
                method, confidence = "unique_global_name_containment", 0.95
        if canonical_id is None and canonical_team_ids:
            candidates: list[tuple[str, str]] = []
            for canonical_team_id in canonical_team_ids:
                candidates.extend(team_candidates.get(canonical_team_id, []))
            candidates = list(dict.fromkeys(candidates))
            name_key = _identity_key(name)
            containment = [
                (candidate_name, candidate_id)
                for candidate_name, candidate_id in candidates
                if min(len(name_key), len(candidate_name)) >= 5
                and (name_key in candidate_name or candidate_name in name_key)
            ]
            containment_targets = list(
                dict.fromkeys(candidate_id for _, candidate_id in containment)
            )
            if len(containment_targets) == 1:
                canonical_id = containment_targets[0]
                method, confidence = "unique_team_name_containment", 0.96
            if canonical_id is None:
                surname = name_key.split()[-1:] or [""]
                surname_matches = [
                    (candidate_name, candidate_id)
                    for candidate_name, candidate_id in candidates
                    if candidate_name.split()[-1:] == surname
                    and (
                        name_key.split()[0] in candidate_name.split()[0]
                        or candidate_name.split()[0] in name_key.split()[0]
                        or SequenceMatcher(
                            None, name_key.split()[0], candidate_name.split()[0]
                        ).ratio() >= 0.80
                    )
                ]
                if len(surname_matches) == 1 and surname[0]:
                    canonical_id = surname_matches[0][1]
                    method, confidence = "unique_team_surname", 0.94
            if canonical_id is None and candidates:
                scored = sorted(
                    [
                        (SequenceMatcher(None, name_key, candidate_name).ratio(), candidate_id)
                        for candidate_name, candidate_id in candidates
                    ],
                    reverse=True,
                )
                if scored[0][0] >= 0.90 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.05):
                    canonical_id = scored[0][1]
                    method, confidence = "team_season_fuzzy_name", round(scored[0][0], 4)
        rows.append(
            {
                "entity_type": "player",
                "provider": PROVIDER,
                "provider_id": provider_id,
                "provider_name": name,
                "canonical_id": canonical_id,
                "provider_team_ids": "|".join(sorted(provider_team_ids)),
                "canonical_team_ids": "|".join(sorted(canonical_team_ids)),
                "valid_from": pd.NA,
                "valid_to": pd.NA,
                "match_method": method or "unresolved",
                "match_confidence": confidence,
                "review_status": (
                    "approved" if method in {"existing_bridge", "configured_alias", "exact_normalized_name"}
                    else "reviewed" if canonical_id else "needs_review"
                ),
            }
        )
    audit = pd.DataFrame(rows)
    resolved = audit[audit["canonical_id"].notna()].loc[:, list(existing.columns)].copy()
    combined = pd.concat([existing, resolved], ignore_index=True)
    combined = combined.drop_duplicates(["entity_type", "provider", "provider_id"], keep="last")
    conflicts = combined.groupby(["entity_type", "provider", "provider_id"])["canonical_id"].nunique()
    if (conflicts > 1).any():
        raise ValueError("A WhoScored player ID maps to multiple canonical player IDs.")
    return audit, combined


def _match_resolution(
    schedule: pd.DataFrame,
    *,
    season: str,
    team_map: Mapping[str, str],
    fixture_path: Path,
    existing: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = schedule.copy()
    work["provider_match_id"] = work["game_id"].map(_provider_id)
    work["provider_home_team_id"] = work["home_team_id"].map(_provider_id)
    work["provider_away_team_id"] = work["away_team_id"].map(_provider_id)
    work["home_team_id"] = work["provider_home_team_id"].map(team_map)
    work["away_team_id"] = work["provider_away_team_id"].map(team_map)
    work["game_date"] = pd.to_datetime(work.get("date"), errors="coerce").dt.strftime("%Y-%m-%d")
    work["kickoff_utc"] = pd.to_datetime(work.get("start_time"), utc=True, errors="coerce")

    fixtures = _read_csv(fixture_path)
    fixture_date_col = "date_played" if "date_played" in fixtures else "date_sched"
    fixture_cols = ["fbref_id", "fpl_id", fixture_date_col, "home_id", "away_id"]
    if "gw_played" in fixtures:
        fixture_cols.append("gw_played")
    fixtures = fixtures[fixture_cols].drop_duplicates().copy()
    fixtures[fixture_date_col] = pd.to_datetime(fixtures[fixture_date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    fixtures = fixtures.rename(
        columns={
            fixture_date_col: "game_date",
            "gw_played": "gameweek",
            "home_id": "home_team_id",
            "away_id": "away_team_id",
        }
    )
    work["fixture_match_method_candidate"] = "registry_fixture_exact"
    work = work.merge(
        fixtures,
        on=["game_date", "home_team_id", "away_team_id"],
        how="left",
        validate="many_to_one",
    )
    # Domestic schedules contain each ordered home/away pairing once.  This
    # survives postponements where providers disagree on the stale kickoff.
    pair_counts = fixtures.groupby(["home_team_id", "away_team_id"]).size()
    unique_pairs = pair_counts[pair_counts.eq(1)].index
    pair_fixtures = fixtures.set_index(["home_team_id", "away_team_id"])
    unresolved_fixture = work["fbref_id"].isna()
    for index in work.index[unresolved_fixture]:
        pair = (work.at[index, "home_team_id"], work.at[index, "away_team_id"])
        if pair not in unique_pairs:
            continue
        candidate = pair_fixtures.loc[pair]
        for column in ("fbref_id", "fpl_id", "gameweek"):
            if column in candidate:
                work.at[index, column] = candidate[column]
        work.at[index, "fixture_match_method_candidate"] = "registry_fixture_team_pair"
    existing_lookup = {
        _provider_id(row.provider_match_id): str(row.match_id)
        for row in existing.itertuples(index=False)
        if str(row.provider).lower() == PROVIDER and pd.notna(row.match_id)
    }
    rows: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        provider_id = row.provider_match_id
        match_id = existing_lookup.get(provider_id)
        method = "existing_bridge" if match_id else ""
        confidence = 1.0 if match_id else 0.0
        if match_id is None and pd.notna(row.fbref_id):
            match_id = _provider_id(row.fbref_id)
            method = row.fixture_match_method_candidate
            confidence = 1.0 if method == "registry_fixture_exact" else 0.98
        rows.append(
            {
                "provider": PROVIDER,
                "provider_match_id": provider_id,
                "match_id": match_id,
                "provider_game": row.game,
                "match_method": method or "unresolved",
                "match_confidence": confidence,
                "competition": row.league,
                "season": season,
                "kickoff_utc": row.kickoff_utc,
                "game_date": row.game_date,
                "home_team_id": row.home_team_id,
                "away_team_id": row.away_team_id,
                "fpl_fixture_id": row.fpl_id,
                "gameweek": getattr(row, "gameweek", pd.NA),
                "review_status": "approved" if match_id else "needs_review",
            }
        )
    audit = pd.DataFrame(rows)
    bridge_cols = [
        "provider", "provider_match_id", "match_id", "provider_game",
        "match_method", "match_confidence",
    ]
    resolved = audit[audit["match_id"].notna()][bridge_cols]
    combined = pd.concat([existing, resolved], ignore_index=True)
    combined = combined.drop_duplicates(["provider", "provider_match_id"], keep="last")
    conflicts = combined.groupby(["provider", "provider_match_id"])["match_id"].nunique()
    if (conflicts > 1).any():
        raise ValueError("A WhoScored match ID maps to multiple canonical match IDs.")
    return work, audit, combined


def _pivot_stats(
    frame: pd.DataFrame,
    *,
    keys: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["stat_key_raw"] = work["stat_key"].astype("string")
    work["metric"] = work["stat_key_raw"].str.replace(r"^stats\.", "", regex=True)
    work["numeric_value"] = pd.to_numeric(work["value"], errors="coerce")
    boolean_value = (
        work["value"].astype("string").str.lower().map({"true": 1.0, "false": 0.0})
    )
    work["numeric_value"] = work["numeric_value"].fillna(boolean_value)
    conflict = (
        work.dropna(subset=["numeric_value"])
        .groupby(keys + ["metric"], dropna=False)["numeric_value"]
        .agg(value_count="size", distinct_values="nunique", minimum="min", maximum="max")
        .reset_index()
    )
    conflict = conflict[conflict["distinct_values"] > 1].reset_index(drop=True)
    work["source_priority"] = work["stat_key_raw"].str.startswith("stats.").astype(int)
    chosen = (
        work.sort_values("source_priority")
        .drop_duplicates(keys + ["metric"], keep="first")
    )
    wide = chosen.pivot(index=keys, columns="metric", values="numeric_value").reset_index()
    wide.columns.name = None
    return wide, conflict


def _qualifier_flag(series: pd.Series, name: str) -> pd.Series:
    return series.astype("string").str.contains(
        rf"['\"]displayName['\"]\s*:\s*['\"]{re.escape(name)}['\"]",
        regex=True,
        na=False,
    )


def _normalize_events(
    events: pd.DataFrame,
    *,
    match_map: Mapping[str, str],
    team_map: Mapping[str, str],
    player_map: Mapping[str, str],
    player_names: Mapping[str, str],
    team_names: Mapping[str, str],
    context: pd.DataFrame,
    season: str,
    provider_season: str,
) -> pd.DataFrame:
    work = _drop_export_index(events.copy())
    work["provider_match_id"] = work["game_id"].map(_provider_id)
    work["provider_event_object_id"] = work.get(
        "id", pd.Series(index=work.index, dtype="object")
    ).map(_provider_id)
    work["provider_event_sequence_id"] = work.get(
        "event_id", pd.Series(index=work.index, dtype="object")
    ).map(_provider_id)
    work["provider_event_id"] = (
        work["provider_match_id"]
        + ":"
        + work["provider_event_object_id"]
        + ":"
        + work["provider_event_sequence_id"]
    )
    work["provider_team_id"] = work["team_id"].map(_provider_id)
    work["provider_player_id"] = work["player_id"].map(_provider_id)
    work["provider_related_player_id"] = work.get("related_player_id", pd.Series(index=work.index, dtype="object")).map(_provider_id)
    work["provider_player_name"] = work.get("player", pd.Series("", index=work.index)).map(_repair_text)
    work["provider_team_name"] = work.get("team", pd.Series("", index=work.index)).map(_repair_text)
    work["match_id"] = work["provider_match_id"].map(match_map)
    work["team_id"] = work["provider_team_id"].map(team_map)
    work["player_id"] = work["provider_player_id"].map(player_map)
    work["related_player_id"] = work["provider_related_player_id"].map(player_map)
    work["player_identity_status"] = np.where(
        work["provider_player_id"].eq(""),
        "not_applicable",
        np.where(work["player_id"].notna(), "resolved", "unresolved"),
    )
    work["related_player_identity_status"] = np.where(
        work["provider_related_player_id"].eq(""),
        "not_applicable",
        np.where(work["related_player_id"].notna(), "resolved", "unresolved"),
    )
    identity_parts = pd.DataFrame(
        {
            "provider": PROVIDER,
            "provider_event_id": work["provider_event_id"],
            "type": work.get("type", ""),
            "period": work.get("period", ""),
            "minute": work.get("minute", ""),
            "second": work.get("second", ""),
            "provider_team_id": work["provider_team_id"],
            "provider_player_id": work["provider_player_id"],
        },
        index=work.index,
    ).fillna("").astype(str)
    stable_hash = pd.util.hash_pandas_object(identity_parts, index=False)
    work["event_id"] = stable_hash.map(lambda value: f"{int(value):016x}")
    work["player"] = work["player_id"].map(player_names).fillna(work["provider_player_name"])
    work["team"] = work["team_id"].map(team_names).fillna(work["provider_team_name"])
    work["event_bin"] = work["type"].map(EVENT_BINS).fillna("unmapped")
    work["is_successful"] = work.get("outcome_type", "").astype("string").str.lower().eq("successful")
    qualifiers = work.get("qualifiers", pd.Series("", index=work.index, dtype="string"))
    work["is_attacking"] = work["event_bin"].isin({"ball_progression", "shot"})
    work["is_defensive"] = work["event_bin"].isin({"defensive_action", "duel", "goalkeeping"})
    work["is_goalkeeper_action"] = work["event_bin"].eq("goalkeeping")
    work["is_shot_event"] = work["type"].isin({"Goal", "MissedShots", "SavedShot", "ShotOnPost", "ChanceMissed"})
    work["is_set_piece"] = (
        work["event_bin"].eq("set_piece")
        | _qualifier_flag(qualifiers, "SetPiece")
        | _qualifier_flag(qualifiers, "FromCorner")
        | _qualifier_flag(qualifiers, "FreeKick")
        | _qualifier_flag(qualifiers, "Penalty")
    )
    work["league"] = work.get("league", pd.Series(index=work.index, dtype="object"))
    work["season"] = season
    work["provider_season"] = provider_season
    work["provider"] = PROVIDER
    work["processing_version"] = PROCESSING_VERSION
    meta = context[[
        "provider_match_id", "game_date", "kickoff_utc", "gameweek",
        "home_team_id", "away_team_id",
    ]].drop_duplicates("provider_match_id")
    work = work.merge(meta, on="provider_match_id", how="left", validate="many_to_one")
    work["opponent_id"] = np.where(
        work["team_id"].eq(work["home_team_id"]), work["away_team_id"],
        np.where(work["team_id"].eq(work["away_team_id"]), work["home_team_id"], pd.NA),
    )
    work["opponent"] = work["opponent_id"].map(team_names)
    work["is_home"] = work["team_id"].eq(work["home_team_id"])
    return work


def _event_aggregates(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = events.copy()
    keys = ["provider_match_id", "provider_player_id", "provider_team_id"]
    outcome = work["is_successful"]
    qualifiers = work.get("qualifiers", pd.Series("", index=work.index, dtype="string"))
    event_type = work["type"]
    set_piece = work.get(
        "is_set_piece",
        pd.Series(False, index=work.index, dtype="bool"),
    ).fillna(False).astype(bool)
    blocked_shot = event_type.eq("SavedShot") & _qualifier_flag(qualifiers, "Blocked")
    own_goal = event_type.eq("Goal") & _qualifier_flag(qualifiers, "OwnGoal")
    shot = event_type.isin({"Goal", "MissedShots", "SavedShot", "ShotOnPost", "ChanceMissed"}) & ~own_goal
    normal_goal = event_type.eq("Goal") & ~own_goal
    pass_event = event_type.eq("Pass")
    x = pd.to_numeric(work.get("x"), errors="coerce")
    end_x = pd.to_numeric(work.get("end_x"), errors="coerce")
    y = pd.to_numeric(work.get("y"), errors="coerce")
    end_y = pd.to_numeric(work.get("end_y"), errors="coerce")
    work["_shot_distance"] = np.sqrt(((100 - x) * 1.05) ** 2 + ((50 - y) * 0.68) ** 2).where(shot)
    flags: dict[str, pd.Series] = {
        "goals": normal_goal,
        "own_goals": own_goal,
        "shots_total": shot,
        "shots_on_target": normal_goal | (event_type.eq("SavedShot") & ~blocked_shot),
        "shots_off_target": event_type.isin({"MissedShots", "ChanceMissed"}),
        "shots_blocked": blocked_shot,
        "shots_on_post": event_type.eq("ShotOnPost"),
        "shots_box": shot & (x >= 83),
        "shots_outside_box": shot & (x < 83),
        "headed_shots": shot & _qualifier_flag(qualifiers, "Head"),
        "penalty_attempts": shot & _qualifier_flag(qualifiers, "Penalty"),
        "penalty_goals": normal_goal & _qualifier_flag(qualifiers, "Penalty"),
        "tackles": event_type.eq("Tackle"),
        "tackles_won": event_type.eq("Tackle") & outcome,
        "tackles_lost": event_type.eq("Tackle") & ~outcome,
        "interceptions": event_type.eq("Interception"),
        "clearances": event_type.eq("Clearance"),
        "blocks": event_type.eq("BlockedPass"),
        "recoveries": event_type.eq("BallRecovery"),
        "errors": event_type.eq("Error"),
        "challenges": event_type.eq("Challenge"),
        "dribbled_past": event_type.eq("Challenge"),
        "passes_attempted": pass_event,
        "passes_completed": pass_event & outcome,
        "passes_live": pass_event & ~set_piece,
        "passes_dead": pass_event & set_piece,
        "passes_free_kick": pass_event & _qualifier_flag(qualifiers, "FreeKickTaken"),
        "through_balls": pass_event & _qualifier_flag(qualifiers, "Throughball"),
        "switches": pass_event & (_qualifier_flag(qualifiers, "SwitchOfPlay") | ((end_y - y).abs() >= 50)),
        "crosses": pass_event & _qualifier_flag(qualifiers, "Cross"),
        "throw_ins": pass_event & _qualifier_flag(qualifiers, "ThrowIn"),
        "corners": pass_event & _qualifier_flag(qualifiers, "CornerTaken"),
        "corners_inswinging": pass_event & _qualifier_flag(qualifiers, "Inswinger"),
        "corners_outswinging": pass_event & _qualifier_flag(qualifiers, "Outswinger"),
        "corners_straight": pass_event & _qualifier_flag(qualifiers, "Straight"),
        "passes_offside": event_type.eq("OffsidePass"),
        "passes_blocked": pass_event & _qualifier_flag(qualifiers, "BlockedPass"),
        "key_passes": pass_event & _qualifier_flag(qualifiers, "KeyPass"),
        "progressive_passes": pass_event & outcome & ((end_x - x) >= 10),
        "passes_final_third": pass_event & outcome & (x < 66.7) & (end_x >= 66.7),
        "passes_penalty_area": pass_event & outcome & (end_x >= 83) & end_y.between(21, 79),
        "crosses_penalty_area": pass_event & outcome & _qualifier_flag(qualifiers, "Cross") & (end_x >= 83) & end_y.between(21, 79),
        "touches": work.get("is_touch", pd.Series(False, index=work.index)).fillna(False).astype(bool),
        "touches_defensive_third": work.get("is_touch", pd.Series(False, index=work.index)).fillna(False).astype(bool) & (x < 33.3),
        "touches_middle_third": work.get("is_touch", pd.Series(False, index=work.index)).fillna(False).astype(bool) & x.between(33.3, 66.7, inclusive="left"),
        "touches_attacking_third": work.get("is_touch", pd.Series(False, index=work.index)).fillna(False).astype(bool) & (x >= 66.7),
        "touches_penalty_area": work.get("is_touch", pd.Series(False, index=work.index)).fillna(False).astype(bool) & (x >= 83) & y.between(21, 79),
        "takeons_attempted": event_type.eq("TakeOn"),
        "takeons_successful": event_type.eq("TakeOn") & outcome,
        "dispossessed": event_type.eq("Dispossessed"),
        "fouls_committed": event_type.eq("Foul"),
        "offsides": event_type.eq("OffsideGiven"),
        "yellow_cards": event_type.eq("Card") & work.get("card_type", "").astype("string").str.contains("Yellow", case=False, na=False),
        "red_cards": event_type.eq("Card") & work.get("card_type", "").astype("string").str.contains("Red", case=False, na=False),
        "aerials_total": event_type.eq("Aerial"),
        "aerials_won": event_type.eq("Aerial") & outcome,
        "defensive_aerials": event_type.eq("Aerial") & (x < 50),
        "offensive_aerials": event_type.eq("Aerial") & (x >= 50),
        "saves": event_type.eq("Save"),
        "claims_high": event_type.eq("Claim"),
        "punches": event_type.eq("Punch"),
        "keeper_pickups": event_type.eq("KeeperPickup"),
        "keeper_sweeper_actions": event_type.eq("KeeperSweeper"),
        "smothers": event_type.eq("Smother"),
        "penalties_faced": event_type.eq("PenaltyFaced"),
        "crosses_not_claimed": event_type.eq("CrossNotClaimed"),
    }
    for name, values in flags.items():
        work[f"_m_{name}"] = values.astype("int16")
    metric_columns = [f"_m_{name}" for name in flags]
    grouped = work.groupby(keys, dropna=False)[metric_columns].sum().reset_index()
    grouped = grouped.rename(columns={f"_m_{name}": name for name in flags})
    distances = work.dropna(subset=["_shot_distance"]).groupby(keys, dropna=False)["_shot_distance"].mean().rename("average_shot_distance").reset_index()
    grouped = grouped.merge(distances, on=keys, how="left")

    assist_rows = work[
        event_type.eq("Goal")
        & ~own_goal
        & work["provider_related_player_id"].ne("")
        & _qualifier_flag(qualifiers, "Assisted")
    ][["provider_match_id", "provider_related_player_id", "provider_team_id"]].copy()
    if not assist_rows.empty:
        assists = (
            assist_rows.rename(columns={"provider_related_player_id": "provider_player_id"})
            .groupby(keys).size().rename("assists").reset_index()
        )
        grouped = grouped.merge(assists, on=keys, how="outer")
    else:
        grouped["assists"] = 0

    foul_drawn = work[event_type.eq("Foul") & work["provider_related_player_id"].ne("")][
        ["provider_match_id", "provider_related_player_id", "provider_team_id"]
    ].copy()
    if not foul_drawn.empty:
        # The victim is normally on the opposing team; team is corrected after canonical context joins.
        drawn = foul_drawn.rename(columns={"provider_related_player_id": "provider_player_id"})
        drawn = drawn.groupby(["provider_match_id", "provider_player_id"]).size().rename("fouls_drawn").reset_index()
        grouped = grouped.merge(drawn, on=["provider_match_id", "provider_player_id"], how="left")
    grouped["fouls_drawn"] = grouped.get("fouls_drawn", 0).fillna(0)
    return grouped.fillna({column: 0 for column in flags}), work


def _coalesced_metric(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for candidate in candidates:
        if candidate in frame:
            result = result.fillna(pd.to_numeric(frame[candidate], errors="coerce"))
    return result


def _derive_rates(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    ratios = {
        "pass_completion_pct": ("passes_completed", "passes_attempted"),
        "save_pct": ("saves", "shots_on_target_against"),
        "takeon_success_pct": ("takeons_successful", "takeons_attempted"),
        "shot_on_target_pct": ("shots_on_target", "shots_total"),
        "goals_per_shot": ("goals", "shots_total"),
        "aerial_success_pct": ("aerials_won", "aerials_total"),
        "tackle_success_pct": ("tackles_won", "tackles"),
    }
    for target, (numerator, denominator) in ratios.items():
        if {numerator, denominator} <= set(work.columns):
            den = pd.to_numeric(work[denominator], errors="coerce")
            num = pd.to_numeric(work[numerator], errors="coerce")
            multiplier = 1.0 if target == "goals_per_shot" else 100.0
            work[target] = np.where(den > 0, multiplier * num / den, np.nan)
    if {"aerials_total", "aerials_won"} <= set(work.columns):
        work["aerials_lost"] = work["aerials_total"] - work["aerials_won"]
    return work


def _reconcile_player_stats(
    player_stats_wide: pd.DataFrame,
    event_metrics: pd.DataFrame,
) -> pd.DataFrame:
    stats = player_stats_wide.copy()
    stats["provider_match_id"] = stats["game_id"].map(_provider_id)
    stats["provider_team_id"] = stats["team_id"].map(_provider_id)
    stats["provider_player_id"] = stats["player_id"].map(_provider_id)
    keys = ["provider_match_id", "provider_team_id", "provider_player_id"]
    joined = stats.merge(event_metrics, on=keys, how="inner", suffixes=("_stat", "_event"))
    metric_pairs = {
        "touches": "touches",
        "tackles_total": "tackles",
        "interceptions": "interceptions",
        "clearances": "clearances",
        "passes_total": "passes_attempted",
        "passes_accurate": "passes_completed",
        "shots_total": "shots_total",
        "shots_on_target": "shots_on_target",
        "aerials_total": "aerials_total",
        "aerials_won": "aerials_won",
        "total_saves": "saves",
    }
    rows: list[dict[str, Any]] = []
    for stat_metric, event_metric in metric_pairs.items():
        stat_column = f"{stat_metric}_stat" if stat_metric in event_metrics else stat_metric
        event_column = f"{event_metric}_event" if event_metric in player_stats_wide else event_metric
        if stat_column not in joined or event_column not in joined:
            continue
        left = pd.to_numeric(joined[stat_column], errors="coerce")
        right = pd.to_numeric(joined[event_column], errors="coerce")
        valid = left.notna() & right.notna()
        difference = (left[valid] - right[valid]).abs()
        rows.append(
            {
                "level": "player_match",
                "stat_metric": stat_metric,
                "event_metric": event_metric,
                "compared_rows": int(valid.sum()),
                "matching_rows": int(difference.eq(0).sum()),
                "match_rate": float(difference.eq(0).mean()) if valid.any() else np.nan,
                "mean_absolute_difference": float(difference.mean()) if valid.any() else np.nan,
                "selected_source": "events",
            }
        )
    return pd.DataFrame(rows)


def _clean_position_code(value: Any) -> str:
    """Return a stable provider position code without manufacturing missing text."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalise_fpl_position(value: Any) -> str:
    code = _clean_position_code(value).upper()
    return FPL_POSITION_ALIASES.get(code, "UNK")


def _season_start_year(value: Any) -> int:
    match = re.match(r"^(\d{4})", _clean_position_code(value))
    return int(match.group(1)) if match else -1


def _career_season_record(
    record: Mapping[str, Any],
    season: str,
    *,
    allow_latest_prior: bool = False,
) -> tuple[Mapping[str, Any], str]:
    career = record.get("career", {})
    if not isinstance(career, Mapping):
        return {}, ""
    candidates = [season]
    match = re.fullmatch(r"(\d{4})-(\d{2}|\d{4})", season)
    if match:
        start, end = match.groups()
        candidates.extend([f"{start}-{end[-2:]}", f"{start}-{int(start) + 1}"])
    for candidate in dict.fromkeys(candidates):
        season_record = career.get(candidate)
        if isinstance(season_record, Mapping):
            return season_record, "season"
    if allow_latest_prior:
        target_year = _season_start_year(season)
        prior = [
            (key, value) for key, value in career.items()
            if isinstance(value, Mapping) and _season_start_year(key) <= target_year
        ]
        if prior:
            _, latest = max(prior, key=lambda item: _season_start_year(item[0]))
            return latest, "latest_prior"
    return {}, ""


def _registry_fpl_position(record: Mapping[str, Any], season: str) -> tuple[str, str, float]:
    """Resolve canonical FPL class without using a tactical match position."""
    season_record, record_source = _career_season_record(
        record, season, allow_latest_prior=True
    )
    prefix = "registry" if record_source == "season" else "registry.latest_prior"
    confidence = 1.0 if record_source == "season" else 0.7
    for field in ("fpl_position", "fpl_pos"):
        position = _normalise_fpl_position(season_record.get(field))
        if position != "UNK":
            return position, f"{prefix}.{field}", confidence
    for field in ("position", "position_detail", "pos"):
        raw = _clean_position_code(season_record.get(field)).upper()
        position = REGISTRY_DETAIL_MAP.get(raw, _normalise_fpl_position(raw))
        if position != "UNK":
            return position, f"{prefix}.{field}", confidence - 0.1
    return "UNK", "unresolved", 0.0


def _registry_position_detail(record: Mapping[str, Any], season: str) -> str:
    season_record, _ = _career_season_record(record, season, allow_latest_prior=True)
    for field in ("position_detail", "position", "pos"):
        raw = _clean_position_code(season_record.get(field)).upper()
        if not raw:
            continue
        if raw in POSITION_MAP:
            detail = POSITION_MAP[raw][0]
        elif raw in REGISTRY_DETAIL_MAP:
            detail = raw
        else:
            detail = "UNK"
        if detail != "UNK":
            return detail
    return "UNK"


def _authoritative_fpl_positions(
    official_players: pd.DataFrame,
    master_fpl: Mapping[str, Any],
    season: str,
) -> dict[str, tuple[str, str, float]]:
    """Build current-season FPL positions with official rows taking precedence."""
    positions: dict[str, tuple[str, str, float]] = {}
    if not official_players.empty and "player_id" in official_players:
        position_columns = [
            column for column in ("fpl_pos", "element_type", "position")
            if column in official_players
        ]
        conflicts: dict[str, set[str]] = {}
        for row in official_players.itertuples(index=False):
            player_id = _provider_id(getattr(row, "player_id", None))
            if not player_id:
                continue
            position = "UNK"
            source_column = ""
            for column in position_columns:
                position = _normalise_fpl_position(getattr(row, column, None))
                if position != "UNK":
                    source_column = column
                    break
            if position == "UNK":
                continue
            conflicts.setdefault(player_id, set()).add(position)
            positions[player_id] = (
                position,
                f"fpl.cleaned_players.{source_column}",
                1.0,
            )
        conflicting = {pid: values for pid, values in conflicts.items() if len(values) > 1}
        if conflicting:
            raise ValueError(f"Conflicting official FPL positions by player_id: {conflicting}")

    for player_id, record in master_fpl.items():
        player_key = str(player_id)
        if player_key in positions or not isinstance(record, Mapping):
            continue
        season_record, _ = _career_season_record(record, season)
        for field in ("fpl_pos", "fpl_position", "position", "element_type"):
            position = _normalise_fpl_position(season_record.get(field))
            if position != "UNK":
                positions[player_key] = (
                    position,
                    f"fpl.master_fpl.{field}",
                    0.99,
                )
                break
    return positions


def _primary_position_by_player(frame: pd.DataFrame) -> dict[str, str]:
    """Choose the observed tactical role with most minutes, then starts/rows/recency."""
    if frame.empty or "player_id" not in frame or "position_detail_match" not in frame:
        return {}
    work = frame.copy()
    work["_position"] = work["position_detail_match"].astype("string")
    work = work[work["_position"].notna() & ~work["_position"].isin(["", "UNK", "Sub"])]
    if work.empty:
        return {}
    work["_minutes"] = pd.to_numeric(work.get("minutes", 0), errors="coerce").fillna(0.0)
    work["_start"] = work.get("is_first_eleven", False)
    if not isinstance(work["_start"], pd.Series):
        work["_start"] = False
    work["_start"] = work["_start"].fillna(False).astype(bool).astype(int)
    recency_source = work.get("kickoff_utc", work.get("game_date", pd.Series("", index=work.index)))
    work["_recency"] = pd.to_datetime(recency_source, errors="coerce", utc=True)
    ranked = (
        work.groupby(["player_id", "_position"], dropna=False)
        .agg(
            _minutes=("_minutes", "sum"),
            _starts=("_start", "sum"),
            _appearances=("_position", "size"),
            _recency=("_recency", "max"),
        )
        .reset_index()
        .sort_values(
            ["player_id", "_minutes", "_starts", "_appearances", "_recency", "_position"],
            ascending=[True, False, False, False, False, True],
            na_position="last",
        )
        .drop_duplicates("player_id", keep="first")
    )
    return dict(zip(ranked["player_id"].astype(str), ranked["_position"].astype(str)))


def _resolve_player_positions(
    roster: pd.DataFrame,
    *,
    master_players: Mapping[str, Any],
    season: str,
    authoritative_fpl: Mapping[str, tuple[str, str, float]] | None = None,
) -> pd.DataFrame:
    """Preserve observed roles and attach independently sourced canonical positions."""
    out = roster.copy()
    raw = out.get("position", pd.Series("", index=out.index)).map(_clean_position_code)
    out["provider_position_match"] = raw.replace("", pd.NA).astype("string")
    out["is_starter"] = (~raw.str.casefold().eq("sub")).astype("uint8")
    mapped = raw.map(lambda value: POSITION_MAP.get(value, (value.upper() or "UNK", "UNK")))
    out["position_detail_match"] = mapped.map(lambda pair: pair[0])
    out["position_group_match"] = mapped.map(lambda pair: pair[1])
    # Backwards-compatible position remains the observed, normalized match role.
    out["position"] = out["position_detail_match"]
    observed = out["position_detail_match"].ne("UNK")
    out["position_source"] = np.where(observed, "whoscored.lineup", "unobserved")
    out["position_confidence"] = np.where(observed, 1.0, 0.0)

    primary_map = _primary_position_by_player(out)
    out["primary_position"] = out["player_id"].astype("string").map(primary_map).fillna("UNK")

    registry_fpl: dict[str, tuple[str, str, float]] = {}
    registry_detail: dict[str, str] = {}
    for player_id, record in master_players.items():
        if not isinstance(record, Mapping):
            continue
        registry_fpl[str(player_id)] = _registry_fpl_position(record, season)
        registry_detail[str(player_id)] = _registry_position_detail(record, season)

    player_ids = out["player_id"].astype("string")
    authority = authoritative_fpl or {}
    authority_values = player_ids.map(
        lambda pid: authority.get(str(pid), ("UNK", "unresolved", 0.0))
    )
    registry_values = player_ids.map(lambda pid: registry_fpl.get(str(pid), ("UNK", "unresolved", 0.0)))
    direct_fpl = out["position_group_match"].where(out["position_group_match"].ne("UNK"))
    primary_fpl = out["primary_position"].map(REGISTRY_DETAIL_MAP).where(lambda values: values.ne("UNK"))
    out["fpl_pos"] = authority_values.map(lambda value: value[0])
    out["fpl_position_source"] = authority_values.map(lambda value: value[1])
    out["fpl_position_confidence"] = authority_values.map(lambda value: value[2])
    unresolved = out["fpl_pos"].eq("UNK")
    registry_used = unresolved & registry_values.map(lambda value: value[0]).ne("UNK")
    if registry_used.any():
        out.loc[registry_used, "fpl_pos"] = registry_values[registry_used].map(lambda value: value[0])
        out.loc[registry_used, "fpl_position_source"] = registry_values[registry_used].map(lambda value: value[1])
        out.loc[registry_used, "fpl_position_confidence"] = registry_values[registry_used].map(lambda value: value[2])
    unresolved = out["fpl_pos"].eq("UNK")
    out.loc[unresolved, "fpl_pos"] = direct_fpl[unresolved].fillna(primary_fpl[unresolved]).fillna("UNK")
    direct_used = unresolved & direct_fpl.notna()
    primary_used = unresolved & direct_fpl.isna() & primary_fpl.notna()
    out.loc[direct_used, ["fpl_position_source", "fpl_position_confidence"]] = [
        "whoscored.match_position", 0.85,
    ]
    out.loc[primary_used, ["fpl_position_source", "fpl_position_confidence"]] = [
        "whoscored.season_primary", 0.8,
    ]

    imputed = out["position_detail_match"].where(observed)
    imputation_source = pd.Series("whoscored.match_position", index=out.index, dtype="string")
    needs_imputation = imputed.isna()
    season_primary = out["primary_position"].where(out["primary_position"].ne("UNK"))
    use_primary = needs_imputation & season_primary.notna()
    imputed.loc[use_primary] = season_primary.loc[use_primary]
    imputation_source.loc[use_primary] = "whoscored.season_primary"
    needs_imputation = imputed.isna()
    registry_details = player_ids.map(registry_detail).replace("UNK", pd.NA)
    use_registry = needs_imputation & registry_details.notna()
    imputed.loc[use_registry] = registry_details.loc[use_registry]
    imputation_source.loc[use_registry] = "registry.season_position"
    needs_imputation = imputed.isna()
    broad_fallback = out["fpl_pos"].map({"GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"})
    use_broad = needs_imputation & broad_fallback.notna()
    imputed.loc[use_broad] = broad_fallback.loc[use_broad]
    imputation_source.loc[use_broad] = "fpl.broad_position"
    imputation_source.loc[imputed.isna()] = "unresolved"
    out["position_detail_match_imputed"] = imputed.fillna("UNK")
    out["position_imputation_source"] = imputation_source
    return out


def _build_player_match(
    player_dictionary: pd.DataFrame,
    lineups: pd.DataFrame,
    player_stats_wide: pd.DataFrame,
    event_metrics: pd.DataFrame,
    events: pd.DataFrame,
    *,
    context: pd.DataFrame,
    match_map: Mapping[str, str],
    team_map: Mapping[str, str],
    player_map: Mapping[str, str],
    master_players: Mapping[str, Any],
    authoritative_fpl: Mapping[str, tuple[str, str, float]],
    team_names: Mapping[str, str],
    season: str,
    provider_season: str,
) -> pd.DataFrame:
    roster_parts: list[pd.DataFrame] = []
    for source in (player_dictionary, lineups):
        if not source.empty:
            available = [c for c in ["game_id", "team_id", "team", "player_id", "player", "position", "shirt_no", "lineup_status", "is_first_eleven"] if c in source]
            roster_parts.append(source[available].copy())
    stat_roster = player_stats_wide[[c for c in ["game_id", "team_id", "team", "player_id", "player"] if c in player_stats_wide]].copy()
    roster_parts.append(stat_roster)
    roster = pd.concat(roster_parts, ignore_index=True, sort=False)
    roster["provider_match_id"] = roster["game_id"].map(_provider_id)
    roster["provider_team_id"] = roster["team_id"].map(_provider_id)
    roster["provider_player_id"] = roster["player_id"].map(_provider_id)
    roster = roster[roster["provider_player_id"].ne("")]
    roster = roster.sort_values("is_first_eleven", ascending=False, na_position="last").drop_duplicates(
        ["provider_match_id", "provider_player_id"], keep="first"
    )
    roster["provider_player_name"] = roster.get("player", "").map(_repair_text)
    roster["provider_team_name"] = roster.get("team", "").map(_repair_text)
    roster["match_id"] = roster["provider_match_id"].map(match_map)
    roster["team_id"] = roster["provider_team_id"].map(team_map)
    roster["player_id"] = roster["provider_player_id"].map(player_map)
    canonical_player_names = {
        str(player_id): str(record.get("name", ""))
        for player_id, record in master_players.items()
    }
    roster["player"] = roster["player_id"].map(canonical_player_names).fillna(roster["provider_player_name"])
    roster["nation"] = roster["player_id"].map(
        {str(player_id): record.get("nation") for player_id, record in master_players.items()}
    )
    roster["born"] = roster["player_id"].map(
        {str(player_id): record.get("born") for player_id, record in master_players.items()}
    )

    stats = player_stats_wide.copy()
    stats["provider_match_id"] = stats["game_id"].map(_provider_id)
    stats["provider_team_id"] = stats["team_id"].map(_provider_id)
    stats["provider_player_id"] = stats["player_id"].map(_provider_id)
    stats = stats.drop(columns=[c for c in ["game_id", "team_id", "team", "player_id", "player"] if c in stats])
    roster = roster.merge(stats, on=["provider_match_id", "provider_team_id", "provider_player_id"], how="left", validate="one_to_one")
    event_key_columns = {"provider_match_id", "provider_team_id", "provider_player_id"}
    event_names = set(event_metrics.columns) - event_key_columns
    event_metrics_for_join = event_metrics.rename(
        columns={name: f"{name}_event" for name in event_names}
    )
    roster = roster.merge(
        event_metrics_for_join,
        on=["provider_match_id", "provider_team_id", "provider_player_id"],
        how="left",
    )

    stat_sources = {
        "rating": ["ratings"], "is_man_of_the_match": ["is_man_of_the_match"],
        "touches": ["touches_event", "touches"],
        "tackles": ["tackles_event", "tackles_total"],
        "tackles_won": ["tackles_won_event", "tackle_successful"],
        "tackles_lost": ["tackles_lost_event", "tackle_unsuccesful"],
        "interceptions": ["interceptions_event", "interceptions"],
        "clearances": ["clearances_event", "clearances"],
        "errors": ["errors_event", "errors"],
        "recoveries": ["recoveries_event"],
        "blocks": ["blocks_event", "shots_blocked"],
        "dribbled_past": ["dribbled_past_event", "dribbled_past"],
        "passes_completed": ["passes_completed_event", "passes_accurate"],
        "passes_attempted": ["passes_attempted_event", "passes_total"],
        "key_passes": ["key_passes_event", "passes_key"],
        "takeons_attempted": ["takeons_attempted_event", "dribbles_attempted"],
        "takeons_successful": ["takeons_successful_event", "dribbles_won"],
        "dispossessed": ["dispossessed_event", "dispossessed"],
        "aerials_total": ["aerials_total_event", "aerials_total"],
        "aerials_won": ["aerials_won_event", "aerials_won"],
        "defensive_aerials": ["defensive_aerials_event", "defensive_aerials"],
        "offensive_aerials": ["offensive_aerials_event", "offensive_aerials"],
        "saves": ["saves_event", "total_saves"],
        "claims_high": ["claims_high_event", "claims_high"],
        "collected": ["collected"], "parried_danger": ["parried_danger"],
        "parried_safe": ["parried_safe"],
    }
    for name in event_names:
        stat_sources.setdefault(name, [f"{name}_event"])
    derived_metrics = pd.DataFrame(
        {
            target: _coalesced_metric(roster, candidates)
            for target, candidates in stat_sources.items()
        },
        index=roster.index,
    )
    roster = pd.concat(
        [roster.drop(columns=list(derived_metrics.columns), errors="ignore"), derived_metrics],
        axis=1,
    )

    # Minutes from lineup/substitution state, capped at the regulation match duration.
    sub = events[events["type"].isin(["SubstitutionOn", "SubstitutionOff"])].copy()
    sub["provider_match_id"] = sub["game_id"].map(_provider_id)
    sub["provider_player_id"] = sub["player_id"].map(_provider_id)
    sub["sub_minute"] = pd.to_numeric(sub.get("minute"), errors="coerce").clip(0, 120)
    on = sub[sub["type"].eq("SubstitutionOn")].groupby(["provider_match_id", "provider_player_id"])["sub_minute"].min()
    off = sub[sub["type"].eq("SubstitutionOff")].groupby(["provider_match_id", "provider_player_id"])["sub_minute"].min()
    extra_time = events.get("period", pd.Series("", index=events.index)).astype("string").str.contains("Extra", case=False, na=False)
    duration = extra_time.groupby(events["game_id"].map(_provider_id)).any().map({True: 120.0, False: 90.0})
    idx = pd.MultiIndex.from_frame(roster[["provider_match_id", "provider_player_id"]])
    start = np.where(roster.get("is_first_eleven", False).fillna(False).astype(bool), 0.0, pd.Series(idx.map(on), index=roster.index).fillna(90.0))
    end = pd.Series(idx.map(off), index=roster.index).fillna(roster["provider_match_id"].map(duration).fillna(90.0))
    match_duration = roster["provider_match_id"].map(duration).fillna(90.0)
    end = pd.Series(np.minimum(end, match_duration), index=roster.index)
    start = np.minimum(start, match_duration)
    roster["minutes"] = (end - start).clip(lower=0, upper=120)
    roster.loc[roster.get("lineup_status", "").astype("string").eq("substitute") & ~roster["provider_player_id"].isin(sub["provider_player_id"]), "minutes"] = 0

    meta = context.drop_duplicates("provider_match_id").copy()
    keep = [
        "provider_match_id", "game", "game_date", "kickoff_utc", "gameweek",
        "home_team_id", "away_team_id", "provider_home_team_id",
        "provider_away_team_id", "home", "away", "home_score", "away_score",
        "status",
    ]
    roster = roster.merge(meta[[c for c in keep if c in meta]], on="provider_match_id", how="left", validate="many_to_one")
    roster["opponent_id"] = np.where(roster["team_id"].eq(roster["home_team_id"]), roster["away_team_id"], roster["home_team_id"])
    roster["provider_opponent_team_id"] = np.where(
        roster["provider_team_id"].eq(roster["provider_home_team_id"]),
        roster["provider_away_team_id"],
        roster["provider_home_team_id"],
    )
    roster["team"] = roster["team_id"].map(team_names)
    roster["opponent"] = roster["opponent_id"].map(team_names)
    roster["is_home"] = roster["team_id"].eq(roster["home_team_id"])
    roster["goals_against"] = np.where(
        roster["is_home"], roster.get("away_score"), roster.get("home_score")
    )
    if "shots_on_target" in event_metrics:
        team_sot = event_metrics.groupby(
            ["provider_match_id", "provider_team_id"], dropna=False
        )["shots_on_target"].sum(min_count=1)
        opponent_index = pd.MultiIndex.from_frame(
            roster[["provider_match_id", "provider_opponent_team_id"]].rename(
                columns={"provider_opponent_team_id": "provider_team_id"}
            )
        )
        roster["shots_on_target_against"] = opponent_index.map(team_sot)
    roster["league"] = context["league"].dropna().iloc[0]
    roster["season"] = season
    roster["provider_season"] = provider_season
    roster = _resolve_player_positions(
        roster,
        master_players=master_players,
        season=season,
        authoritative_fpl=authoritative_fpl,
    )
    roster["provider"] = PROVIDER
    roster["processing_version"] = PROCESSING_VERSION
    roster["metric_definition_version"] = PROCESSING_VERSION
    roster["metric_source_policy"] = "events_primary_match_stats_metadata"
    roster["coverage_status"] = "covered"
    roster = _derive_rates(roster)
    roster["carries"] = np.nan
    roster["progressive_carries"] = np.nan
    return roster


def _build_team_match(
    team_stats_wide: pd.DataFrame,
    player_match: pd.DataFrame,
    *,
    context: pd.DataFrame,
    match_map: Mapping[str, str],
    team_map: Mapping[str, str],
    team_names: Mapping[str, str],
    season: str,
    provider_season: str,
) -> pd.DataFrame:
    work = team_stats_wide.copy()
    work["provider_match_id"] = work["game_id"].map(_provider_id)
    work["provider_team_id"] = work["team_id"].map(_provider_id)
    work["match_id"] = work["provider_match_id"].map(match_map)
    work["team_id"] = work["provider_team_id"].map(team_map)
    work["provider_team_name"] = work.get("team", "").map(_repair_text)
    work["team"] = work["team_id"].map(team_names)
    work = work.drop(columns=[c for c in ["game_id"] if c in work])

    additive = sorted({metric for metrics in PLAYER_TABLES.values() for metric in metrics} - NON_SUM_METRICS)
    available = [metric for metric in additive if metric in player_match]
    summed = player_match.groupby(["provider_match_id", "provider_team_id"], dropna=False)[available].sum(min_count=1).reset_index()
    work = work.merge(summed, on=["provider_match_id", "provider_team_id"], how="outer", suffixes=("", "_players"))
    direct = {
        "rating": ["ratings"], "possession": ["possession"], "average_age": ["average_age"],
        "touches": ["touches_players", "touches"],
        "tackles": ["tackles_players", "tackles_total"],
        "tackles_won": ["tackles_won_players", "tackle_successful"],
        "tackles_lost": ["tackles_lost_players", "tackle_unsuccesful"],
        "interceptions": ["interceptions_players", "interceptions"],
        "clearances": ["clearances_players", "clearances"],
        "errors": ["errors_players", "errors"],
        "passes_completed": ["passes_completed_players", "passes_accurate"],
        "passes_attempted": ["passes_attempted_players", "passes_total"],
        "key_passes": ["key_passes_players", "passes_key"],
        "takeons_attempted": ["takeons_attempted_players", "dribbles_attempted"],
        "takeons_successful": ["takeons_successful_players", "dribbles_won"],
        "dispossessed": ["dispossessed_players", "dispossessed"],
        "aerials_total": ["aerials_total_players", "aerials_total"],
        "aerials_won": ["aerials_won_players", "aerials_won"],
        "defensive_aerials": ["defensive_aerials_players", "defensive_aerials"],
        "offensive_aerials": ["offensive_aerials_players", "offensive_aerials"],
        "shots_total": ["shots_total_players", "shots_total"],
        "shots_on_target": ["shots_on_target_players", "shots_on_target"],
        "shots_off_target": ["shots_off_target_players", "shots_off_target"],
        "shots_blocked": ["shots_blocked_players", "shots_blocked"],
        "shots_on_post": ["shots_on_post_players", "shots_on_post"],
    }
    source_policy = dict(direct)
    for metric in available:
        source_policy[metric] = [f"{metric}_players", metric]
    derived_metrics = pd.DataFrame(
        {
            target: _coalesced_metric(work, candidates)
            for target, candidates in source_policy.items()
        },
        index=work.index,
    )
    work = pd.concat(
        [work.drop(columns=list(derived_metrics.columns), errors="ignore"), derived_metrics],
        axis=1,
    )

    possession_numeric = pd.to_numeric(work.get("possession"), errors="coerce")
    possession_sum = possession_numeric.groupby(work["provider_match_id"]).transform("sum")
    work["possession"] = possession_numeric.where(possession_sum.between(99.0, 101.0))
    work = work.copy()

    meta_cols = [
        "provider_match_id", "game", "game_date", "kickoff_utc", "gameweek", "status",
        "home_team_id", "away_team_id", "home", "away", "home_score",
        "away_score", "score", "venue", "referee", "attendance", "home_formation",
        "away_formation",
    ]
    meta = context[[c for c in meta_cols if c in context]].drop_duplicates("provider_match_id")
    work = work.merge(meta, on="provider_match_id", how="left", validate="many_to_one")
    work["opponent_id"] = np.where(work["team_id"].eq(work["home_team_id"]), work["away_team_id"], work["home_team_id"])
    work["opponent"] = work["opponent_id"].map(team_names)
    work["is_home"] = work["team_id"].eq(work["home_team_id"])
    work["formation"] = np.where(work["is_home"], work.get("home_formation"), work.get("away_formation"))
    work["opponent_formation"] = np.where(work["is_home"], work.get("away_formation"), work.get("home_formation"))
    work["goals"] = np.where(work["is_home"], work.get("home_score"), work.get("away_score"))
    work["goals_against"] = np.where(work["is_home"], work.get("away_score"), work.get("home_score"))
    work["shots_on_target_against"] = work.groupby("provider_match_id")["shots_on_target"].transform(lambda values: values.iloc[::-1].to_numpy() if len(values) == 2 else np.full(len(values), np.nan))
    work["league"] = context["league"].dropna().iloc[0]
    work["season"] = season
    work["provider_season"] = provider_season
    work["provider"] = PROVIDER
    work["processing_version"] = PROCESSING_VERSION
    work["metric_definition_version"] = PROCESSING_VERSION
    work["metric_source_policy"] = "player_event_sums_primary_team_stats_metadata"
    work["coverage_status"] = "covered"
    return _derive_rates(work)


def _build_schedule_table(
    context: pd.DataFrame,
    *,
    match_map: Mapping[str, str],
    team_names: Mapping[str, str],
    season: str,
    provider_season: str,
) -> pd.DataFrame:
    base = context.drop_duplicates("provider_match_id").copy()
    base["match_id"] = base["provider_match_id"].map(match_map)
    common = [
        "league", "game", "game_date", "kickoff_utc", "gameweek", "match_id",
        "provider_match_id", "home_team_id", "away_team_id", "status", "home_score",
        "away_score", "score", "venue", "referee", "attendance",
    ]
    common = [column for column in common if column in base]
    home = base[common].copy()
    home["team_id"] = base["home_team_id"]
    home["provider_team_id"] = base["provider_home_team_id"]
    home["provider_team_name"] = base.get("home_team")
    home["opponent_id"] = base["away_team_id"]
    home["is_home"] = True
    home["formation"] = base.get("home_formation")
    home["opponent_formation"] = base.get("away_formation")
    away = base[common].copy()
    away["team_id"] = base["away_team_id"]
    away["provider_team_id"] = base["provider_away_team_id"]
    away["provider_team_name"] = base.get("away_team")
    away["opponent_id"] = base["home_team_id"]
    away["is_home"] = False
    away["formation"] = base.get("away_formation")
    away["opponent_formation"] = base.get("home_formation")
    out = pd.concat([home, away], ignore_index=True)
    out["team"] = out["team_id"].map(team_names)
    out["opponent"] = out["opponent_id"].map(team_names)
    out["home"] = out["home_team_id"].map(team_names)
    out["away"] = out["away_team_id"].map(team_names)
    out["season"] = season
    out["provider_season"] = provider_season
    out["provider"] = PROVIDER
    out["processing_version"] = PROCESSING_VERSION
    out["metric_definition_version"] = PROCESSING_VERSION
    out["metric_source_policy"] = "schedule_and_registry_fixture"
    return out


def _ordered(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    provenance = [
        "provider", "processing_version", "metric_definition_version",
        "metric_source_policy", "coverage_status",
    ]
    ordered = list(dict.fromkeys(columns + provenance))
    missing = [column for column in ordered if column not in frame]
    if missing:
        frame = pd.concat(
            [frame, pd.DataFrame(np.nan, index=frame.index, columns=missing)],
            axis=1,
        )
    return frame[ordered].copy()


def _season_table(frame: pd.DataFrame, *, player: bool, metrics: list[str]) -> pd.DataFrame:
    entity_key = ["player_id"] if player else ["team_id"]
    base = ["league", "season", "provider_season", *entity_key]
    work = frame.copy()
    numeric_metrics = [metric for metric in metrics if metric in work and pd.api.types.is_numeric_dtype(work[metric])]
    sum_metrics = [metric for metric in numeric_metrics if metric not in NON_SUM_METRICS]
    grouped = work.groupby(base, dropna=False)
    out = grouped[sum_metrics].sum(min_count=1).reset_index() if sum_metrics else grouped.size().rename("_rows").reset_index().drop(columns="_rows")
    if player:
        def representative(values: pd.Series) -> Any:
            non_null = values.dropna()
            if non_null.empty:
                return pd.NA
            modes = non_null.mode()
            return modes.iloc[0] if not modes.empty else non_null.iloc[0]

        metadata_columns = [
            column for column in (
                "player", "nation", "born", "position", "primary_position", "fpl_pos",
                "fpl_position_source", "fpl_position_confidence",
            )
            if column in work
        ]
        metadata = grouped[metadata_columns].agg(representative).reset_index()
        out = out.merge(metadata, on=base, how="left", validate="one_to_one")
        if "primary_position" in out:
            primary = out["primary_position"].astype("string")
        else:
            temporary = work.rename(columns={"position": "position_detail_match"})
            primary_map = _primary_position_by_player(temporary)
            primary = out["player_id"].astype("string").map(primary_map).fillna("UNK")
            out["primary_position"] = primary
        # At season grain, `position` means the minutes-weighted primary tactical role.
        out["position"] = primary.where(primary.ne("UNK"), out.get("position", "UNK"))
    match_counts = grouped["match_id"].nunique().rename("matches_covered").reset_index()
    out = out.merge(match_counts, on=base, how="left", validate="one_to_one")
    if player:
        played = grouped["minutes"].apply(
            lambda values: int((pd.to_numeric(values, errors="coerce") > 0).sum())
        ).rename("matches_played").reset_index()
        out = out.merge(played, on=base, how="left", validate="one_to_one")
        if "minutes" not in out and "minutes" in work:
            minutes = grouped["minutes"].sum(min_count=1).rename("minutes").reset_index()
            out = out.merge(minutes, on=base, how="left", validate="one_to_one")
    else:
        out["matches_played"] = out["matches_covered"]
    for metric in numeric_metrics:
        if metric in NON_SUM_METRICS and metric not in {
            "pass_completion_pct", "save_pct", "takeon_success_pct", "shot_on_target_pct",
            "goals_per_shot", "aerial_success_pct", "tackle_success_pct",
        }:
            out[metric] = grouped[metric].mean().to_numpy()
    out = _derive_rates(out)
    out["provider"] = PROVIDER
    out["processing_version"] = PROCESSING_VERSION
    out["metric_definition_version"] = PROCESSING_VERSION
    out["metric_source_policy"] = "aggregated_from_whoscored_match_tables"
    out["coverage_status"] = "partial_or_complete_from_match_coverage"
    return out


def _write_table_families(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    level: str,
    player: bool,
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    id_columns = PLAYER_ID_COLUMNS if player and level == "match" else ID_COLUMNS
    for table, metrics in PLAYER_TABLES.items():
        if not player and table == "keepers":
            output_name = "keeper"
        else:
            output_name = table
        if level == "match":
            table_frame = _ordered(frame.copy(), id_columns + metrics)
        else:
            table_frame = _season_table(frame, player=player, metrics=metrics)
            identity = ["league", "season", "provider_season"]
            identity += [
                "player_id", "player", "nation", "born", "position", "primary_position",
                "fpl_pos", "fpl_position_source", "fpl_position_confidence",
            ] if player else ["team_id"]
            table_frame = _ordered(table_frame, identity + ["matches_played", "matches_covered"] + metrics)
        path = output_dir / f"{'player' if player else 'team'}_{level}" / f"{output_name}.csv"
        _atomic_csv(table_frame, path)
        key = ["match_id", "player_id"] if player and level == "match" else ["match_id", "team_id"] if level == "match" else ["season", "player_id"] if player else ["season", "team_id"]
        manifests.append(
            {
                "table": f"{'player' if player else 'team'}_{level}/{output_name}",
                "path": str(path),
                "rows": len(table_frame),
                "duplicate_keys": int(table_frame.duplicated(key).sum()),
                "sha256": _hash_file(path),
            }
        )
    return manifests


def clean_whoscored_season(
    *,
    raw_root: Path,
    out_root: Path,
    league: str,
    season: str,
    registry_root: Path,
    teams_config_path: Path,
    fpl_root: Path = Path("data/processed/fpl"),
    master_fpl_path: Path | None = None,
    player_aliases_path: Path = Path("data/config/player_spelling_aliases.json"),
    allow_partial: bool = False,
    min_event_coverage: float = 1.0,
    strict_identities: bool = True,
    write_events_parquet: bool = True,
    rebuild_provider_bridges: bool = False,
    force: bool = False,
) -> CleanResult:
    canonical_season, provider_season = normalize_season(season)
    source_dir = raw_root / "WhoScored" / league / provider_season
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    output_dir = out_root / league / canonical_season
    if output_dir.exists() and any(output_dir.rglob("*.csv")) and not force:
        raise FileExistsError(f"WhoScored output already exists at {output_dir}; pass --force to replace tables.")

    schedule = _drop_export_index(_read_csv(source_dir / "ws_schedule.csv"))
    player_dictionary = _drop_export_index(_read_csv(source_dir / "derived" / "player_dictionary.csv"))
    lineups = _drop_export_index(_read_csv(source_dir / "derived" / "lineups.csv"))
    match_info_path = source_dir / "derived" / "match_info.csv"
    match_info = _drop_export_index(_read_csv(match_info_path)) if match_info_path.is_file() else pd.DataFrame()
    player_stats_raw = _drop_export_index(_read_csv(source_dir / "stats" / "player_match_stats.csv"))
    team_stats_raw = _drop_export_index(_read_csv(source_dir / "stats" / "team_match_stats.csv"))
    events_raw = _drop_export_index(_read_csv(source_dir / "events" / "events.csv"))

    bridge_root = registry_root / "bridges"
    identity_columns = [
        "entity_type", "provider", "provider_id", "provider_name", "canonical_id",
        "valid_from", "valid_to", "match_method", "match_confidence", "review_status",
    ]
    match_bridge_columns = [
        "provider", "provider_match_id", "match_id", "provider_game",
        "match_method", "match_confidence",
    ]
    team_bridge_path = bridge_root / "team_ids.csv"
    player_bridge_path = bridge_root / "player_ids.csv"
    match_bridge_path = bridge_root / "match_ids.csv"
    existing_team = _existing_bridge(team_bridge_path, identity_columns)
    existing_player = _existing_bridge(player_bridge_path, identity_columns)
    existing_match = _existing_bridge(match_bridge_path, match_bridge_columns)
    if rebuild_provider_bridges:
        existing_team = existing_team[
            ~existing_team["provider"].astype(str).str.lower().eq(PROVIDER)
        ].copy()
        existing_player = existing_player[
            ~existing_player["provider"].astype(str).str.lower().eq(PROVIDER)
        ].copy()
        existing_match = existing_match[
            ~existing_match["provider"].astype(str).str.lower().eq(PROVIDER)
        ].copy()
    master_players = _load_json(registry_root / "master_players.json")
    master_teams = _load_json(registry_root / "master_teams.json")
    official_fpl_path = fpl_root / canonical_season / "season" / "cleaned_players.csv"
    official_fpl_players = (
        _drop_export_index(_read_csv(official_fpl_path))
        if official_fpl_path.is_file()
        else pd.DataFrame()
    )
    resolved_master_fpl_path = master_fpl_path or (registry_root / "master_fpl.json")
    master_fpl: Mapping[str, Any] = {}
    if resolved_master_fpl_path.is_file():
        try:
            loaded_master_fpl = _load_json(resolved_master_fpl_path)
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning(
                "Ignoring unreadable optional FPL master %s: %s",
                resolved_master_fpl_path,
                exc,
            )
        else:
            if isinstance(loaded_master_fpl, Mapping):
                master_fpl = loaded_master_fpl
            else:
                LOG.warning(
                    "Ignoring optional FPL master %s because its root is not an object.",
                    resolved_master_fpl_path,
                )
    authoritative_fpl = _authoritative_fpl_positions(
        official_fpl_players,
        master_fpl,
        canonical_season,
    )
    player_names = {
        str(player_id): str(record.get("name", player_id))
        for player_id, record in master_players.items()
    }
    team_names = {
        str(team_id): str(record.get("name", team_id))
        for team_id, record in master_teams.items()
    }

    raw_teams = pd.concat(
        [
            schedule[["home_team_id", "home_team"]].rename(columns={"home_team_id": "provider_team_id", "home_team": "provider_team_name"}),
            schedule[["away_team_id", "away_team"]].rename(columns={"away_team_id": "provider_team_id", "away_team": "provider_team_name"}),
        ],
        ignore_index=True,
    )
    team_audit, team_bridges = _team_resolution(
        raw_teams,
        teams_config=_load_json(teams_config_path),
        team_lookup=_load_json(registry_root / "_id_lookup_teams.json"),
        existing=existing_team,
    )
    team_map = dict(zip(team_bridges["provider_id"].astype(str), team_bridges["canonical_id"].astype(str)))

    fixture_path = registry_root / "fixtures" / canonical_season / "fixture_calendar.csv"
    schedule_context, match_audit, match_bridges = _match_resolution(
        schedule,
        season=canonical_season,
        team_map=team_map,
        fixture_path=fixture_path,
        existing=existing_match,
    )
    whoscored_match_bridges = match_bridges.loc[
        match_bridges["provider"].astype(str).str.lower().eq(PROVIDER)
    ]
    match_map = dict(zip(
        whoscored_match_bridges["provider_match_id"].astype(str),
        whoscored_match_bridges["match_id"].astype(str),
    ))
    context = schedule_context.copy()
    context["home"] = context["home_team_id"].map(team_names)
    context["away"] = context["away_team_id"].map(team_names)
    if not match_info.empty:
        info = match_info.copy()
        info["provider_match_id"] = info["game_id"].map(_provider_id)
        extra = [c for c in ["provider_match_id", "venue", "referee", "attendance", "home_formation", "away_formation", "has_events"] if c in info]
        context = context.merge(info[extra].drop_duplicates("provider_match_id"), on="provider_match_id", how="left", suffixes=("", "_info"))

    raw_players = player_dictionary[["player_id", "player", "team_id"]].rename(
        columns={
            "player_id": "provider_player_id",
            "player": "provider_player_name",
            "team_id": "provider_team_id",
        }
    )
    event_players = events_raw[["player_id", "player", "team_id"]].dropna(subset=["player_id"]).rename(
        columns={
            "player_id": "provider_player_id",
            "player": "provider_player_name",
            "team_id": "provider_team_id",
        }
    )
    raw_players = pd.concat([raw_players, event_players], ignore_index=True).drop_duplicates(
        ["provider_player_id", "provider_team_id"]
    )
    player_audit, player_bridges = _player_resolution(
        raw_players,
        player_lookup=_load_json(registry_root / "_id_lookup_players.json"),
        master_players=master_players,
        official_fpl_players=official_fpl_players,
        player_aliases=_load_json(player_aliases_path) if player_aliases_path.is_file() else {},
        team_map=team_map,
        season=canonical_season,
        existing=existing_player,
    )
    player_map = dict(zip(player_bridges["provider_id"].astype(str), player_bridges["canonical_id"].astype(str)))

    unresolved_counts = {
        "teams": int(team_audit["canonical_id"].isna().sum()),
        "matches": int(match_audit["match_id"].isna().sum()),
        "players": int(player_audit["canonical_id"].isna().sum()),
    }
    if strict_identities and any(unresolved_counts.values()):
        audit_dir = output_dir / "audits"
        _atomic_csv(team_audit, audit_dir / "team_identity_resolution.csv")
        _atomic_csv(match_audit, audit_dir / "match_identity_resolution.csv")
        _atomic_csv(player_audit, audit_dir / "player_identity_resolution.csv")
        raise ValueError(f"Unresolved WhoScored identities: {unresolved_counts}. Review audit files or use --no-strict-identities for a partial diagnostic run.")

    scheduled_ids = set(schedule["game_id"].map(_provider_id))
    event_ids = set(events_raw["game_id"].map(_provider_id))
    player_stat_ids = set(player_stats_raw["game_id"].map(_provider_id))
    team_stat_ids = set(team_stats_raw["game_id"].map(_provider_id))
    team_count = len(set(schedule["home_team_id"].map(_provider_id)) | set(schedule["away_team_id"].map(_provider_id)))
    expected_matches = team_count * (team_count - 1) if team_count >= 2 else len(scheduled_ids)
    event_coverage = len(scheduled_ids & event_ids) / len(scheduled_ids) if scheduled_ids else 0.0
    coverage = pd.DataFrame(
        [{
            "league": league, "season": canonical_season, "provider_season": provider_season,
            "teams": team_count, "expected_matches": expected_matches,
            "scheduled_matches": len(scheduled_ids), "event_matches": len(event_ids),
            "player_stat_matches": len(player_stat_ids), "team_stat_matches": len(team_stat_ids),
            "covered_schedule_matches": len(scheduled_ids & event_ids),
            "missing_event_matches": len(scheduled_ids - event_ids),
            "schedule_complete": len(scheduled_ids) >= expected_matches,
            "event_coverage": event_coverage, "min_event_coverage": min_event_coverage,
            "coverage_status": "complete" if len(scheduled_ids) >= expected_matches and event_coverage >= min_event_coverage else "partial",
        }]
    )
    if not allow_partial and coverage.iloc[0]["coverage_status"] != "complete":
        _atomic_csv(coverage, output_dir / "audits" / "coverage.csv")
        raise ValueError(
            f"Incomplete WhoScored coverage: schedule={len(scheduled_ids)}/{expected_matches}, "
            f"events={len(scheduled_ids & event_ids)}/{len(scheduled_ids)} ({event_coverage:.1%}). "
            "Pass --allow-partial to publish explicitly partial tables."
        )

    player_stats_wide, player_conflicts = _pivot_stats(player_stats_raw, keys=["game_id", "team_id", "player_id", "player"])
    team_stats_wide, team_conflicts = _pivot_stats(team_stats_raw, keys=["game_id", "team_id", "team"])
    normalized_events = _normalize_events(
        events_raw,
        match_map=match_map,
        team_map=team_map,
        player_map=player_map,
        player_names=player_names,
        team_names=team_names,
        context=context,
        season=canonical_season,
        provider_season=provider_season,
    )
    event_player_applicable = normalized_events["provider_player_id"].ne("")
    event_player_unresolved = event_player_applicable & normalized_events["player_id"].isna()
    coverage["events_with_provider_player"] = int(event_player_applicable.sum())
    coverage["events_with_unresolved_player_id"] = int(event_player_unresolved.sum())
    coverage["resolved_event_player_identity_rate"] = (
        float((~event_player_unresolved[event_player_applicable]).mean())
        if event_player_applicable.any()
        else 0.0
    )
    unresolved_event_players = normalized_events.loc[
        event_player_unresolved,
        [
            "provider_player_id", "provider_player_name", "provider_team_id",
            "provider_team_name", "player_identity_status",
        ],
    ].drop_duplicates()
    event_metrics, _ = _event_aggregates(normalized_events)
    metric_reconciliation = _reconcile_player_stats(player_stats_wide, event_metrics)
    player_match = _build_player_match(
        player_dictionary, lineups, player_stats_wide, event_metrics, events_raw,
        context=context, match_map=match_map, team_map=team_map, player_map=player_map,
        master_players=master_players, authoritative_fpl=authoritative_fpl,
        team_names=team_names,
        season=canonical_season, provider_season=provider_season,
    )
    unresolved_player_match = player_match[
        player_match[["match_id", "player_id", "team_id"]].isna().any(axis=1)
    ].copy()
    coverage["candidate_player_match_rows"] = len(player_match)
    coverage["published_player_match_rows"] = len(player_match) - len(unresolved_player_match)
    coverage["excluded_unresolved_player_match_rows"] = len(unresolved_player_match)
    coverage["resolved_player_identity_rate"] = (
        float(player_audit["canonical_id"].notna().mean()) if len(player_audit) else 0.0
    )
    player_match = player_match.dropna(subset=["match_id", "player_id", "team_id"]).copy()
    identity_collisions = player_match[
        player_match.duplicated(["match_id", "player_id"], keep=False)
    ].sort_values(["match_id", "player_id", "provider_player_id"])
    if not identity_collisions.empty:
        collision_path = output_dir / "audits" / "player_match_identity_collisions.csv"
        _atomic_csv(identity_collisions, collision_path)
        raise ValueError(
            f"{len(identity_collisions)} WhoScored rows collide on canonical "
            f"(match_id, player_id); review {collision_path}."
        )
    team_match = _build_team_match(
        team_stats_wide, player_match, context=context, match_map=match_map,
        team_map=team_map, team_names=team_names,
        season=canonical_season, provider_season=provider_season,
    )
    team_match = team_match.dropna(subset=["match_id", "team_id"]).copy()

    coverage_status = str(coverage.iloc[0]["coverage_status"])
    player_match["coverage_status"] = coverage_status
    team_match["coverage_status"] = coverage_status
    normalized_events["coverage_status"] = coverage_status

    manifest_rows: list[dict[str, Any]] = []
    manifest_rows += _write_table_families(player_match, output_dir=output_dir, level="match", player=True)
    manifest_rows += _write_table_families(player_match, output_dir=output_dir, level="season", player=True)
    manifest_rows += _write_table_families(team_match, output_dir=output_dir, level="match", player=False)
    manifest_rows += _write_table_families(team_match, output_dir=output_dir, level="season", player=False)

    schedule_columns = ID_COLUMNS + TEAM_EXTRA_TABLES["schedule"]
    schedule_table = _ordered(
        _build_schedule_table(
            context,
            match_map=match_map,
            team_names=team_names,
            season=canonical_season,
            provider_season=provider_season,
        ),
        schedule_columns,
    )
    schedule_table["coverage_status"] = coverage_status
    schedule_path = output_dir / "team_match" / "schedule.csv"
    _atomic_csv(schedule_table, schedule_path)
    manifest_rows.append({"table": "team_match/schedule", "path": str(schedule_path), "rows": len(schedule_table), "duplicate_keys": int(schedule_table.duplicated(["match_id", "team_id"]).sum()), "sha256": _hash_file(schedule_path)})

    shot_zone_columns = [c for c in team_match if str(c).startswith("shot_zones.")]
    shot_zones = _ordered(team_match.copy(), ID_COLUMNS + shot_zone_columns)
    shot_zone_path = output_dir / "team_match" / "shot_zones.csv"
    _atomic_csv(shot_zones, shot_zone_path)
    manifest_rows.append({"table": "team_match/shot_zones", "path": str(shot_zone_path), "rows": len(shot_zones), "duplicate_keys": int(shot_zones.duplicated(["match_id", "team_id"]).sum()), "sha256": _hash_file(shot_zone_path)})

    shot_zone_season = _season_table(
        team_match,
        player=False,
        metrics=shot_zone_columns,
    )
    shot_zone_season = _ordered(
        shot_zone_season,
        ["league", "season", "provider_season", "team_id", "matches_played", "matches_covered", *shot_zone_columns],
    )
    shot_zone_season_path = output_dir / "team_season" / "shot_zones.csv"
    _atomic_csv(shot_zone_season, shot_zone_season_path)
    manifest_rows.append({"table": "team_season/shot_zones", "path": str(shot_zone_season_path), "rows": len(shot_zone_season), "duplicate_keys": int(shot_zone_season.duplicated(["season", "team_id"]).sum()), "sha256": _hash_file(shot_zone_season_path)})

    for column in ("shot_creating_actions", "goal_creating_actions"):
        team_match[column] = np.nan
    gsc_metrics = TEAM_EXTRA_TABLES["goal_shot_creation"]
    gsc_match = _ordered(team_match.copy(), ID_COLUMNS + gsc_metrics)
    gsc_match_path = output_dir / "team_match" / "goal_shot_creation.csv"
    _atomic_csv(gsc_match, gsc_match_path)
    manifest_rows.append({"table": "team_match/goal_shot_creation", "path": str(gsc_match_path), "rows": len(gsc_match), "duplicate_keys": int(gsc_match.duplicated(["match_id", "team_id"]).sum()), "sha256": _hash_file(gsc_match_path)})
    gsc_season = _season_table(team_match, player=False, metrics=gsc_metrics)
    gsc_season = _ordered(gsc_season, ["league", "season", "provider_season", "team_id", "matches_played", "matches_covered", *gsc_metrics])
    gsc_season_path = output_dir / "team_season" / "goal_shot_creation.csv"
    _atomic_csv(gsc_season, gsc_season_path)
    manifest_rows.append({"table": "team_season/goal_shot_creation", "path": str(gsc_season_path), "rows": len(gsc_season), "duplicate_keys": int(gsc_season.duplicated(["season", "team_id"]).sum()), "sha256": _hash_file(gsc_season_path)})

    event_dir = output_dir / "events"
    if write_events_parquet:
        event_dir.mkdir(parents=True, exist_ok=True)
        event_path = event_dir / "normalized_events.parquet"
        normalized_events.to_parquet(event_path, index=False)
        manifest_rows.append({"table": "events/normalized_events", "path": str(event_path), "rows": len(normalized_events), "duplicate_keys": int(normalized_events.get("event_id", pd.Series(dtype="object")).duplicated().sum()), "sha256": _hash_file(event_path)})
    event_summary = normalized_events.groupby(["event_bin", "type"], dropna=False).size().rename("event_count").reset_index().sort_values(["event_bin", "event_count"], ascending=[True, False])
    event_summary_path = event_dir / "event_summary.csv"
    _atomic_csv(event_summary, event_summary_path)
    manifest_rows.append({"table": "events/event_summary", "path": str(event_summary_path), "rows": len(event_summary), "duplicate_keys": 0, "sha256": _hash_file(event_summary_path)})

    audit_dir = output_dir / "audits"
    _atomic_csv(coverage, audit_dir / "coverage.csv")
    _atomic_csv(team_audit, audit_dir / "team_identity_resolution.csv")
    _atomic_csv(match_audit, audit_dir / "match_identity_resolution.csv")
    _atomic_csv(player_audit, audit_dir / "player_identity_resolution.csv")
    unresolved_columns = [
        column
        for column in (
            "provider_match_id", "provider_player_id", "provider_player_name",
            "provider_team_id", "provider_team_name", "game", "game_date",
        )
        if column in unresolved_player_match
    ]
    _atomic_csv(
        unresolved_player_match[unresolved_columns].drop_duplicates(),
        audit_dir / "excluded_unresolved_player_match_rows.csv",
    )
    _atomic_csv(
        unresolved_event_players,
        audit_dir / "unresolved_event_player_identities.csv",
    )
    stat_conflicts = pd.concat([
        player_conflicts.assign(level="player_match"),
        team_conflicts.assign(level="team_match"),
    ], ignore_index=True, sort=False)
    _atomic_csv(stat_conflicts, audit_dir / "stat_value_conflicts.csv")

    consumed = {
        "ratings", "is_man_of_the_match", "touches", "tackles_total", "tackle_successful",
        "tackle_unsuccesful", "interceptions", "clearances", "errors", "shots_blocked",
        "dribbled_past", "passes_accurate", "passes_total", "passes_key", "dribbles_attempted",
        "dribbles_won", "dispossessed", "aerials_total", "aerials_won", "defensive_aerials",
        "offensive_aerials", "total_saves", "claims_high", "collected", "parried_danger",
        "parried_safe", "possession", "average_age", "shots_total", "shots_on_target",
        "shots_off_target", "shots_on_post",
    }
    all_stats = pd.concat([
        player_stats_raw[["stat_group", "stat_name", "stat_key"]].assign(level="player_match"),
        team_stats_raw[["stat_group", "stat_name", "stat_key"]].assign(level="team_match"),
    ], ignore_index=True).drop_duplicates()
    all_stats["normalized_stat_key"] = all_stats["stat_key"].astype("string").str.replace(r"^stats\.", "", regex=True)
    unmapped = all_stats[~all_stats["normalized_stat_key"].isin(consumed)].sort_values(["level", "stat_group", "stat_key"])
    _atomic_csv(unmapped, audit_dir / "unmapped_statistics.csv")

    _atomic_csv(metric_reconciliation, audit_dir / "metric_reconciliation.csv")

    _atomic_csv(team_bridges, team_bridge_path)
    _atomic_csv(player_bridges, player_bridge_path)
    _atomic_csv(match_bridges, match_bridge_path)

    table_manifest = pd.DataFrame(manifest_rows)
    _atomic_csv(table_manifest, audit_dir / "table_manifest.csv")
    run_meta = {
        "provider": PROVIDER,
        "processing_version": PROCESSING_VERSION,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "league": league,
        "season": canonical_season,
        "provider_season": provider_season,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "official_fpl_positions_path": str(official_fpl_path),
        "master_fpl_path": str(resolved_master_fpl_path),
        "authoritative_fpl_positions": len(authoritative_fpl),
        "coverage_status": coverage_status,
        "allow_partial": allow_partial,
        "strict_identities": strict_identities,
        "unresolved_identities": unresolved_counts,
        "tables": len(table_manifest),
    }
    _atomic_json(run_meta, output_dir / "run_metadata.json")
    return CleanResult(output_dir, table_manifest, coverage, player_audit, team_audit, match_audit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean native WhoScored match statistics and events into registry-keyed table families.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/whoscored"))
    parser.add_argument("--out-root", type=Path, default=Path("data/processed/whoscored"))
    parser.add_argument("--league", required=True)
    parser.add_argument("--season", required=True, help="Split-year season, e.g. 2025-2026.")
    parser.add_argument("--registry-root", type=Path, default=Path("data/processed/registry"))
    parser.add_argument("--fpl-root", type=Path, default=Path("data/processed/fpl"))
    parser.add_argument("--master-fpl", type=Path, default=None)
    parser.add_argument("--teams-config", type=Path, default=Path("data/config/teams.json"))
    parser.add_argument("--player-aliases", type=Path, default=Path("data/config/player_spelling_aliases.json"))
    parser.add_argument("--allow-partial", action="store_true", help="Publish outputs with explicit partial coverage metadata.")
    parser.add_argument("--min-event-coverage", type=float, default=1.0)
    parser.add_argument("--strict-identities", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-events-parquet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rebuild-provider-bridges",
        action="store_true",
        help="Re-resolve WhoScored bridges while preserving rows for other providers.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    result = clean_whoscored_season(
        raw_root=args.raw_root,
        out_root=args.out_root,
        league=args.league,
        season=args.season,
        registry_root=args.registry_root,
        fpl_root=args.fpl_root,
        master_fpl_path=args.master_fpl,
        teams_config_path=args.teams_config,
        player_aliases_path=args.player_aliases,
        allow_partial=args.allow_partial,
        min_event_coverage=args.min_event_coverage,
        strict_identities=args.strict_identities,
        write_events_parquet=args.write_events_parquet,
        rebuild_provider_bridges=args.rebuild_provider_bridges,
        force=args.force,
    )
    LOG.info("WhoScored clean completed: output=%s tables=%d coverage=%s", result.output_dir, len(result.table_manifest), result.coverage.iloc[0]["coverage_status"])


if __name__ == "__main__":
    main()
