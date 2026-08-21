from __future__ import annotations

import argparse
import json
import logging
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from fpl_assistant.providers.fpl.pipelines.clean_and_enrich import (
    PLAYER_BRIDGE_COLUMNS,
    canonical,
    load_fpl_code_registry,
    load_overrides,
    normalise_fpl_position,
    normalized_provider_code,
    season_shortform,
)
from fpl_assistant.canonical.identity import stable_canonical_id


ELIGIBILITY_COLUMNS = [
    "season",
    "player_id",
    "player",
    "pos",
    "team_id",
    "valid_from",
    "valid_until",
    "registered",
    "confirmed_unavailable",
    "unavailable_reason",
    "information_timestamp",
    "source",
    "timestamp_safe",
    "reconstruction_method",
    "first_match_id",
    "last_match_id",
    "fixtures_observed",
]

PANEL_PROVENANCE_COLUMNS = [
    "registered",
    "confirmed_unavailable",
    "unavailable_reason",
    "eligible_for_fixture",
    "information_timestamp",
    "eligibility_source",
    "eligibility_timestamp_safe",
    "named_on_bench",
    "did_not_play",
    "observation_status",
    "row_source",
]

FINISHED_STATUSES = {"complete", "completed", "finished", "ft", "aet", "pen"}
CONFIRMED_OUT_FPL_STATUSES = {"i", "s", "u"}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_timestamp(value: str | pd.Timestamp | None = None) -> pd.Timestamp:
    timestamp = pd.Timestamp(value or datetime.now(timezone.utc))
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _file_timestamp(path: Path) -> pd.Timestamp:
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")


def _iso(timestamp: pd.Timestamp) -> str:
    return timestamp.isoformat().replace("+00:00", "Z")


def _fixture_key(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")


def normalize_fixture_calendar(frame: pd.DataFrame, season: str) -> pd.DataFrame:
    fixtures = frame.copy()
    if "match_id" not in fixtures and "fbref_id" in fixtures:
        fixtures["match_id"] = fixtures["fbref_id"]
    if "fbref_id" not in fixtures and "match_id" in fixtures:
        fixtures["fbref_id"] = fixtures["match_id"]
    required = {"fpl_id", "match_id", "team_id", "opponent_id", "date_sched", "status"}
    missing = sorted(required - set(fixtures.columns))
    if missing:
        raise ValueError(f"{season}: fixture calendar missing columns: {missing}")
    fixtures["season"] = season
    fixtures["_fixture_key"] = _fixture_key(fixtures["fpl_id"])
    fixtures["_fixture_time"] = pd.to_datetime(
        fixtures.get("date_played", ""), errors="coerce", utc=True
    ).fillna(pd.to_datetime(fixtures["date_sched"], errors="coerce", utc=True))
    if fixtures["_fixture_time"].isna().any():
        raise ValueError(f"{season}: fixture calendar contains unparseable dates")
    duplicates = fixtures.duplicated(["match_id", "team_id"], keep=False)
    if duplicates.any():
        raise ValueError(f"{season}: duplicate (match_id, team_id) fixture rows")
    return fixtures


def _global_fpl_code_registry(registry_root: Path, processed_fpl_root: Path) -> dict[str, str]:
    mapping, _ = load_fpl_code_registry(
        registry_root,
        processed_fpl_root,
        target_season="9999-10000",
    )
    return mapping


def _resolve_and_publish_historical_identities(
    season: str,
    *,
    raw_players: pd.DataFrame,
    fpl: pd.DataFrame,
    registry_root: Path,
    code_registry: dict[str, str],
    league: str = "ENG-Premier League",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve stable FPL codes and publish any missing canonical identities."""
    master_path = registry_root / "master_players.json"
    master_fpl_path = registry_root / "master_fpl.json"
    master_teams_path = registry_root / "master_teams.json"
    lookup_path = registry_root / "_id_lookup_players.json"
    bridge_path = registry_root / "bridges" / "player_ids.csv"
    master = json.loads(master_path.read_text(encoding="utf-8")) if master_path.is_file() else {}
    master_fpl = json.loads(master_fpl_path.read_text(encoding="utf-8")) if master_fpl_path.is_file() else {}
    master_teams = json.loads(master_teams_path.read_text(encoding="utf-8")) if master_teams_path.is_file() else {}
    lookup = json.loads(lookup_path.read_text(encoding="utf-8")) if lookup_path.is_file() else {}
    overrides = load_overrides(registry_root / "overrides.json")
    bridges = _read_csv(bridge_path) if bridge_path.is_file() else pd.DataFrame(columns=PLAYER_BRIDGE_COLUMNS)
    for column in PLAYER_BRIDGE_COLUMNS:
        if column not in bridges:
            bridges[column] = ""
    bridges = bridges[PLAYER_BRIDGE_COLUMNS].copy()
    fpl_bridges = bridges[bridges["provider"].str.lower().eq("fpl")]
    bridge_counts = fpl_bridges.groupby("provider_id")["canonical_id"].nunique()
    if (bridge_counts > 1).any():
        raise ValueError("FPL player bridge contains conflicting canonical IDs")
    bridge_map = (
        fpl_bridges.drop_duplicates("provider_id", keep="last")
        .set_index("provider_id")["canonical_id"]
        .to_dict()
    )

    identity_rows: list[dict[str, Any]] = []
    methods: dict[str, int] = {}
    long_season = season
    short_season = season_shortform(season)
    for row in raw_players.to_dict("records"):
        element = str(row.get("id", "")).strip()
        code = normalized_provider_code(row.get("code"))
        first_name = str(row.get("first_name", "") or "").strip()
        second_name = str(row.get("second_name", "") or "").strip()
        name = f"{first_name} {second_name}".strip() or str(row.get("web_name", "") or "").strip()
        name_key = canonical(name)
        method = ""
        player_id = str(bridge_map.get(code, "") or "")
        if player_id:
            method = "provider_bridge"
        if not player_id:
            player_id = str(overrides.get(name_key, "") or "")
            if player_id:
                method = "reviewed_override"
        if not player_id:
            lookup_id = str(lookup.get(name_key, "") or "")
            if lookup_id in master:
                player_id = lookup_id
                method = "canonical_name"
        if not player_id:
            candidate = str(code_registry.get(code, "") or "")
            candidate_name = canonical((master.get(candidate) or {}).get("name", ""))
            if candidate in master and candidate_name == name_key:
                player_id = candidate
                method = "historical_provider_code"
        if not player_id:
            player_id = stable_canonical_id("player", "fpl", code, length=8)
            existing_name = canonical((master.get(player_id) or {}).get("name", ""))
            if existing_name and existing_name != name_key:
                player_id = stable_canonical_id("player", "fpl", code, length=12)
            method = "generated_from_fpl_code"

        if player_id not in master:
            master[player_id] = {
                "name": name,
                "nation": None,
                "born": row.get("birth_date") or None,
                "career": {},
            }
        if not lookup.get(name_key):
            lookup[name_key] = player_id
        if code and code not in bridge_map:
            bridges.loc[len(bridges)] = {
                "entity_type": "player",
                "provider": "fpl",
                "provider_id": code,
                "provider_name": name,
                "canonical_id": player_id,
                "valid_from": long_season,
                "valid_to": "",
                "match_method": method,
                "match_confidence": "1.0" if method != "generated_from_fpl_code" else "0.8",
                "review_status": "approved" if method != "generated_from_fpl_code" else "needs_review",
            }
            bridge_map[code] = player_id
        methods[method] = methods.get(method, 0) + 1
        identity_rows.append(
            {
                "element": element,
                "fpl_code": code,
                "player_id": player_id,
                "canonical_player_name": (master.get(player_id) or {}).get("name") or name,
                "raw_player_name": name,
                "fpl_pos": normalise_fpl_position(row.get("element_type")),
                "identity_method": method,
            }
        )

    identities = pd.DataFrame(identity_rows)
    if identities["element"].duplicated().any() or identities["fpl_code"].duplicated().any():
        raise ValueError(f"{season}: raw FPL roster has duplicate element/code identities")
    element_map = identities.set_index("element")
    fpl = fpl.copy()
    fpl["player_id"] = fpl["element"].astype(str).map(element_map["player_id"])
    fpl["name"] = fpl["element"].astype(str).map(element_map["canonical_player_name"])
    fpl["fpl_code"] = fpl["element"].astype(str).map(element_map["fpl_code"])
    resolved_pos = fpl["element"].astype(str).map(element_map["fpl_pos"])
    if "fpl_pos" not in fpl:
        fpl["fpl_pos"] = resolved_pos
    else:
        fpl["fpl_pos"] = resolved_pos.fillna(fpl["fpl_pos"])
    if fpl["player_id"].isna().any():
        raise ValueError(f"{season}: FPL gameweek rows contain unknown elements")

    # Publish FPL-owned season membership without discarding richer historical
    # provider fields already present in the canonical player records.
    sort_columns = [column for column in ("kickoff_time", "round", "fixture") if column in fpl]
    ordered = fpl.sort_values(sort_columns) if sort_columns else fpl
    for player_id, group in ordered.groupby("player_id", sort=True):
        last = group.iloc[-1]
        record = master[str(player_id)]
        career = record.setdefault("career", {})
        season_record = dict(career.get(long_season) or {})
        fpl_pos = normalise_fpl_position(last.get("fpl_pos")) or str(last.get("fpl_pos", ""))
        season_record.update(
            {
                "team": str(last.get("team", "")),
                "team_id": str(last.get("team_id", "")),
                "fpl_position": fpl_pos,
                "league": league,
            }
        )
        season_record.setdefault("position", str(last.get("position", "")) or fpl_pos)
        season_record.setdefault("position_detail", "UNK")
        teams = [
            {"team": str(team), "team_id": str(team_id)}
            for team_id, team in group[["team_id", "team"]].drop_duplicates().itertuples(index=False, name=None)
        ]
        if len(teams) > 1:
            season_record["teams"] = teams
        career[long_season] = season_record

        fpl_record = dict(master_fpl.get(str(player_id)) or {})
        fpl_record["name"] = fpl_record.get("name") or record.get("name")
        fpl_record["player_id"] = str(player_id)
        fpl_record.setdefault("nation", record.get("nation"))
        fpl_record.setdefault("born", record.get("born"))
        fpl_record.setdefault("career", {})[short_season] = dict(season_record)
        master_fpl[str(player_id)] = fpl_record

    for (team_id, team), group in ordered.groupby(["team_id", "team"], sort=True):
        team_id = str(team_id)
        team_record = dict(master_teams.get(team_id) or {"name": str(team), "career": {}})
        team_record.setdefault("career", {})[long_season] = {
            "league": league,
            "players": [
                {"id": str(player_id), "name": str(master[str(player_id)].get("name", ""))}
                for player_id in sorted(group["player_id"].astype(str).unique())
            ],
        }
        master_teams[team_id] = team_record

    bridge_conflicts = bridges.groupby(["provider", "provider_id"])["canonical_id"].nunique()
    if (bridge_conflicts > 1).any():
        raise ValueError("Historical identity publication created conflicting provider bridges")
    _atomic_write_json(master, master_path)
    _atomic_write_json(master_fpl, master_fpl_path)
    _atomic_write_json(master_teams, master_teams_path)
    _atomic_write_json(lookup, lookup_path)
    _atomic_write_csv(
        bridges.sort_values(["provider", "provider_id"], kind="stable"),
        bridge_path,
    )
    return fpl, {f"identity_{key}": value for key, value in sorted(methods.items())}


def load_historical_fpl_universe(
    season: str,
    *,
    merged_path: Path,
    players_raw_path: Path,
    code_registry: dict[str, str],
    registry_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fpl = _read_csv(merged_path)
    required = {"fixture", "player_id", "team_id", "minutes", "element"}
    missing = sorted(required - set(fpl.columns))
    if missing:
        raise ValueError(f"{season}: merged_gws.csv missing columns: {missing}")

    raw_players = _read_csv(players_raw_path)
    if not {"id", "code"}.issubset(raw_players.columns):
        raise ValueError(f"{season}: players_raw.csv lacks id/code identity fields")
    original_ids = fpl["player_id"].astype(str)
    fpl, identity_audit = _resolve_and_publish_historical_identities(
        season,
        raw_players=raw_players,
        fpl=fpl,
        registry_root=registry_root,
        code_registry=code_registry,
    )
    changed = fpl["player_id"].astype(str).ne(original_ids)
    fpl["season"] = season
    fpl["_fixture_key"] = _fixture_key(fpl["fixture"])

    key = ["_fixture_key", "player_id", "team_id"]
    duplicate_rows = fpl.duplicated(key, keep=False)
    if duplicate_rows.any():
        preview = fpl.loc[duplicate_rows, key + ["element", "name"]].head(20)
        raise ValueError(
            f"{season}: canonical identity repair still leaves duplicate player-fixture rows: "
            f"{preview.to_dict('records')}"
        )
    audit = {
        "source_rows": len(fpl),
        "provider_codes_resolved": int(fpl["fpl_code"].ne("").sum()),
        "canonical_ids_repaired": int(changed.sum()),
        "information_timestamp": _iso(_file_timestamp(merged_path)),
        "source": "fpl_merged_gws_retrospective",
        "timestamp_safe": False,
        **identity_audit,
    }
    return fpl, audit


def load_preseason_fpl_universe(
    season: str,
    *,
    roster_path: Path,
    fixtures: pd.DataFrame,
    information_timestamp: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    roster = _read_csv(roster_path)
    required = {"player_id", "team_id", "name", "fpl_pos"}
    missing = sorted(required - set(roster.columns))
    if missing:
        raise ValueError(f"{season}: cleaned preseason roster missing columns: {missing}")
    if roster[list(required)].eq("").any(axis=None):
        raise ValueError(f"{season}: cleaned preseason roster has incomplete identities")

    universe = roster.merge(
        fixtures[["_fixture_key", "fpl_id", "team_id"]],
        on="team_id",
        how="inner",
        validate="many_to_many",
    )
    universe["season"] = season
    universe["fixture"] = universe["fpl_id"]
    universe["player"] = universe["name"]
    universe["pos"] = universe.get("position", universe["fpl_pos"])
    universe["minutes"] = pd.NA
    universe["starts"] = pd.NA
    audit = {
        "source_rows": len(universe),
        "roster_players": int(roster["player_id"].nunique()),
        "information_timestamp": _iso(information_timestamp),
        "source": "fpl_bootstrap_roster_snapshot",
        "timestamp_safe": True,
    }
    return universe, audit


def build_availability_snapshot(
    season: str,
    *,
    raw_players_path: Path,
    roster_path: Path,
    information_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    raw = _read_csv(raw_players_path)
    roster = _read_csv(roster_path)
    join_keys = (
        ("id", "fpl_element_id")
        if "fpl_element_id" in roster.columns
        else ("code", "fpl_code")
    )
    left_key, right_key = join_keys
    if left_key not in raw or right_key not in roster:
        raise ValueError(f"{season}: cannot join raw FPL availability to canonical roster")
    snapshot = roster[[right_key, "player_id", "team_id", "name", "fpl_pos"]].merge(
        raw,
        left_on=right_key,
        right_on=left_key,
        how="left",
        validate="one_to_one",
        suffixes=("", "_raw"),
    )
    status = snapshot.get("status", pd.Series("", index=snapshot.index)).astype(str).str.lower()
    snapshot["season"] = season
    snapshot["player"] = snapshot["name"]
    snapshot["pos"] = snapshot["fpl_pos"]
    snapshot["availability_status"] = status.map(
        {"a": "available", "d": "doubtful", "i": "injured", "s": "suspended", "u": "unavailable"}
    ).fillna("unknown")
    snapshot["confirmed_unavailable"] = status.isin(CONFIRMED_OUT_FPL_STATUSES)
    snapshot["unavailable_reason"] = snapshot.get(
        "news", pd.Series("", index=snapshot.index)
    ).fillna("")
    snapshot["information_timestamp"] = _iso(information_timestamp)
    snapshot["source"] = "official_fpl_bootstrap"
    snapshot["source_path"] = str(raw_players_path)
    columns = [
        "season", "player_id", "player", "pos", "team_id",
        "availability_status", "confirmed_unavailable", "unavailable_reason",
        "information_timestamp", "source", "source_path",
    ]
    return snapshot[columns].sort_values(["team_id", "player_id"]).reset_index(drop=True)


def _observed_calendar_source(calendar_path: Path) -> tuple[pd.DataFrame, Path]:
    archive = calendar_path.with_name("player_fixture_calendar_observed.csv")
    current = _read_csv(calendar_path) if calendar_path.is_file() else pd.DataFrame()
    current_is_expanded = "eligibility_source" in current.columns or "row_source" in current.columns
    if current_is_expanded and archive.is_file():
        return _read_csv(archive), archive
    return current, calendar_path


def _normalize_observed(frame: pd.DataFrame) -> pd.DataFrame:
    observed = frame.copy()
    if observed.empty:
        return observed
    if "match_id" not in observed and "fbref_id" in observed:
        observed["match_id"] = observed["fbref_id"]
    if "fbref_id" not in observed and "match_id" in observed:
        observed["fbref_id"] = observed["match_id"]
    for column in ("match_id", "player_id", "team_id"):
        if column not in observed:
            raise ValueError(f"Observed player calendar lacks {column}")
        observed[column] = observed[column].astype(str)
    if observed.duplicated(["match_id", "player_id", "team_id"]).any():
        raise ValueError("Observed player calendar has duplicate canonical keys")
    return observed


def expand_player_fixture_calendar(
    season: str,
    *,
    fixtures: pd.DataFrame,
    universe: pd.DataFrame,
    observed: pd.DataFrame,
    availability: pd.DataFrame | None,
    information_timestamp: str,
    eligibility_source: str,
    timestamp_safe: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bridge_columns = [
        "_fixture_key", "match_id", "fbref_id", "fpl_id", "gw_orig", "gw_played",
        "date_sched", "date_played", "team_id", "opponent_id", "team", "venue",
        "is_home", "gf", "ga", "xg", "xga", "status",
    ]
    bridge = fixtures[[c for c in bridge_columns if c in fixtures.columns]].copy()
    fpl = universe.merge(
        bridge,
        on=["_fixture_key", "team_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_fixture"),
    )
    if fpl["match_id"].eq("").any() or fpl["match_id"].isna().any():
        raise ValueError(f"{season}: FPL universe contains rows without canonical match IDs")
    fpl["match_id"] = fpl["match_id"].astype(str)
    fpl["player_id"] = fpl["player_id"].astype(str)
    fpl["team_id"] = fpl["team_id"].astype(str)
    key = ["match_id", "player_id", "team_id"]
    if fpl.duplicated(key).any():
        raise ValueError(f"{season}: FPL universe has duplicate canonical player-fixture keys")

    observed = _normalize_observed(observed)
    observed_ids_remapped = 0
    observed_ids_fuzzy_remapped = 0
    if not observed.empty and "player" in observed.columns:
        fpl_names = fpl[key + [c for c in ("name", "player") if c in fpl.columns]].copy()
        fpl_name_column = "name" if "name" in fpl_names else "player"
        fpl_names["_name_key"] = fpl_names[fpl_name_column].map(canonical)
        name_counts = fpl_names.groupby(["match_id", "team_id", "_name_key"])["player_id"].nunique()
        unique_names = name_counts[name_counts.eq(1)].index
        name_map = (
            fpl_names.set_index(["match_id", "team_id", "_name_key"])["player_id"]
            .loc[unique_names]
            .to_dict()
        )
        observed["_name_key"] = observed["player"].map(canonical)
        candidate_ids = pd.Series(
            [
                name_map.get((match_id, team_id, name_key))
                for match_id, team_id, name_key in observed[
                    ["match_id", "team_id", "_name_key"]
                ].itertuples(index=False, name=None)
            ],
            index=observed.index,
            dtype="string",
        )
        current_keys = pd.MultiIndex.from_frame(observed[key])
        fpl_keys_before = pd.MultiIndex.from_frame(fpl[key])
        remap = ~current_keys.isin(fpl_keys_before) & candidate_ids.notna()
        observed_ids_remapped = int(remap.sum())
        observed.loc[remap, "player_id"] = candidate_ids.loc[remap]
        observed = observed.drop(columns="_name_key")

        # Provider display names and official FPL names often differ (for
        # example, "Dani Ceballos" versus the registered full name). Resolve
        # remaining rows only within the same match/team and, when known, the
        # same observed minute total. A strict score and margin avoid guessing
        # between genuinely ambiguous squad members.
        current_keys = pd.MultiIndex.from_frame(observed[key])
        unresolved_indices = observed.index[~current_keys.isin(fpl_keys_before)]
        fpl_candidates = {
            group_key: group.copy()
            for group_key, group in fpl.groupby(["match_id", "team_id"], sort=False)
        }
        for index in unresolved_indices:
            row = observed.loc[index]
            candidates = fpl_candidates.get((row["match_id"], row["team_id"]))
            if candidates is None or candidates.empty:
                continue
            observed_minutes = pd.to_numeric(
                pd.Series([row.get("minutes")]), errors="coerce"
            ).iloc[0]
            observed_name = canonical(str(row.get("player", "")))
            scored: list[tuple[float, str]] = []
            for candidate in candidates.to_dict("records"):
                candidate_name = canonical(
                    str(candidate.get("name") or candidate.get("player") or "")
                )
                if not observed_name or not candidate_name:
                    continue
                score = SequenceMatcher(None, observed_name, candidate_name).ratio()
                if observed_name in candidate_name or candidate_name in observed_name:
                    score = max(score, 0.92)
                candidate_minutes = pd.to_numeric(
                    pd.Series([candidate.get("minutes")]), errors="coerce"
                ).iloc[0]
                if pd.notna(observed_minutes) and pd.notna(candidate_minutes):
                    score += 0.05 if candidate_minutes == observed_minutes else -0.02
                scored.append((score, str(candidate["player_id"])))
            scored.sort(reverse=True)
            if not scored:
                continue
            best_score, best_id = scored[0]
            runner_up = scored[1][0] if len(scored) > 1 else 0.0
            if best_score >= 0.72 and (len(scored) == 1 or best_score - runner_up >= 0.12):
                observed.at[index, "player_id"] = best_id
                observed_ids_fuzzy_remapped += 1
        if observed.duplicated(key).any():
            observed["_completeness"] = observed.replace("", pd.NA).notna().sum(axis=1)
            observed = (
                observed.sort_values("_completeness", ascending=False)
                .drop_duplicates(key, keep="first")
                .drop(columns="_completeness")
            )
    fpl_index = pd.MultiIndex.from_frame(fpl[key])
    observed_index = (
        pd.MultiIndex.from_frame(observed[key])
        if not observed.empty
        else pd.MultiIndex.from_arrays([[], [], []], names=key)
    )
    all_index = fpl_index.union(observed_index)
    output = observed.set_index(key).reindex(all_index).reset_index() if not observed.empty else all_index.to_frame(index=False)
    for column in observed.columns:
        if column not in output.columns:
            output[column] = pd.NA
    output["_had_provider_observation"] = pd.MultiIndex.from_frame(output[key]).isin(observed_index)
    output["_had_fpl_roster_row"] = pd.MultiIndex.from_frame(output[key]).isin(fpl_index)

    fpl_keyed = fpl.set_index(key).reindex(pd.MultiIndex.from_frame(output[key]))
    fixture_fill_columns = [
        "fbref_id", "fpl_id", "gw_orig", "gw_played", "date_sched", "date_played",
        "opponent_id", "team", "venue", "is_home", "gf", "ga", "xg", "xga", "status",
    ]
    for column in fixture_fill_columns:
        if column not in fpl_keyed:
            continue
        values = pd.Series(fpl_keyed[column].to_numpy(), index=output.index)
        if column not in output:
            output[column] = values
        else:
            output[column] = output[column].replace("", pd.NA).fillna(values)
    output["fbref_id"] = output.get("fbref_id", output["match_id"]).replace("", pd.NA).fillna(output["match_id"])
    derived_home = pd.to_numeric(output.get("is_home"), errors="coerce")
    existing_home = pd.to_numeric(output.get("was_home"), errors="coerce")
    output["was_home"] = derived_home.fillna(existing_home).astype("Int64")

    identity_map = {"name": "player", "player": "player", "position": "pos", "fpl_pos": "pos"}
    for source, target in identity_map.items():
        if source not in fpl_keyed:
            continue
        values = pd.Series(fpl_keyed[source].to_numpy(), index=output.index).replace("", pd.NA)
        if target not in output:
            output[target] = values
        else:
            output[target] = output[target].replace("", pd.NA).fillna(values)

    authoritative_stats = {
        "minutes": "minutes",
        "starts": "is_starter",
        "xP": "xp",
        "total_points": "total_points",
        "bonus": "bonus",
        "bps": "bps",
        "clean_sheets": "clean_sheets",
        "yellow_cards": "yellow_crd",
        "red_cards": "red_crd",
        "own_goals": "own_goals",
        "saves": "saves",
        "value": "price",
    }
    for source, target in authoritative_stats.items():
        if source not in fpl_keyed:
            continue
        values = pd.Series(fpl_keyed[source].to_numpy(), index=output.index).replace("", pd.NA)
        fpl_mask = output["_had_fpl_roster_row"] & values.notna()
        if target not in output:
            output[target] = pd.NA
        output.loc[fpl_mask, target] = values.loc[fpl_mask]

    status = output.get("status", pd.Series("", index=output.index)).astype(str).str.lower()
    completed = status.isin(FINISHED_STATUSES) | output.get(
        "date_played", pd.Series("", index=output.index)
    ).astype(str).ne("")
    output["minutes"] = pd.to_numeric(output.get("minutes"), errors="coerce")
    minutes = output["minutes"]
    output.loc[~completed, "minutes"] = pd.NA
    minutes = pd.to_numeric(output.get("minutes"), errors="coerce")
    output.loc[completed & minutes.isna(), "minutes"] = 0
    minutes = pd.to_numeric(output["minutes"], errors="coerce")

    if "is_starter" not in output:
        output["is_starter"] = pd.NA
    output["is_starter"] = pd.to_numeric(output["is_starter"], errors="coerce")
    output.loc[completed & minutes.eq(0), "is_starter"] = 0
    output.loc[~completed, "is_starter"] = pd.NA
    output["named_on_bench"] = (
        completed & minutes.eq(0) & output["_had_provider_observation"]
    )
    output["did_not_play"] = pd.Series(pd.NA, index=output.index, dtype="boolean")
    output.loc[completed, "did_not_play"] = minutes.loc[completed].fillna(0).eq(0)
    output["observation_status"] = np.select(
        [
            ~completed,
            completed & minutes.gt(0),
            completed & output["named_on_bench"],
        ],
        ["fixture_pending", "played", "unused_substitute"],
        default="not_in_matchday_squad",
    )
    output["is_active"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output.loc[completed, "is_active"] = minutes.loc[completed].fillna(0).gt(0).astype(int)

    if "starter_source" not in output:
        output["starter_source"] = pd.NA
    reconstructed_dnp = completed & minutes.eq(0) & ~output["_had_provider_observation"]
    output.loc[reconstructed_dnp, "starter_source"] = "reconstructed_fpl_dnp"
    output.loc[~completed, "starter_source"] = "pending"
    is_goalkeeper = output.get("pos", pd.Series("", index=output.index)).astype(str).str.upper().str.contains("GK")
    goalkeeper_started = completed & is_goalkeeper & minutes.gt(0)
    goalkeeper_needs_imputation = goalkeeper_started & pd.to_numeric(
        output["is_starter"], errors="coerce"
    ).fillna(0).eq(0)
    output.loc[goalkeeper_started, "is_starter"] = 1
    output.loc[goalkeeper_needs_imputation, "starter_source"] = "imputed"

    output["registered"] = True
    output["confirmed_unavailable"] = False
    output["unavailable_reason"] = ""
    if availability is not None and not availability.empty:
        availability_keyed = availability.drop_duplicates("player_id", keep="last").set_index("player_id")
        output["confirmed_unavailable"] = output["player_id"].map(
            availability_keyed["confirmed_unavailable"]
        ).fillna(False).astype(bool)
        output["unavailable_reason"] = output["player_id"].map(
            availability_keyed["unavailable_reason"]
        ).fillna("")
    output["eligible_for_fixture"] = output["registered"] & ~output["confirmed_unavailable"]
    output["information_timestamp"] = information_timestamp
    output["eligibility_source"] = eligibility_source
    output["eligibility_timestamp_safe"] = bool(timestamp_safe)
    output["row_source"] = np.select(
        [
            output["_had_fpl_roster_row"] & output["_had_provider_observation"],
            output["_had_fpl_roster_row"] & completed & minutes.eq(0),
            output["_had_fpl_roster_row"] & ~completed,
        ],
        ["fpl_roster_plus_provider", "fpl_eligible_dnp", "fpl_preseason_roster"],
        default="provider_only_observation",
    )

    fixture_time = pd.to_datetime(
        output.get("date_played", ""), errors="coerce", utc=True
    ).fillna(pd.to_datetime(output.get("date_sched", ""), errors="coerce", utc=True))
    output["_fixture_time"] = fixture_time
    output = output.sort_values(["player_id", "_fixture_time", "match_id"]).reset_index(drop=True)
    output["days_since_last"] = (
        output.groupby("player_id")["_fixture_time"].diff().dt.total_seconds().div(86400).fillna(0).clip(lower=0).astype(int)
    )
    output = output.sort_values(["_fixture_time", "team_id", "player_id"]).reset_index(drop=True)

    leading = [
        "match_id", "fbref_id", "fpl_id", "gw_orig", "gw_played", "date_sched", "date_played",
        "team_id", "opponent_id", "team", "venue", "was_home", "player_id", "player", "pos",
        "minutes", "days_since_last", "is_active", "is_starter", "starter_source",
    ]
    trailing = [c for c in PANEL_PROVENANCE_COLUMNS if c in output]
    internal = {"_had_provider_observation", "_had_fpl_roster_row", "_fixture_time", "is_home", "status"}
    remainder = [
        c for c in output.columns
        if c not in leading and c not in trailing and c not in internal
    ]
    output = output[[c for c in leading if c in output] + remainder + trailing]
    audit = {
        "season": season,
        "rows": len(output),
        "players": int(output["player_id"].nunique()),
        "matches": int(output["match_id"].nunique()),
        "played_rows": int(output["observation_status"].eq("played").sum()),
        "unused_substitute_rows": int(output["observation_status"].eq("unused_substitute").sum()),
        "not_in_matchday_squad_rows": int(output["observation_status"].eq("not_in_matchday_squad").sum()),
        "pending_rows": int(output["observation_status"].eq("fixture_pending").sum()),
        "provider_only_rows": int(output["row_source"].eq("provider_only_observation").sum()),
        "observed_ids_remapped_by_match_team_name": observed_ids_remapped,
        "observed_ids_remapped_by_constrained_fuzzy_name": observed_ids_fuzzy_remapped,
    }
    return output, audit


def build_effective_dated_eligibility(
    season: str,
    panel: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    fixture_order = fixtures[["match_id", "team_id", "_fixture_time"]].copy()
    fixture_order = fixture_order.sort_values(["team_id", "_fixture_time", "match_id"])
    fixture_order["_team_fixture_index"] = fixture_order.groupby("team_id").cumcount()
    work = panel.merge(
        fixture_order,
        on=["match_id", "team_id"],
        how="left",
        validate="many_to_one",
    )
    if work["_team_fixture_index"].isna().any():
        raise ValueError(f"{season}: eligibility rows lack team fixture ordering")
    work = work.sort_values(["player_id", "team_id", "_team_fixture_index"])
    previous = work.groupby(["player_id", "team_id"])["_team_fixture_index"].shift(1)
    new_segment = previous.isna() | work["_team_fixture_index"].sub(previous).ne(1)
    work["_segment"] = new_segment.groupby([work["player_id"], work["team_id"]]).cumsum()

    rows: list[dict[str, Any]] = []
    for (_, _, _), group in work.groupby(["player_id", "team_id", "_segment"], sort=True):
        group = group.sort_values("_fixture_time")
        first = group.iloc[0]
        last = group.iloc[-1]
        rows.append(
            {
                "season": season,
                "player_id": first["player_id"],
                "player": first.get("player", ""),
                "pos": first.get("pos", ""),
                "team_id": first["team_id"],
                "valid_from": _iso(pd.Timestamp(first["_fixture_time"])),
                "valid_until": _iso(pd.Timestamp(last["_fixture_time"])),
                "registered": True,
                "confirmed_unavailable": bool(first.get("confirmed_unavailable", False)),
                "unavailable_reason": first.get("unavailable_reason", ""),
                "information_timestamp": first.get("information_timestamp", ""),
                "source": first.get("eligibility_source", ""),
                "timestamp_safe": bool(first.get("eligibility_timestamp_safe", False)),
                "reconstruction_method": (
                    "fixture_observed_membership"
                    if first.get("observation_status") != "fixture_pending"
                    else "preseason_roster_cross_fixture"
                ),
                "first_match_id": first["match_id"],
                "last_match_id": last["match_id"],
                "fixtures_observed": len(group),
            }
        )
    return pd.DataFrame(rows, columns=ELIGIBILITY_COLUMNS)


def _append_availability_history(snapshot: pd.DataFrame, history_path: Path) -> None:
    history = _read_csv(history_path) if history_path.is_file() else pd.DataFrame()
    combined = pd.concat([history, snapshot], ignore_index=True)
    combined = combined.drop_duplicates(
        ["season", "player_id", "information_timestamp"], keep="last"
    ).sort_values(["information_timestamp", "team_id", "player_id"])
    _atomic_write_csv(combined, history_path)


def process_season(
    season: str,
    *,
    processed_fpl_root: Path,
    raw_fpl_root: Path,
    fixtures_root: Path,
    registry_root: Path,
    eligibility_root: Path,
    availability_root: Path,
    as_of: pd.Timestamp,
    force: bool,
) -> dict[str, Any]:
    fixture_dir = fixtures_root / season
    fixture_path = fixture_dir / "fixture_calendar.csv"
    calendar_path = fixture_dir / "player_fixture_calendar.csv"
    if not fixture_path.is_file():
        raise FileNotFoundError(fixture_path)
    fixtures = normalize_fixture_calendar(_read_csv(fixture_path), season)

    merged_path = processed_fpl_root / season / "gws" / "merged_gws.csv"
    raw_players_path = raw_fpl_root / season / "players_raw.csv"
    roster_path = processed_fpl_root / season / "season" / "cleaned_players.csv"
    availability: pd.DataFrame | None = None
    if merged_path.is_file():
        code_registry = _global_fpl_code_registry(registry_root, processed_fpl_root)
        universe, source_audit = load_historical_fpl_universe(
            season,
            merged_path=merged_path,
            players_raw_path=raw_players_path,
            code_registry=code_registry,
            registry_root=registry_root,
        )
    else:
        if not roster_path.is_file() or not raw_players_path.is_file():
            raise FileNotFoundError(
                f"{season}: requires merged_gws.csv or a cleaned/raw preseason roster"
            )
        source_timestamp = _file_timestamp(raw_players_path)
        universe, source_audit = load_preseason_fpl_universe(
            season,
            roster_path=roster_path,
            fixtures=fixtures,
            information_timestamp=source_timestamp,
        )
        availability = build_availability_snapshot(
            season,
            raw_players_path=raw_players_path,
            roster_path=roster_path,
            information_timestamp=source_timestamp,
        )
        timestamp_label = source_timestamp.strftime("%Y%m%dT%H%M%SZ")
        snapshot_path = availability_root / season / "snapshots" / f"availability__{timestamp_label}.csv"
        if force or not snapshot_path.exists():
            _atomic_write_csv(availability, snapshot_path)
        _append_availability_history(
            availability,
            availability_root / season / "availability_history.csv",
        )
        schedule_snapshot = fixture_dir / "snapshots" / f"fixture_calendar__{as_of.strftime('%Y%m%dT%H%M%SZ')}.csv"
        if force or not schedule_snapshot.exists():
            _atomic_write_csv(fixtures.drop(columns=["_fixture_key", "_fixture_time"]), schedule_snapshot)

    observed, observed_source = _observed_calendar_source(calendar_path)
    observed = _normalize_observed(observed)
    archive_path = calendar_path.with_name("player_fixture_calendar_observed.csv")
    if observed_source == calendar_path and (force or not archive_path.exists()):
        _atomic_write_csv(observed, archive_path)

    panel, panel_audit = expand_player_fixture_calendar(
        season,
        fixtures=fixtures,
        universe=universe,
        observed=observed,
        availability=availability,
        information_timestamp=source_audit["information_timestamp"],
        eligibility_source=source_audit["source"],
        timestamp_safe=bool(source_audit["timestamp_safe"]),
    )
    eligibility = build_effective_dated_eligibility(season, panel, fixtures)
    eligibility_path = eligibility_root / season / "player_eligibility.csv"
    audit = {
        **source_audit,
        **panel_audit,
        "eligibility_intervals": len(eligibility),
        "observed_calendar_source": str(observed_source),
        "calendar_output": str(calendar_path),
        "eligibility_output": str(eligibility_path),
        "generated_at": _iso(as_of),
    }
    _atomic_write_csv(panel, calendar_path)
    _atomic_write_csv(eligibility, eligibility_path)
    _atomic_write_json(audit, eligibility_root / season / "backfill_audit.json")
    return audit


def _discover_seasons(fixtures_root: Path, requested: Iterable[str]) -> list[str]:
    selected = [value for value in requested if value]
    if selected:
        return selected
    return sorted(path.name for path in fixtures_root.iterdir() if path.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build effective-dated FPL eligibility and expand player calendars "
            "with eligible DNP/pending rows."
        )
    )
    parser.add_argument("--season", action="append", default=[])
    parser.add_argument(
        "--processed-fpl-root",
        type=Path,
        default=Path("data/processed/fpl/ENG-Premier League"),
    )
    parser.add_argument(
        "--raw-fpl-root",
        type=Path,
        default=Path("data/raw/fpl/ENG-Premier League"),
    )
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path("data/processed/registry/fixtures"),
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=Path("data/processed/registry"),
    )
    parser.add_argument(
        "--eligibility-root",
        type=Path,
        default=Path("data/processed/registry/eligibility"),
    )
    parser.add_argument(
        "--availability-root",
        type=Path,
        default=Path("data/processed/registry/availability"),
    )
    parser.add_argument("--as-of", help="UTC information timestamp; defaults to now")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    as_of = _utc_timestamp(args.as_of)
    seasons = _discover_seasons(args.fixtures_root, args.season)
    failures: list[tuple[str, Exception]] = []
    for season in seasons:
        try:
            audit = process_season(
                season,
                processed_fpl_root=args.processed_fpl_root,
                raw_fpl_root=args.raw_fpl_root,
                fixtures_root=args.fixtures_root,
                registry_root=args.registry_root,
                eligibility_root=args.eligibility_root,
                availability_root=args.availability_root,
                as_of=as_of,
                force=args.force,
            )
            logging.info(
                "%s: %d player-fixture rows (%d DNP, %d pending), %d eligibility intervals",
                season,
                audit["rows"],
                audit["not_in_matchday_squad_rows"] + audit["unused_substitute_rows"],
                audit["pending_rows"],
                audit["eligibility_intervals"],
            )
        except Exception as exc:
            logging.exception("%s eligibility/DNP publication failed", season)
            failures.append((season, exc))
    if failures:
        raise RuntimeError(
            "Eligibility/DNP publication failed for: "
            + ", ".join(season for season, _ in failures)
        )


if __name__ == "__main__":
    main()
