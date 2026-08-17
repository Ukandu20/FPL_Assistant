from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


COMMON_FIELDS = {
    "match_id", "player_id", "team_id", "opponent_id", "season", "gameweek",
    "kickoff_utc", "fpl_position", "venue", "provider_record_id", "retrieved_at",
}

PROVIDER_FIELDS: dict[str, set[str]] = {
    "fpl": {
        "minutes", "started", "named_on_bench", "availability_status",
        "availability_reason", "fpl_points", "price", "goals", "assists",
        "clean_sheet", "goals_conceded", "saves", "yellow_cards", "red_cards",
        "second_yellow_cards", "own_goals", "penalties_saved", "penalties_faced",
    },
    "understat": {
        "minutes", "npxg", "non_penalty_goals", "xa", "xg", "goals",
        "assists", "key_passes",
    },
    "whoscored": {
        "minutes", "started", "named_on_bench", "shots_on_target", "shots_in_box",
        "key_passes", "big_chances_created", "shot_creating_actions", "tackles_won",
        "interceptions", "clearances", "blocks", "recoveries", "saves",
        "shots_on_target_faced", "goals_conceded", "post_shot_xg", "penalties_saved",
        "penalties_faced", "yellow_cards", "red_cards", "availability_status",
        "availability_reason",
    },
}


@dataclass(frozen=True)
class AdaptedPlayerMatches:
    records: pd.DataFrame
    provenance: pd.DataFrame


def adapt_player_matches(
    frame: pd.DataFrame,
    *,
    provider: str,
    field_map: Mapping[str, str],
) -> AdaptedPlayerMatches:
    """Map one provider without borrowing definitions from another provider."""
    normalized_provider = provider.lower().strip()
    if normalized_provider not in PROVIDER_FIELDS:
        raise ValueError(f"Unsupported archetype provider: {provider}")
    missing_sources = set(field_map) - set(frame)
    if missing_sources:
        raise KeyError(f"{provider} input missing mapped fields: {sorted(missing_sources)}")
    targets = list(field_map.values())
    if len(targets) != len(set(targets)):
        raise ValueError("Provider field map contains duplicate canonical targets")
    unsupported = set(targets) - COMMON_FIELDS - PROVIDER_FIELDS[normalized_provider]
    if unsupported:
        raise ValueError(
            f"{provider} cannot define canonical fields owned by another provider: {sorted(unsupported)}"
        )
    required_targets = {"match_id", "player_id"}
    if not required_targets.issubset(targets):
        raise ValueError("Provider field map must define match_id and player_id")

    records = frame[list(field_map)].rename(columns=dict(field_map)).copy()
    records["provider"] = normalized_provider
    provenance_rows: list[dict[str, object]] = []
    for source, target in field_map.items():
        if target in COMMON_FIELDS:
            continue
        non_null = records[target].notna()
        for index in records.index[non_null]:
            provenance_rows.append(
                {
                    "match_id": records.at[index, "match_id"],
                    "player_id": records.at[index, "player_id"],
                    "metric": target,
                    "provider": normalized_provider,
                    "source_field": source,
                    "value": records.at[index, target],
                }
            )
    return AdaptedPlayerMatches(records=records, provenance=pd.DataFrame(provenance_rows))


def validate_provider_ownership(
    records: pd.DataFrame,
    provider_fields: Mapping[str, str],
) -> list[str]:
    violations: list[str] = []
    if "provider" not in records:
        return ["missing provider column"]
    for field, owner in provider_fields.items():
        if field not in records:
            continue
        wrong = records[field].notna() & records["provider"].astype(str).str.lower().ne(owner)
        if wrong.any():
            violations.append(f"{field} must come from {owner}; found {int(wrong.sum())} mismatches")
    return violations


def understat_team_rows_to_matches(rows: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical Understat team-match rows into one match row."""
    required = {"game_id", "game_date", "season", "team_id", "venue", "goals", "xg"}
    missing = required - set(rows)
    if missing:
        raise KeyError(f"Understat team rows missing: {sorted(missing)}")
    output: list[dict[str, object]] = []
    for game_id, group in rows.groupby("game_id", sort=False):
        venue = group["venue"].astype(str).str.upper()
        home = group.loc[venue.eq("H")]
        away = group.loc[venue.eq("A")]
        if len(home) != 1 or len(away) != 1:
            raise ValueError(f"Understat game {game_id} must have exactly one home and away row")
        home_row = home.iloc[0]
        away_row = away.iloc[0]
        kickoff = pd.to_datetime(
            f"{home_row['game_date']} {home_row.get('game_time', '00:00:00')}",
            utc=True,
            errors="raise",
        )
        output.append(
            {
                "match_id": str(game_id), "kickoff_utc": kickoff,
                "season": str(home_row["season"]),
                "gameweek": home_row.get("round", pd.NA),
                "home_team_id": str(home_row["team_id"]),
                "away_team_id": str(away_row["team_id"]),
                "home_goals": float(home_row["goals"]),
                "away_goals": float(away_row["goals"]),
                "home_xg": float(home_row["xg"]),
                "away_xg": float(away_row["xg"]),
                "provider": "understat",
            }
        )
    return pd.DataFrame(output).sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)


__all__ = [
    "AdaptedPlayerMatches", "COMMON_FIELDS", "PROVIDER_FIELDS",
    "adapt_player_matches", "understat_team_rows_to_matches", "validate_provider_ownership",
]
