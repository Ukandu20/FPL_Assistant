from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

from .staging import _bridge_map


DEFENSIVE_EVENT_COLUMNS = [
    "match_id",
    "player_id",
    "team_id",
    "tackles",
    "tackles_won",
    "interceptions",
    "clearances",
    "blocks",
    "recoveries",
    "saves",
    "aerial_duels",
    "aerial_duels_won",
    "defensive_contributions_def",
    "defensive_contributions_outfield",
    "source",
]


@dataclass(frozen=True)
class DefensiveAggregationResult:
    player_match: pd.DataFrame
    event_audit: pd.DataFrame


def canonicalize_whoscored_event_ids(
    events: pd.DataFrame,
    *,
    player_bridges: pd.DataFrame,
    team_bridges: pd.DataFrame,
    match_bridges: pd.DataFrame,
    strict: bool = True,
) -> pd.DataFrame:
    """Resolve native WhoScored event IDs before defensive aggregation."""

    work = events.copy()
    source_match = "match_id" if "match_id" in work else "game_id"
    required = {source_match, "player_id", "team_id"}
    missing = required - set(work.columns)
    if missing:
        raise KeyError(f"WhoScored events missing ID columns: {sorted(missing)}")
    player_map = _bridge_map(player_bridges, provider="whoscored")
    team_map = _bridge_map(team_bridges, provider="whoscored")
    match_map = _bridge_map(
        match_bridges,
        provider="whoscored",
        provider_id_field="provider_match_id",
    )
    work["match_id"] = work[source_match].astype(str).map(match_map)
    work["player_id"] = work["player_id"].astype("string").map(player_map)
    work["team_id"] = work["team_id"].astype("string").map(team_map)
    if "related_player_id" in work:
        work["related_player_id"] = (
            work["related_player_id"].astype("string").map(player_map)
        )
    unresolved = work[work[["match_id", "player_id", "team_id"]].isna().any(axis=1)]
    if strict and not unresolved.empty:
        raise ValueError(
            f"{len(unresolved)} WhoScored events have unresolved canonical IDs."
        )
    return work.dropna(subset=["match_id", "player_id", "team_id"]).reset_index(
        drop=True
    )


def _norm(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("displayName") or value.get("name") or value.get("value")
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _qualifier_names(value: Any) -> set[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return {_norm(value)}
    if not isinstance(value, Iterable) or isinstance(value, (bytes, Mapping)):
        return set()
    names: set[str] = set()
    for qualifier in value:
        if isinstance(qualifier, Mapping):
            names.add(_norm(qualifier.get("type")))
            names.add(_norm(qualifier.get("value")))
        else:
            names.add(_norm(qualifier))
    return {name for name in names if name}


def aggregate_whoscored_defensive_events(
    events: pd.DataFrame,
    *,
    match_id_column: str = "match_id",
) -> DefensiveAggregationResult:
    """Aggregate WhoScored event rows into canonical player-match actions.

    The output intentionally keeps defender and MID/FWD contribution formulas
    separate. Provider-derived totals must later be validated against official
    FPL outcomes before becoming labels.
    """

    required = {match_id_column, "player_id", "team_id", "type"}
    missing = required - set(events.columns)
    if missing:
        raise KeyError(f"WhoScored events missing columns: {sorted(missing)}")
    work = events.copy()
    for column in ("outcome_type", "qualifiers", "related_player_id"):
        if column not in work:
            work[column] = pd.NA
    work["_type"] = work["type"].map(_norm)
    work["_outcome"] = work["outcome_type"].map(_norm)
    work["_qualifiers"] = work["qualifiers"].map(_qualifier_names)
    player_team_lookup = {
        (row[match_id_column], row["player_id"]): row["team_id"]
        for _, row in work.dropna(subset=["player_id", "team_id"]).iterrows()
    }
    match_teams = (
        work.dropna(subset=["team_id"])
        .groupby(match_id_column)["team_id"]
        .agg(lambda values: tuple(pd.unique(values)))
        .to_dict()
    )

    def related_team(row: pd.Series, related: Any) -> Any:
        mapped = player_team_lookup.get((row[match_id_column], related))
        if mapped is not None:
            return mapped
        teams = match_teams.get(row[match_id_column], ())
        alternatives = [team for team in teams if team != row["team_id"]]
        return alternatives[0] if len(alternatives) == 1 else pd.NA

    counters: dict[tuple[Any, Any, Any], dict[str, int]] = {}

    def increment(
        row: pd.Series,
        metric: str,
        *,
        player_id: Any | None = None,
        team_id: Any | None = None,
        amount: int = 1,
    ) -> None:
        pid = row["player_id"] if player_id is None else player_id
        if pd.isna(pid):
            return
        tid = row["team_id"] if team_id is None else team_id
        key = (row[match_id_column], pid, tid)
        counters.setdefault(
            key,
            {
                "tackles": 0,
                "tackles_won": 0,
                "interceptions": 0,
                "clearances": 0,
                "blocks": 0,
                "recoveries": 0,
                "saves": 0,
                "aerial_duels": 0,
                "aerial_duels_won": 0,
            },
        )
        counters[key][metric] += amount

    recognized: list[bool] = []
    for _, row in work.iterrows():
        event_type = row["_type"]
        outcome = row["_outcome"]
        qualifiers = row["_qualifiers"]
        handled = False
        if event_type in {"tackle", "tackles"}:
            increment(row, "tackles")
            if outcome in {"successful", "success"}:
                increment(row, "tackles_won")
            handled = True
        elif event_type in {"interception", "interceptions"}:
            increment(row, "interceptions")
            handled = True
        elif event_type in {"clearance", "clearances"}:
            increment(row, "clearances")
            handled = True
        elif event_type in {"blockedpass", "block", "shotblock"}:
            increment(row, "blocks")
            handled = True
        elif event_type in {"ballrecovery", "recovery"}:
            increment(row, "recoveries")
            handled = True
        elif event_type in {"save", "keepersave"}:
            increment(row, "saves")
            handled = True
        elif event_type == "savedshot" and pd.notna(row["related_player_id"]):
            related = row["related_player_id"]
            related_team_id = related_team(row, related)
            if {"blocked", "blockedshot"} & qualifiers:
                increment(
                    row,
                    "blocks",
                    player_id=related,
                    team_id=related_team_id,
                )
            else:
                increment(
                    row,
                    "saves",
                    player_id=related,
                    team_id=related_team_id,
                )
            handled = True
        elif event_type in {"aerial", "aerialduel"}:
            increment(row, "aerial_duels")
            if outcome in {"successful", "success"}:
                increment(row, "aerial_duels_won")
            handled = True

        # WhoScored commonly represents a blocked shot on the shooting event,
        # with the blocker referenced separately.
        if (
            event_type != "savedshot"
            and
            {"blocked", "blockedshot"} & qualifiers
            and pd.notna(row["related_player_id"])
        ):
            related = row["related_player_id"]
            increment(
                row,
                "blocks",
                player_id=related,
                team_id=related_team(row, related),
            )
            handled = True
        recognized.append(handled)

    rows: list[dict] = []
    for (match_id, player_id, team_id), values in counters.items():
        defender_total = (
            values["tackles"]
            + values["interceptions"]
            + values["clearances"]
            + values["blocks"]
        )
        outfield_total = defender_total + values["recoveries"]
        rows.append(
            {
                "match_id": match_id,
                "player_id": player_id,
                "team_id": team_id,
                **values,
                "defensive_contributions_def": defender_total,
                "defensive_contributions_outfield": outfield_total,
                "source": "whoscored_events",
            }
        )

    audit = (
        work.assign(recognized=recognized)
        .groupby(["_type", "recognized"], dropna=False)
        .size()
        .rename("event_count")
        .reset_index()
        .rename(columns={"_type": "event_type"})
        .sort_values(["recognized", "event_count"], ascending=[True, False])
    )
    return DefensiveAggregationResult(
        player_match=pd.DataFrame(rows, columns=DEFENSIVE_EVENT_COLUMNS),
        event_audit=audit.reset_index(drop=True),
    )
