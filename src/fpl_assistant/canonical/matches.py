from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from .identity import stable_canonical_id


MATCH_DIMENSION_COLUMNS = [
    "match_id",
    "competition",
    "season",
    "kickoff_utc",
    "gameweek",
    "round",
    "home_team_id",
    "away_team_id",
    "status",
]

MATCH_BRIDGE_COLUMNS = [
    "provider",
    "provider_match_id",
    "match_id",
    "provider_game",
    "match_method",
    "match_confidence",
]


@dataclass(frozen=True)
class MatchRegistryResult:
    matches: pd.DataFrame
    bridges: pd.DataFrame
    unresolved: pd.DataFrame


def _team_bridge_lookup(team_bridges: pd.DataFrame) -> dict[tuple[str, str], str]:
    needed = {"entity_type", "provider", "provider_id", "canonical_id"}
    missing = needed - set(team_bridges.columns)
    if missing:
        raise KeyError(f"Team bridges missing columns: {sorted(missing)}")
    rows = team_bridges[team_bridges["entity_type"].astype(str).eq("team")]
    return {
        (str(row.provider).lower(), str(row.provider_id)): str(row.canonical_id)
        for row in rows.itertuples(index=False)
    }


def build_match_registry(
    schedules: pd.DataFrame,
    *,
    team_bridges: pd.DataFrame,
    existing_matches: pd.DataFrame | None = None,
    existing_bridges: pd.DataFrame | None = None,
    kickoff_tolerance: timedelta = timedelta(hours=36),
    provider_priority: tuple[str, ...] = ("fpl", "fbref", "whoscored", "understat"),
) -> MatchRegistryResult:
    required = {
        "provider",
        "provider_match_id",
        "competition",
        "season",
        "kickoff_utc",
        "home_provider_team_id",
        "away_provider_team_id",
    }
    missing = required - set(schedules.columns)
    if missing:
        raise KeyError(f"Schedule records missing columns: {sorted(missing)}")

    work = schedules.copy()
    work["provider"] = work["provider"].astype(str).str.lower().str.strip()
    work["provider_match_id"] = work["provider_match_id"].astype(str).str.strip()
    work["kickoff_utc"] = pd.to_datetime(work["kickoff_utc"], utc=True, errors="coerce")
    for column in ("provider_game", "gameweek", "round", "status"):
        if column not in work:
            work[column] = pd.NA

    team_lookup = _team_bridge_lookup(team_bridges)
    work["home_team_id"] = [
        team_lookup.get((provider, str(provider_id)))
        for provider, provider_id in zip(
            work["provider"], work["home_provider_team_id"]
        )
    ]
    work["away_team_id"] = [
        team_lookup.get((provider, str(provider_id)))
        for provider, provider_id in zip(
            work["provider"], work["away_provider_team_id"]
        )
    ]
    unresolved_mask = (
        work["kickoff_utc"].isna()
        | work["home_team_id"].isna()
        | work["away_team_id"].isna()
    )
    unresolved = work.loc[unresolved_mask].copy()
    work = work.loc[~unresolved_mask].copy()

    existing_matches_df = (
        existing_matches.copy()
        if existing_matches is not None
        else pd.DataFrame(columns=MATCH_DIMENSION_COLUMNS)
    )
    existing_bridges_df = (
        existing_bridges.copy()
        if existing_bridges is not None
        else pd.DataFrame(columns=MATCH_BRIDGE_COLUMNS)
    )
    bridge_lookup = {
        (str(row.provider).lower(), str(row.provider_match_id)): str(row.match_id)
        for row in existing_bridges_df.itertuples(index=False)
    }
    priority = {provider: rank for rank, provider in enumerate(provider_priority)}
    work["_priority"] = work["provider"].map(priority).fillna(len(priority)).astype(int)

    assignments: dict[int, tuple[str, str, float]] = {}
    if not existing_matches_df.empty:
        existing = existing_matches_df.copy()
        existing["kickoff_utc"] = pd.to_datetime(
            existing["kickoff_utc"], utc=True, errors="coerce"
        )
        for index, row in work.iterrows():
            explicit = bridge_lookup.get((row["provider"], row["provider_match_id"]))
            if explicit:
                assignments[index] = (explicit, "existing_bridge", 1.0)
                continue
            candidates = existing[
                existing["competition"].astype(str).eq(str(row["competition"]))
                & existing["season"].astype(str).eq(str(row["season"]))
                & existing["home_team_id"].astype(str).eq(str(row["home_team_id"]))
                & existing["away_team_id"].astype(str).eq(str(row["away_team_id"]))
            ].copy()
            if not candidates.empty:
                delta = (candidates["kickoff_utc"] - row["kickoff_utc"]).abs()
                candidates = candidates[delta <= kickoff_tolerance]
            if len(candidates) == 1:
                assignments[index] = (
                    str(candidates.iloc[0]["match_id"]),
                    "existing_match_window",
                    0.99,
                )

    clusters: list[list[int]] = []
    group_cols = ["competition", "season", "home_team_id", "away_team_id"]
    for _, group in work.sort_values("kickoff_utc").groupby(group_cols, dropna=False):
        current: list[int] = []
        anchor = None
        for index, row in group.iterrows():
            if index in assignments:
                continue
            kickoff = row["kickoff_utc"]
            if anchor is None or kickoff - anchor <= kickoff_tolerance:
                current.append(index)
                anchor = kickoff if anchor is None else anchor
            else:
                clusters.append(current)
                current = [index]
                anchor = kickoff
        if current:
            clusters.append(current)

    for cluster in clusters:
        rows = work.loc[cluster].sort_values(["_priority", "kickoff_utc"])
        best = rows.iloc[0]
        match_id = stable_canonical_id(
            "match",
            best["competition"],
            best["season"],
            best["home_team_id"],
            best["away_team_id"],
            best["kickoff_utc"].date().isoformat(),
            length=16,
        )
        for index in cluster:
            assignments[index] = (match_id, "teams_kickoff_window", 0.95)

    work["match_id"] = [assignments[index][0] for index in work.index]
    work["match_method"] = [assignments[index][1] for index in work.index]
    work["match_confidence"] = [assignments[index][2] for index in work.index]

    dimensions: list[dict] = []
    for match_id, group in work.groupby("match_id"):
        best = group.sort_values(["_priority", "kickoff_utc"]).iloc[0]
        dimensions.append(
            {
                "match_id": match_id,
                "competition": best["competition"],
                "season": best["season"],
                "kickoff_utc": best["kickoff_utc"],
                "gameweek": best.get("gameweek"),
                "round": best.get("round"),
                "home_team_id": best["home_team_id"],
                "away_team_id": best["away_team_id"],
                "status": best.get("status"),
            }
        )

    new_matches = pd.DataFrame(dimensions, columns=MATCH_DIMENSION_COLUMNS)
    matches = pd.concat(
        [existing_matches_df.reindex(columns=MATCH_DIMENSION_COLUMNS), new_matches],
        ignore_index=True,
    ).drop_duplicates("match_id", keep="last")

    new_bridges = work.rename(columns={"provider_game": "provider_game"})[
        [
            "provider",
            "provider_match_id",
            "match_id",
            "provider_game",
            "match_method",
            "match_confidence",
        ]
    ]
    combined_bridges = pd.concat(
        [
            existing_bridges_df.reindex(columns=MATCH_BRIDGE_COLUMNS),
            new_bridges,
        ],
        ignore_index=True,
    )

    conflicts = (
        combined_bridges.groupby(["provider", "provider_match_id"])["match_id"].nunique()
    )
    if (conflicts > 1).any():
        raise ValueError("A provider match ID maps to multiple canonical matches.")
    bridges = combined_bridges.drop_duplicates(
        ["provider", "provider_match_id"], keep="last"
    )

    return MatchRegistryResult(
        matches=matches.sort_values(["season", "kickoff_utc"]).reset_index(drop=True),
        bridges=bridges.sort_values(["provider", "provider_match_id"]).reset_index(
            drop=True
        ),
        unresolved=unresolved.reset_index(drop=True),
    )
