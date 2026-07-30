from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_PLAYER_SOURCE_POLICY: dict[str, tuple[str, ...]] = {
    "started": ("fpl", "fbref", "whoscored"),
    "named_on_bench": ("fbref", "whoscored", "fpl"),
    "minutes": ("fpl", "fbref", "whoscored", "understat"),
    "goals": ("fpl", "understat", "fbref", "whoscored"),
    "assists": ("fpl", "understat", "fbref", "whoscored"),
    "xg": ("understat", "fbref", "whoscored"),
    "xa": ("understat", "fbref", "whoscored"),
    "shots": ("understat", "fbref", "whoscored"),
    "shots_on_target": ("fbref", "whoscored", "understat"),
    "key_passes": ("understat", "whoscored", "fbref"),
    "tackles": ("whoscored",),
    "tackles_won": ("whoscored",),
    "interceptions": ("whoscored",),
    "clearances": ("whoscored",),
    "blocks": ("whoscored",),
    "recoveries": ("whoscored",),
    "saves": ("fpl", "fbref", "whoscored"),
    "shots_on_target_faced": ("fbref", "whoscored"),
    "yellow_cards": ("fpl", "fbref", "whoscored", "understat"),
    "red_cards": ("fpl", "fbref", "whoscored", "understat"),
    "own_goals": ("fpl", "fbref", "understat", "whoscored"),
}


DEFAULT_TEAM_SOURCE_POLICY: dict[str, tuple[str, ...]] = {
    "goals": ("fpl", "understat", "fbref", "whoscored"),
    "goals_against": ("fpl", "understat", "fbref", "whoscored"),
    "xg": ("understat", "fbref", "whoscored"),
    "xga": ("understat", "fbref", "whoscored"),
    "shots": ("fbref", "whoscored", "understat"),
    "shots_against": ("fbref", "whoscored"),
    "shots_on_target": ("fbref", "whoscored"),
    "shots_on_target_against": ("fbref", "whoscored"),
    "ppda": ("understat",),
    "ppda_allowed": ("understat",),
    "deep_completions": ("understat",),
    "deep_completions_allowed": ("understat",),
    "elo_pre_match": ("clubelo",),
    "elo_post_match": ("clubelo",),
}


@dataclass(frozen=True)
class FactBuildResult:
    facts: pd.DataFrame
    provenance: pd.DataFrame
    conflicts: pd.DataFrame


def _values_conflict(values: Sequence[object], tolerance: float) -> bool:
    non_null = [value for value in values if pd.notna(value)]
    if len(non_null) < 2:
        return False
    numeric = pd.to_numeric(pd.Series(non_null), errors="coerce")
    if numeric.notna().all():
        return float(numeric.max() - numeric.min()) > tolerance
    return len({str(value) for value in non_null}) > 1


def build_canonical_facts(
    records: pd.DataFrame,
    *,
    entity_column: str,
    source_policy: Mapping[str, Sequence[str]],
    key_columns: Sequence[str] | None = None,
    numeric_tolerance: float = 1e-9,
) -> FactBuildResult:
    """Choose one deterministic source per metric and retain field provenance."""

    keys = list(key_columns or ["match_id", entity_column])
    required = {"provider", *keys}
    missing = required - set(records.columns)
    if missing:
        raise KeyError(f"Fact records missing required columns: {sorted(missing)}")

    work = records.copy()
    work["provider"] = work["provider"].astype(str).str.lower().str.strip()
    if "retrieved_at" not in work:
        work["retrieved_at"] = pd.NaT
    work["retrieved_at"] = pd.to_datetime(
        work["retrieved_at"], utc=True, errors="coerce"
    )
    if "provider_record_id" not in work:
        work["provider_record_id"] = pd.NA

    fact_rows: list[dict] = []
    provenance_rows: list[dict] = []
    conflict_rows: list[dict] = []

    for key_values, group in work.groupby(keys, dropna=False, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        base = dict(zip(keys, key_values))
        fact = dict(base)
        for metric, providers in source_policy.items():
            if metric not in group:
                fact[metric] = pd.NA
                continue
            candidates = group[group[metric].notna()].copy()
            if candidates.empty:
                fact[metric] = pd.NA
                continue
            ranks = {provider: rank for rank, provider in enumerate(providers)}
            candidates["_source_rank"] = (
                candidates["provider"].map(ranks).fillna(len(ranks) + 100).astype(int)
            )
            candidates = candidates.sort_values(
                ["_source_rank", "retrieved_at", "provider_record_id"],
                ascending=[True, False, True],
                na_position="last",
                kind="stable",
            )
            chosen = candidates.iloc[0]
            fact[metric] = chosen[metric]
            has_conflict = _values_conflict(
                candidates[metric].tolist(), numeric_tolerance
            )
            provenance = {
                **base,
                "metric": metric,
                "value": chosen[metric],
                "provider": chosen["provider"],
                "provider_record_id": chosen["provider_record_id"],
                "retrieved_at": chosen["retrieved_at"],
                "source_rank": int(chosen["_source_rank"]),
                "candidate_count": int(len(candidates)),
                "conflict": bool(has_conflict),
            }
            provenance_rows.append(provenance)
            if has_conflict:
                conflict_rows.append(
                    {
                        **base,
                        "metric": metric,
                        "chosen_provider": chosen["provider"],
                        "chosen_value": chosen[metric],
                        "candidate_values": [
                            {
                                "provider": row["provider"],
                                "value": row[metric],
                            }
                            for _, row in candidates.iterrows()
                        ],
                    }
                )
        fact_rows.append(fact)

    return FactBuildResult(
        facts=pd.DataFrame(fact_rows),
        provenance=pd.DataFrame(provenance_rows),
        conflicts=pd.DataFrame(conflict_rows),
    )
