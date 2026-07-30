from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Mapping

import pandas as pd


def _bridge_map(
    bridges: pd.DataFrame,
    *,
    provider: str,
    provider_id_field: str = "provider_id",
) -> dict[str, str]:
    canonical_field = (
        "match_id" if provider_id_field == "provider_match_id" else "canonical_id"
    )
    required = {"provider", provider_id_field, canonical_field}
    missing = required - set(bridges.columns)
    if missing:
        raise KeyError(f"Identity bridge missing columns: {sorted(missing)}")
    subset = bridges[bridges["provider"].astype(str).str.lower().eq(provider.lower())]
    if subset[provider_id_field].astype(str).duplicated().any():
        raise ValueError(f"Duplicate {provider} IDs in identity bridge.")
    return dict(
        zip(
            subset[provider_id_field].astype(str),
            subset[canonical_field].astype(str),
        )
    )


def _record_id(provider: str, values: tuple[object, ...]) -> str:
    payload = "|".join([provider, *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def stage_match_facts(
    frame: pd.DataFrame,
    *,
    provider: str,
    entity_type: str,
    entity_bridges: pd.DataFrame,
    match_bridges: pd.DataFrame,
    provider_entity_id_column: str,
    provider_match_id_column: str,
    metric_columns: Mapping[str, str],
    retrieved_at: str | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Map a provider table into canonical long-lived match fact inputs.

    ``metric_columns`` maps provider column names to canonical metric names.
    The result is suitable for :func:`build_canonical_facts`; source columns
    remain available only in raw/staging storage and do not leak downstream.
    """

    if entity_type not in {"player", "team"}:
        raise ValueError("entity_type must be player or team")
    required = {
        provider_entity_id_column,
        provider_match_id_column,
        *metric_columns.keys(),
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{provider} fact input missing columns: {sorted(missing)}")

    entity_map = _bridge_map(entity_bridges, provider=provider)
    match_map = _bridge_map(
        match_bridges,
        provider=provider,
        provider_id_field="provider_match_id",
    )
    work = frame.copy()
    entity_column = f"{entity_type}_id"
    work[entity_column] = (
        work[provider_entity_id_column].astype(str).map(entity_map)
    )
    work["match_id"] = work[provider_match_id_column].astype(str).map(match_map)
    unresolved = work[work[[entity_column, "match_id"]].isna().any(axis=1)]
    if strict and not unresolved.empty:
        sample = unresolved[
            [provider_entity_id_column, provider_match_id_column]
        ].head(10)
        raise ValueError(
            f"{len(unresolved)} {provider} rows have unresolved canonical IDs: "
            f"{sample.to_dict(orient='records')}"
        )
    work = work.dropna(subset=[entity_column, "match_id"]).copy()
    duplicate = work.duplicated(
        [provider_match_id_column, provider_entity_id_column], keep=False
    )
    if duplicate.any():
        raise ValueError(
            f"{provider} contains duplicate {entity_type}-match rows: "
            f"{work.loc[duplicate, [provider_match_id_column, provider_entity_id_column]].head(10).to_dict(orient='records')}"
        )
    work = work.rename(columns=dict(metric_columns))
    timestamp = retrieved_at or datetime.now(timezone.utc).isoformat()
    work["provider"] = provider.lower()
    work["retrieved_at"] = timestamp
    work["provider_record_id"] = [
        _record_id(
            provider.lower(),
            (
                row[provider_match_id_column],
                row[provider_entity_id_column],
            ),
        )
        for _, row in work.iterrows()
    ]
    columns = [
        "provider",
        "provider_record_id",
        "retrieved_at",
        "match_id",
        entity_column,
        *metric_columns.values(),
    ]
    return work.loc[:, list(dict.fromkeys(columns))].reset_index(drop=True)
