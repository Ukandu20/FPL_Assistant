from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Mapping

import pandas as pd


DIMENSION_COLUMNS = [
    "canonical_id",
    "entity_type",
    "canonical_name",
    "birth_date",
    "valid_from",
    "valid_to",
]

BRIDGE_COLUMNS = [
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


def normalize_identity_text(value: object) -> str:
    if value is None:
        value = ""
    else:
        try:
            if bool(pd.isna(value)):
                value = ""
        except (TypeError, ValueError):
            pass
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def stable_canonical_id(namespace: str, *parts: object, length: int = 12) -> str:
    normalized = "|".join(
        normalize_identity_text(part) for part in (namespace, *parts)
    )
    if not normalized.replace("|", ""):
        raise ValueError("Cannot generate a canonical ID from empty identity parts.")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class RegistryBuildResult:
    dimensions: pd.DataFrame
    bridges: pd.DataFrame
    review: pd.DataFrame


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _validate_provider_keys(bridges: pd.DataFrame) -> None:
    if bridges.empty:
        return
    grouped = (
        bridges.groupby(["entity_type", "provider", "provider_id"], dropna=False)[
            "canonical_id"
        ]
        .nunique()
    )
    conflicts = grouped[grouped > 1]
    if not conflicts.empty:
        raise ValueError(
            "A provider identity maps to multiple canonical IDs: "
            f"{list(conflicts.index[:10])}"
        )


def build_identity_registry(
    records: pd.DataFrame,
    *,
    entity_type: str,
    existing_dimensions: pd.DataFrame | None = None,
    existing_bridges: pd.DataFrame | None = None,
    aliases: Mapping[str, str] | None = None,
    generate_missing: bool = True,
) -> RegistryBuildResult:
    """Resolve provider identities into persistent canonical entities.

    Required record columns are ``provider``, ``provider_id`` and
    ``provider_name``. Optional explicit ``canonical_id`` values always win.
    Existing provider bridges are then applied, followed by aliases and unique
    normalized-name matches. Remaining identities receive deterministic IDs
    only when ``generate_missing`` is enabled.
    """

    required = {"provider", "provider_id", "provider_name"}
    missing = required - set(records.columns)
    if missing:
        raise KeyError(f"Identity records missing required columns: {sorted(missing)}")
    if entity_type not in {"player", "team"}:
        raise ValueError("entity_type must be 'player' or 'team'.")

    work = records.copy()
    work["entity_type"] = entity_type
    work["provider"] = work["provider"].astype(str).str.strip().str.lower()
    work["provider_id"] = work["provider_id"].astype(str).str.strip()
    work["provider_name"] = work["provider_name"].astype(str).str.strip()
    work["_name_key"] = work["provider_name"].map(normalize_identity_text)
    for column in ("canonical_id", "canonical_name", "birth_date", "valid_from", "valid_to"):
        if column not in work:
            work[column] = pd.NA

    dimensions = (
        existing_dimensions.copy()
        if existing_dimensions is not None
        else _empty(DIMENSION_COLUMNS)
    )
    bridges = (
        existing_bridges.copy()
        if existing_bridges is not None
        else _empty(BRIDGE_COLUMNS)
    )
    for column in DIMENSION_COLUMNS:
        if column not in dimensions:
            dimensions[column] = pd.NA
    for column in BRIDGE_COLUMNS:
        if column not in bridges:
            bridges[column] = pd.NA
    _validate_provider_keys(bridges)

    bridge_lookup = {
        (str(row.entity_type), str(row.provider), str(row.provider_id)): str(
            row.canonical_id
        )
        for row in bridges.itertuples(index=False)
        if pd.notna(row.canonical_id)
    }
    dimension_names: dict[str, list[str]] = {}
    for row in dimensions.itertuples(index=False):
        if str(row.entity_type) != entity_type or pd.isna(row.canonical_id):
            continue
        key = normalize_identity_text(row.canonical_name)
        dimension_names.setdefault(key, []).append(str(row.canonical_id))

    aliases_normalized = {
        normalize_identity_text(key): str(value)
        for key, value in (aliases or {}).items()
    }

    resolved_rows: list[dict] = []
    review_rows: list[dict] = []
    dimension_additions: dict[str, dict] = {}
    seen_provider_keys: dict[tuple[str, str, str], str] = {}

    for row in work.to_dict("records"):
        provider_key = (entity_type, row["provider"], row["provider_id"])
        canonical_id = (
            str(row["canonical_id"]).strip()
            if pd.notna(row["canonical_id"]) and str(row["canonical_id"]).strip()
            else None
        )
        method = "explicit"
        confidence = 1.0
        review_status = "approved"

        if canonical_id is None and provider_key in bridge_lookup:
            canonical_id = bridge_lookup[provider_key]
            method = "existing_bridge"
        if canonical_id is None and row["_name_key"] in aliases_normalized:
            canonical_id = aliases_normalized[row["_name_key"]]
            method = "alias"
            confidence = 0.99
        if canonical_id is None:
            candidates = dimension_names.get(row["_name_key"], [])
            if len(candidates) == 1:
                canonical_id = candidates[0]
                method = "unique_normalized_name"
                confidence = 0.9
                review_status = "reviewed"
            elif len(candidates) > 1:
                review_rows.append(
                    {
                        **row,
                        "reason": "ambiguous_normalized_name",
                        "candidate_canonical_ids": candidates,
                    }
                )
                continue
        if canonical_id is None and generate_missing:
            canonical_id = stable_canonical_id(
                entity_type,
                row["_name_key"],
                row.get("birth_date", ""),
            )
            method = "generated"
            confidence = 0.75
            review_status = "needs_review"
        if canonical_id is None:
            review_rows.append(
                {**row, "reason": "unresolved", "candidate_canonical_ids": []}
            )
            continue

        existing_target = seen_provider_keys.get(provider_key)
        if existing_target is not None and existing_target != canonical_id:
            raise ValueError(
                f"Conflicting canonical IDs for provider identity {provider_key}: "
                f"{existing_target!r} and {canonical_id!r}"
            )
        seen_provider_keys[provider_key] = canonical_id

        canonical_name = (
            str(row["canonical_name"]).strip()
            if pd.notna(row["canonical_name"]) and str(row["canonical_name"]).strip()
            else row["provider_name"]
        )
        dimension_additions.setdefault(
            canonical_id,
            {
                "canonical_id": canonical_id,
                "entity_type": entity_type,
                "canonical_name": canonical_name,
                "birth_date": row.get("birth_date"),
                "valid_from": row.get("valid_from"),
                "valid_to": row.get("valid_to"),
            },
        )
        resolved_rows.append(
            {
                "entity_type": entity_type,
                "provider": row["provider"],
                "provider_id": row["provider_id"],
                "provider_name": row["provider_name"],
                "canonical_id": canonical_id,
                "valid_from": row.get("valid_from"),
                "valid_to": row.get("valid_to"),
                "match_method": method,
                "match_confidence": confidence,
                "review_status": review_status,
            }
        )

    additions_df = pd.DataFrame(dimension_additions.values(), columns=DIMENSION_COLUMNS)
    dimensions = pd.concat([dimensions[DIMENSION_COLUMNS], additions_df], ignore_index=True)
    dimensions = dimensions.drop_duplicates("canonical_id", keep="first")

    resolved_df = pd.DataFrame(resolved_rows, columns=BRIDGE_COLUMNS)
    bridges = pd.concat([bridges[BRIDGE_COLUMNS], resolved_df], ignore_index=True)
    bridges = bridges.drop_duplicates(
        ["entity_type", "provider", "provider_id"], keep="last"
    )
    _validate_provider_keys(bridges)

    return RegistryBuildResult(
        dimensions=dimensions.sort_values("canonical_id").reset_index(drop=True),
        bridges=bridges.sort_values(
            ["entity_type", "provider", "provider_id"]
        ).reset_index(drop=True),
        review=pd.DataFrame(review_rows),
    )
