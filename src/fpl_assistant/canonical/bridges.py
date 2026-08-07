from __future__ import annotations

from pathlib import Path

import pandas as pd


MATCH_BRIDGE_COLUMNS = [
    "provider",
    "provider_match_id",
    "match_id",
    "provider_game",
    "match_method",
    "match_confidence",
]


def upsert_match_bridges(path: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """Atomically upsert authoritative provider -> FBref match mappings."""
    incoming = rows.reindex(columns=MATCH_BRIDGE_COLUMNS).copy()
    incoming["provider"] = incoming["provider"].astype("string").str.lower().str.strip()
    incoming["provider_match_id"] = incoming["provider_match_id"].astype("string").str.strip()
    incoming["match_id"] = incoming["match_id"].astype("string").str.strip()
    incoming = incoming.dropna(subset=["provider", "provider_match_id", "match_id"])
    conflicts = incoming.groupby(["provider", "provider_match_id"])["match_id"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Incoming provider match IDs map to multiple FBref match IDs")
    incoming = incoming.drop_duplicates(["provider", "provider_match_id"], keep="last")

    existing = (
        pd.read_csv(path, dtype="string").reindex(columns=MATCH_BRIDGE_COLUMNS)
        if path.is_file()
        else pd.DataFrame(columns=MATCH_BRIDGE_COLUMNS)
    )
    if not existing.empty and not incoming.empty:
        keys = pd.MultiIndex.from_frame(incoming[["provider", "provider_match_id"]])
        old_keys = pd.MultiIndex.from_frame(existing[["provider", "provider_match_id"]])
        existing = existing.loc[~old_keys.isin(keys)]
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.sort_values(["provider", "provider_match_id"]).reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)
    return combined
