from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .contracts import TableContract, validate_contract


@dataclass(frozen=True)
class FeatureSnapshot:
    frame: pd.DataFrame
    feature_version: str
    as_of_timestamp: pd.Timestamp
    contract_name: str
    contract_version: str


def build_feature_snapshot(
    frame: pd.DataFrame,
    *,
    contract: TableContract,
    feature_version: str,
    as_of_timestamp: str | pd.Timestamp,
    known_at_columns: Sequence[str] = ("known_at", "retrieved_at"),
) -> FeatureSnapshot:
    as_of = pd.Timestamp(as_of_timestamp)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")
    work = frame.copy()
    for column in known_at_columns:
        if column not in work:
            continue
        known = pd.to_datetime(work[column], utc=True, errors="coerce")
        leaked = known.notna() & (known > as_of)
        if leaked.any():
            raise ValueError(
                f"Feature snapshot contains {int(leaked.sum())} rows where "
                f"{column} is later than as_of_timestamp."
            )
    work["as_of_timestamp"] = as_of
    work["feature_version"] = feature_version
    validate_contract(work, contract)
    return FeatureSnapshot(
        frame=work,
        feature_version=feature_version,
        as_of_timestamp=as_of,
        contract_name=contract.name,
        contract_version=contract.version,
    )
