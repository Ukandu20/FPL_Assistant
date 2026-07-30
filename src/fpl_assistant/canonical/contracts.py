from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd
from pandas.api import types as pd_types


@dataclass(frozen=True)
class ColumnContract:
    dtype_family: str | None = None
    nullable: bool = True
    minimum: float | None = None
    maximum: float | None = None
    allowed: tuple[Any, ...] = ()


@dataclass(frozen=True)
class TableContract:
    name: str
    version: str
    key: tuple[str, ...]
    columns: Mapping[str, ColumnContract]
    allow_extra_columns: bool = True


@dataclass(frozen=True)
class ContractViolation:
    contract: str
    column: str | None
    rule: str
    count: int
    sample: tuple[Any, ...] = field(default_factory=tuple)


def validate_contract(
    frame: pd.DataFrame,
    contract: TableContract,
    *,
    raise_on_error: bool = True,
) -> list[ContractViolation]:
    violations: list[ContractViolation] = []
    missing = [column for column in contract.columns if column not in frame]
    for column in missing:
        violations.append(
            ContractViolation(contract.name, column, "missing_column", 1)
        )
    if not contract.allow_extra_columns:
        extras = [column for column in frame if column not in contract.columns]
        for column in extras:
            violations.append(
                ContractViolation(contract.name, column, "unexpected_column", 1)
            )
    missing_keys = [column for column in contract.key if column not in frame]
    if not missing_keys and contract.key:
        duplicate = frame.duplicated(list(contract.key), keep=False)
        if duplicate.any():
            violations.append(
                ContractViolation(
                    contract.name,
                    None,
                    "duplicate_key",
                    int(duplicate.sum()),
                    tuple(
                        map(
                            tuple,
                            frame.loc[duplicate, list(contract.key)]
                            .head(5)
                            .itertuples(index=False, name=None),
                        )
                    ),
                )
            )

    for column, rule in contract.columns.items():
        if column not in frame:
            continue
        series = frame[column]
        if rule.dtype_family:
            checks = {
                "numeric": pd_types.is_numeric_dtype,
                "integer": pd_types.is_integer_dtype,
                "float": pd_types.is_float_dtype,
                "boolean": pd_types.is_bool_dtype,
                "datetime": pd_types.is_datetime64_any_dtype,
                "string": pd_types.is_string_dtype,
            }
            check = checks.get(rule.dtype_family)
            if check is None:
                raise ValueError(
                    f"Unknown dtype family {rule.dtype_family!r} in "
                    f"{contract.name}.{column}"
                )
            if not check(series.dtype):
                violations.append(
                    ContractViolation(
                        contract.name,
                        column,
                        f"dtype_family_{rule.dtype_family}",
                        int(len(series)),
                        (str(series.dtype),),
                    )
                )
        if not rule.nullable and series.isna().any():
            violations.append(
                ContractViolation(
                    contract.name,
                    column,
                    "null_not_allowed",
                    int(series.isna().sum()),
                )
            )
        non_null = series.dropna()
        if rule.minimum is not None or rule.maximum is not None:
            numeric = pd.to_numeric(non_null, errors="coerce")
            not_numeric = numeric.isna()
            if not_numeric.any():
                violations.append(
                    ContractViolation(
                        contract.name,
                        column,
                        "numeric_required",
                        int(not_numeric.sum()),
                        tuple(non_null.loc[not_numeric].head(5)),
                    )
                )
        if rule.minimum is not None:
            invalid = numeric < rule.minimum
            if invalid.any():
                violations.append(
                    ContractViolation(
                        contract.name,
                        column,
                        f"minimum_{rule.minimum}",
                        int(invalid.sum()),
                        tuple(non_null.loc[invalid].head(5)),
                    )
                )
        if rule.maximum is not None:
            invalid = numeric > rule.maximum
            if invalid.any():
                violations.append(
                    ContractViolation(
                        contract.name,
                        column,
                        f"maximum_{rule.maximum}",
                        int(invalid.sum()),
                        tuple(non_null.loc[invalid].head(5)),
                    )
                )
        if rule.allowed:
            invalid = ~non_null.isin(rule.allowed)
            if invalid.any():
                violations.append(
                    ContractViolation(
                        contract.name,
                        column,
                        "allowed_values",
                        int(invalid.sum()),
                        tuple(non_null.loc[invalid].head(5)),
                    )
                )

    if violations and raise_on_error:
        summary = "; ".join(
            f"{item.column or '<table>'}:{item.rule} ({item.count})"
            for item in violations
        )
        raise ValueError(f"Contract {contract.name}@{contract.version} failed: {summary}")
    return violations


PLAYER_FIXTURE_CONTRACT = TableContract(
    name="player_fixture_panel",
    version="1.0.0",
    key=("match_id", "player_id"),
    columns={
        "match_id": ColumnContract(nullable=False),
        "player_id": ColumnContract(nullable=False),
        "team_id": ColumnContract(nullable=False),
        "opponent_id": ColumnContract(nullable=False),
        "minutes": ColumnContract(minimum=0, maximum=120),
        "did_not_play": ColumnContract(),
        "as_of_timestamp": ColumnContract(nullable=False),
    },
)


PLAYER_MATCH_FACT_CONTRACT = TableContract(
    name="fact_player_match",
    version="1.0.0",
    key=("match_id", "player_id"),
    columns={
        "match_id": ColumnContract(nullable=False),
        "player_id": ColumnContract(nullable=False),
        "minutes": ColumnContract(minimum=0, maximum=120),
        "xg": ColumnContract(minimum=0),
        "xa": ColumnContract(minimum=0),
        "saves": ColumnContract(minimum=0),
        "tackles": ColumnContract(minimum=0),
        "interceptions": ColumnContract(minimum=0),
        "clearances": ColumnContract(minimum=0),
        "blocks": ColumnContract(minimum=0),
        "recoveries": ColumnContract(minimum=0),
    },
)
